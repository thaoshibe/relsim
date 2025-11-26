import argparse
import json
import os
import random
from datetime import datetime
from typing import Dict, List, Any

import glob
import numpy as np
import torch
import torch.nn.functional as F
import wandb
import yaml
from PIL import Image
from peft import LoraConfig, get_peft_model
from qwen_vl_utils import process_vision_info
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

class Config:
    """Configuration class to load and store all parameters."""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def __getattr__(self, name):
        return self.config.get(name, None)
    
    def get(self, *keys, default=None):
        """Get nested config value using dot notation."""
        value = self.config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return default
        return value if value is not None else default


class AnonymousCaptionDataset(Dataset):
    """Dataset for anonymous caption training."""
    
    def __init__(
        self,
        json_path: str,
        processor: AutoProcessor,
        image_folder: str,
        target_size: tuple,
        prompt_template: str,
        shuffle: bool = True
    ):
        """
        Initialize the dataset.
        
        Args:
            json_path: Path to JSON file containing training data
            processor: Hugging Face processor for the model
            image_folder: Root folder containing images
            target_size: Target image size (width, height)
            prompt_template: Template for the captioning prompt
            shuffle: Whether to shuffle the dataset
        """
        self.processor = processor
        self.image_folder = image_folder
        self.target_size = tuple(target_size)
        self.prompt_template = prompt_template
        self.samples = []
        
        # Load and process data
        self._load_data(json_path)
        
        if shuffle:
            random.shuffle(self.samples)
        
        print(f"Loaded {len(self.samples)} samples from {json_path}")
    
    def _load_data(self, json_path: str):
        """Load data from JSON file."""
        try:
            with open(json_path, "r") as f:
                raw_data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {json_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in {json_path}: {e}")
        
        # Process each entry
        for entry in raw_data:
            index_group = str(entry['index_group'])
            anonymous_caption = entry['anonymous_caption']
            
            # Get all images in this folder
            image_folder_path = os.path.join(self.image_folder, index_group)
            image_paths = glob.glob(os.path.join(image_folder_path, '*.png'))
            
            # Link all images with the caption
            for image_path in image_paths:
                self.samples.append({
                    "image_path": image_path,
                    "label": anonymous_caption,
                })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get a single sample."""
        sample = self.samples[idx]
        img_path = sample["image_path"]
        response = sample["label"]
        
        # Load and resize image
        image = Image.open(img_path).convert("RGB")
        image = image.resize(self.target_size, Image.Resampling.LANCZOS)
        
        # Prepare messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.prompt_template},
                ],
            },
            {
                "role": "assistant",
                "content": response
            }
        ]
        
        # Process inputs
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
        )
        
        # Remove batch dimension for non-special keys
        inputs = {
            k: v[0] if k not in ['pixel_values', 'image_grid_thw'] and v.ndim > 0 else v
            for k, v in inputs.items()
        }
        
        # Create labels (mask instruction part)
        input_ids = inputs["input_ids"].clone()
        labels = input_ids.clone()
        
        # Find assistant response and mask everything before it
        tokenized_response = self.processor.tokenizer.encode(
            response, add_special_tokens=False
        )
        input_ids_list = input_ids.tolist()
        
        for i in range(len(input_ids_list) - len(tokenized_response) + 1):
            if input_ids_list[i:i+len(tokenized_response)] == tokenized_response:
                labels[:i] = -100
                break
        
        # Warn if all labels are masked
        if (labels != -100).sum().item() == 0:
            print(f"⚠️ WARNING: All labels masked for {img_path}")
        
        inputs["labels"] = labels
        return inputs


class SimpleExampleEvalCallback(TrainerCallback):
    """Callback to generate one example output during evaluation."""
    
    def __init__(
        self,
        model,
        processor,
        test_json_path: str,
        image_folder: str,
        target_size: tuple,
        prompt_template: str,
        eval_interval: int,
        generation_config: Dict[str, Any]
    ):
        self.model = model
        self.processor = processor
        self.test_json_path = test_json_path
        self.image_folder = image_folder
        self.target_size = tuple(target_size)
        self.prompt_template = prompt_template
        self.eval_interval = eval_interval
        self.generation_config = generation_config
        
        # Load test data
        with open(test_json_path, "r") as f:
            self.test_data = json.load(f)
        
        # Select a random test image
        self.sample_image_path = self._select_random_image()
    
    def _select_random_image(self) -> str:
        """Select a random image from the test data."""
        if not self.test_data:
            raise ValueError("No test data found")
        
        # Pick a random entry from the list
        random_entry = random.choice(self.test_data)
        random_group = str(random_entry['index_group'])
        
        # Get all images in this group
        image_paths = glob.glob(
            os.path.join(self.image_folder, random_group, '*.png')
        )
        
        if not image_paths:
            raise ValueError(f"No images found in group {random_group}")
        
        # Pick a random image
        return random.choice(image_paths)
    
    def on_step_end(self, args, state, control, **kwargs):
        """Called at the end of each training step."""
        if state.global_step % self.eval_interval == 0 and state.global_step > 0:
            self.run_eval(state.global_step)
    
    def generate_caption(self, image_path: str) -> str:
        """Generate caption for a single image."""
        # Load and resize image
        image = Image.open(image_path).convert("RGB")
        image = image.resize(self.target_size, Image.Resampling.LANCZOS)
        
        # Prepare messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.prompt_template},
                ],
            }
        ]
        
        # Process input
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)
        
        # Generate response
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **self.generation_config)
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            generated_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
        
        return generated_text
    
    def run_eval(self, step: int):
        """Run evaluation at the current step."""
        print(f"\n{'='*60}")
        print(f"🔮 Example Generation at step {step}")
        print(f"{'='*60}")
        
        self.model.eval()
        
        try:
            # Generate caption for the sample image
            caption = self.generate_caption(self.sample_image_path)
            
            # Print result
            print(f"\n📸 Image: {self.sample_image_path}")
            print(f"💬 Generated caption: {caption}")
            print(f"{'='*60}\n")
            
            # Log to wandb using a table for better visualization
            table = wandb.Table(columns=["step", "image", "caption"])
            table.add_data(step, wandb.Image(self.sample_image_path), caption)
            wandb.log({
                "eval/sample_captions": table,
                "step": step,
            })
        
        except Exception as e:
            print(f"❌ Error during evaluation: {e}")
            import traceback
            traceback.print_exc()
        
        self.model.train()


def create_data_collator(pad_token_id: int = 0):
    """Create a data collator function for batching."""
    
    def data_collator(batch):
        """Collate a batch of samples."""
        collated = {}
        
        # Text fields
        for key in ["input_ids", "attention_mask", "labels"]:
            if key in batch[0]:
                vals = [b[key] for b in batch]
                max_len = max(v.shape[0] for v in vals)
                pad_val = -100 if key == "labels" else pad_token_id
                
                collated[key] = torch.stack([
                    F.pad(v, (0, max_len - v.shape[0]), value=pad_val)
                    if v.shape[0] < max_len else v
                    for v in vals
                ])
        
        # Image field
        if "pixel_values" in batch[0]:
            vals = [b["pixel_values"] for b in batch]
            
            if vals[0].ndim == 2:  # (num_patches, hidden_dim)
                collated["pixel_values"] = torch.stack(vals, dim=0)
            elif vals[0].ndim == 4:  # (1, C, H, W)
                collated["pixel_values"] = torch.cat(vals, dim=0)
            elif vals[0].ndim == 3:  # (C, H, W)
                max_h = max(v.shape[1] for v in vals)
                max_w = max(v.shape[2] for v in vals)
                collated["pixel_values"] = torch.stack([
                    F.pad(v, (0, max_w - v.shape[2], 0, max_h - v.shape[1]), value=0)
                    if v.shape[1] < max_h or v.shape[2] < max_w else v
                    for v in vals
                ])
            else:
                raise ValueError(f"Unexpected pixel_values shape: {vals[0].shape}")
        
        # Handle image_grid_thw (Qwen2VL specific)
        if "image_grid_thw" in batch[0]:
            vals = [b["image_grid_thw"] for b in batch]
            collated["image_grid_thw"] = torch.cat(vals, dim=0)
        
        return collated
    
    return data_collator


def setup_wandb(config: Config, run_name: str):
    """Setup Weights & Biases logging."""
    if config.get('wandb', 'enabled', default=True):
        api_key = config.get('wandb', 'api_key')
        if api_key:
            os.environ["WANDB_API_KEY"] = api_key
        
        wandb.init(
            project=config.get('wandb', 'project', default='anonymous-caption'),
            name=run_name,
        )


def load_model_and_processor(config: Config):
    """Load model and processor with LoRA configuration."""
    model_name = config.get('model', 'name')
    
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    
    print("Loading model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=config.get('model', 'torch_dtype', default='auto'),
        device_map=config.get('model', 'device_map', default='auto')
    )
    print("Model and processor loaded successfully!")
    
    # Apply LoRA
    lora_config = LoraConfig(
        r=config.get('lora', 'r', default=16),
        lora_alpha=config.get('lora', 'lora_alpha', default=32),
        target_modules=config.get('lora', 'target_modules', default=["q_proj", "v_proj"]),
        lora_dropout=config.get('lora', 'lora_dropout', default=0.05),
        bias=config.get('lora', 'bias', default='none'),
        task_type=config.get('lora', 'task_type', default='CAUSAL_LM'),
    )
    model = get_peft_model(model, lora_config)
    print(f"✅ LoRA applied. Trainable parameters: {model.print_trainable_parameters()}")
    
    return model, processor


def main():
    """Main training function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen2.5-VL for anonymous captioning"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--train_json",
        type=str,
        default=None,
        help="Override training JSON path from config"
    )
    parser.add_argument(
        "--test_json",
        type=str,
        default=None,
        help="Override test JSON path from config"
    )
    args = parser.parse_args()
    
    # Load configuration
    config = Config(args.config)
    
    # Override with command line args if provided
    train_json = args.train_json or config.get('data', 'train_json')
    test_json = args.test_json or config.get('data', 'test_json')
    
    # Set random seed
    seed = config.get('misc', 'seed', default=42)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create run name
    run_name = "anonymous_caption_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Setup wandb
    setup_wandb(config, run_name)
    
    # Load model and processor
    model, processor = load_model_and_processor(config)
    
    # Create datasets
    print("\nLoading datasets...")
    train_dataset = AnonymousCaptionDataset(
        json_path=train_json,
        processor=processor,
        image_folder=config.get('data', 'train_image_folder'),
        target_size=config.get('image', 'target_size', default=[448, 448]),
        prompt_template=config.get('prompt', 'anonymous_captioning'),
        shuffle=config.get('misc', 'shuffle_data', default=True)
    )
    
    test_dataset = AnonymousCaptionDataset(
        json_path=test_json,
        processor=processor,
        image_folder=config.get('data', 'train_image_folder'),
        target_size=config.get('image', 'target_size', default=[448, 448]),
        prompt_template=config.get('prompt', 'anonymous_captioning'),
        shuffle=False
    )
    
    # Setup output directory
    output_dir = config.get('training', 'output_dir')
    filename = os.path.basename(train_json)
    output_path = f"{output_dir}_{filename}"
    
    # Create training arguments
    training_args = TrainingArguments(
        output_dir=output_path,
        per_device_train_batch_size=config.get('training', 'per_device_train_batch_size', default=32),
        gradient_accumulation_steps=config.get('training', 'gradient_accumulation_steps', default=1),
        learning_rate=config.get('training', 'learning_rate', default=2e-4),
        num_train_epochs=config.get('training', 'num_train_epochs', default=50),
        warmup_steps=config.get('training', 'warmup_steps', default=50),
        logging_steps=config.get('training', 'logging_steps', default=1),
        logging_first_step=config.get('training', 'logging_first_step', default=True),
        save_strategy=config.get('training', 'save_strategy', default='steps'),
        save_steps=config.get('training', 'save_steps', default=100),
        save_total_limit=config.get('training', 'save_total_limit'),
        eval_strategy=config.get('training', 'eval_strategy', default='no'),
        fp16=config.get('training', 'fp16', default=True),
        report_to="wandb" if config.get('wandb', 'enabled', default=True) else "none",
        run_name=run_name,
    )
    
    # Create evaluation callback
    eval_callback = SimpleExampleEvalCallback(
        model=model,
        processor=processor,
        test_json_path=test_json,
        image_folder=config.get('data', 'test_image_folder'),
        target_size=config.get('image', 'target_size', default=[448, 448]),
        prompt_template=config.get('prompt', 'anonymous_captioning'),
        eval_interval=config.get('evaluation', 'interval', default=50),
        generation_config={
            'max_new_tokens': config.get('generation', 'max_new_tokens', default=128),
            'do_sample': config.get('generation', 'do_sample', default=False),
            'temperature': config.get('generation', 'temperature', default=1.0),
        }
    )
    
    # Create data collator
    data_collator_fn = create_data_collator(pad_token_id=processor.tokenizer.pad_token_id or 0)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=processor,
        data_collator=data_collator_fn,
        callbacks=[eval_callback],
    )
    
    # Train
    print("\n🚀 Starting training...")
    trainer.train()
    
    # Save model
    model.save_pretrained(output_path)
    processor.save_pretrained(output_path)
    
    print("\n✅ Training completed!")
    print(f"📁 LoRA adapter saved to: {output_path}")
    print("📊 Check your wandb dashboard for training logs!")
    
    if config.get('wandb', 'enabled', default=True):
        wandb.finish()


if __name__ == "__main__":
    main()

