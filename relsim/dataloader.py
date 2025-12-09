import json
import torch
import torch.nn.functional as F
import os
import random
from torch.utils.data import Dataset
from qwen_vl_utils import process_vision_info
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

class ImageTextDataset(Dataset):
    def __init__(self, json_files, processor, image_folder, shuffle=True, max_samples=None):
        """
        Load and merge specified JSONL files.
        Expected JSONL format (one JSON per line): {"image_hash": "...", "caption": "...", "url_link": "..."}
        
        Args:
            json_files: List of JSONL file paths to load
            processor: Qwen processor for image/text processing
            image_folder: Path to folder containing images (e.g., "/path/to/images/")
            shuffle: Whether to shuffle the data (default: True for training)
            max_samples: Maximum number of samples to use (default: None, use all)

        COMMENT: now all images are center cropped to square and resized to 448x448 ~~ Think about how to improve this later~
        """
        self.processor = processor
        self.image_folder = image_folder
        self.max_samples = max_samples
        
        # Load and merge all specified JSONL files
        print(f"Loading {len(json_files)} JSONL file(s)...")
        
        self.samples = []
        for json_file in json_files:
            print(f"  Loading {json_file}...")
            with open(json_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:  # Skip empty lines
                        data = json.loads(line)
                        image_hash = data["image_hash"]
                        caption = data["caption"]
                        self.samples.append((image_hash, caption))
        
        print(f"Total loaded: {len(self.samples)} samples (before shuffle/limit)")
        
        # Shuffle the data if requested
        if shuffle:
            random.shuffle(self.samples)
        
        # Limit samples if max_samples is specified
        if max_samples is not None and max_samples < len(self.samples):
            original_count = len(self.samples)
            self.samples = self.samples[:max_samples]
            print(f"Limited to {len(self.samples)} samples (from {original_count})")
        else:
            status = "(shuffled)" if shuffle else ""
            print(f"Using all {len(self.samples)} samples {status}")
        
        # Counter for corrupted images
        self.corrupted_count = 0
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image_hash, caption = self.samples[idx]
        
        # Construct local image path
        image_path = os.path.join(self.image_folder, f"{image_hash}.png")
        
        try:
            # Load image from local disk
            image = Image.open(image_path)
            
            # Load the image fully to catch truncation errors early
            image.load()
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Center crop to square then resize (preserve aspect ratio)
            width, height = image.size
            min_dim = min(width, height)
            
            # Center crop to square
            left = (width - min_dim) // 2
            top = (height - min_dim) // 2
            right = left + min_dim
            bottom = top + min_dim
            image = image.crop((left, top, right, bottom))

        except (OSError, Exception) as e:
            self.corrupted_count += 1
            # Print out duplicated messages to catch attention :D
            print("="*100)
            print("="*100)
            print("="*100)
            print('ATTENTION ATTENTION ATTENTION')
            print(f"Warning: Failed to load image {image_hash} from {image_path}: {e} (corrupted: {self.corrupted_count})")
            print(f"Warning: Failed to load image {image_hash} from {image_path}: {e} (corrupted: {self.corrupted_count})")
            print(f"Warning: Failed to load image {image_hash} from {image_path}: {e} (corrupted: {self.corrupted_count})")
            print(f"Warning: Failed to load image {image_hash} from {image_path}: {e} (corrupted: {self.corrupted_count})")
            # Return a black placeholder image if loading fails
            image = Image.new('RGB', (448, 448), color='black')
        
        # Resize to target size
        target_size = (448, 448)
        image = image.resize(target_size, Image.Resampling.LANCZOS)
        
        # Prepare minimal messages for image processing
        # Add query token after image to extract features
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "<|query|>"},  # Query token to extract features
                ],
            }
        ]
        
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
        
        # Remove batch dimension
        inputs = {
            k: v[0] if k not in ['pixel_values', 'image_grid_thw'] and v.ndim > 0 else v
            for k, v in inputs.items()
        }
        
        return {
            "inputs": inputs,
            "caption": caption,
            "image_id": image_hash,
            "image_path": image_path
        }


def create_collate_fn(processor):
    """Create collate function with processor."""
    def collate_fn(batch):
        """Custom collate function to handle variable-sized inputs."""
        captions = [item["caption"] for item in batch]
        image_ids = [item["image_id"] for item in batch]
        image_paths = [item["image_path"] for item in batch]
        
        # Collate inputs
        input_ids = [item["inputs"]["input_ids"] for item in batch]
        attention_mask = [item["inputs"]["attention_mask"] for item in batch]
        
        # Pad sequences
        max_len = max(ids.shape[0] for ids in input_ids)
        input_ids_padded = torch.stack([
            F.pad(ids, (0, max_len - ids.shape[0]), value=processor.tokenizer.pad_token_id)
            for ids in input_ids
        ])
        attention_mask_padded = torch.stack([
            F.pad(mask, (0, max_len - mask.shape[0]), value=0)
            for mask in attention_mask
        ])
        
        # Handle pixel_values and image_grid_thw
        pixel_values = torch.cat([item["inputs"]["pixel_values"] for item in batch], dim=0)
        image_grid_thw = torch.cat([item["inputs"]["image_grid_thw"] for item in batch], dim=0)
        
        return {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask_padded,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "captions": captions,
            "image_ids": image_ids,
            "image_paths": image_paths
        }
    return collate_fn

if __name__ == "__main__":

    #########################################################
    #
    #          DEBUG: Without the <|query|> token
    #
    #########################################################

    processor_name = "Qwen/Qwen2.5-VL-7B-Instruct"
    processor = AutoProcessor.from_pretrained(processor_name)
    dataset = ImageTextDataset(json_files=["./data/anonymous_captions_train.jsonl"], processor=processor, image_folder="../data/anonymous_captions_train_images")
    print(dataset[0])
    input_ids = dataset[0]["inputs"]["input_ids"]
    print(processor.decode(input_ids))

    # from train_score_model import QwenWithQueryToken
    # # Load processor first, then model
    # model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
    # processor = AutoProcessor.from_pretrained(model_name)
    
    # base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    #     model_name,
    #     torch_dtype=torch.bfloat16,
    #     device_map="auto"
    # )

    # model = QwenWithQueryToken(base_model, processor, hidden_size=3584)
    # dataset = ImageTextDataset(json_files=["./data/anonymous_captions_train.jsonl"], processor=processor, image_folder="../data/anonymous_captions_train_images")
    # print(dataset[0])
    # # test to decode one input_ids
    # input_ids = dataset[0]["inputs"]["input_ids"]
    # print('<|query|> id: ', model.query_token_id)
    # print(processor.decode(input_ids))

    ##### We should see 151665 as the <|query|> id