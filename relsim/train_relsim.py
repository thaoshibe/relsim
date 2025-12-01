"""
Train a score model using Qwen VLM (with LoRA) and sentence transformers.
Uses contrastive learning to align image features from Qwen with text embeddings from sentence transformers.
"""
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.utils.data import DataLoader
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)
from peft import LoraConfig, get_peft_model
from datetime import datetime
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import wandb

# Import dataset and collate function from separate module
from dataloader import ImageTextDataset, create_collate_fn
from relsim_model import QwenWithQueryToken

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train score model with Qwen VLM and sentence transformers")
    parser.add_argument(
        "--json_files",
        type=str,
        nargs='+',
        required=True,
        help="List of JSONL files to load and merge (space-separated)"
    )
    parser.add_argument(
        "--image_folder",
        type=str,
        required=True,
        help="Path to folder containing images (e.g., /path/to/images/)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./ckpt",
        help="Base output directory for checkpoints (run_name will be appended)"
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Run name for this experiment (default: auto-generated with timestamp)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for training"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.07,
        help="Temperature for contrastive loss"
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every N steps"
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Log metrics every N steps"
    )
    args = parser.parse_args()
    
    # Generate run_name if not provided
    if args.run_name is None:
        run_name = f'score_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    else:
        run_name = args.run_name
    
    # Append run_name to output_dir
    args.output_dir = os.path.join(args.output_dir, run_name)
    print(f"📂 Output directory: {args.output_dir}")
    wandb.init(
        project="relsim",
        name=run_name,
        config=vars(args)
    )
    
    # =======================
    # 1. Load Models
    # =======================
    print("Loading Qwen model and processor...")
    model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
    processor = AutoProcessor.from_pretrained(model_name)

    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # Apply LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    base_model = get_peft_model(base_model, lora_config)
    print(f"✅ LoRA applied. Trainable parameters: {base_model.print_trainable_parameters()}")
    
    # Wrap with query token (must be done AFTER LoRA but BEFORE creating dataset)
    # Note: base_model already on GPU via device_map="auto", so don't call .cuda() again
    model = QwenWithQueryToken(base_model, processor, hidden_size=3584)
    
    print("Loading sentence transformer (frozen)...")
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    sentence_model = sentence_model.cuda()
    sentence_model.eval()
    # Freeze sentence transformer
    for param in sentence_model.parameters():
        param.requires_grad = False
    
    print("Models loaded successfully!")
    
    # Verify that processor has the query token
    query_token_in_processor = '<|query|>' in processor.tokenizer.get_vocab()
    print(f"🔍 Query token '<|query|>' in processor vocabulary: {query_token_in_processor}")
    if not query_token_in_processor:
        raise ValueError("Query token not found in processor! This is a bug.")
    
    dataset = ImageTextDataset(args.json_files, processor, args.image_folder)
    collate_fn = create_collate_fn(processor)
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=0.01
    )
    
    num_training_steps = len(dataloader) * args.num_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_training_steps)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n🚀 Starting training...")
    global_step = 0
    
    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for batch_idx, batch in enumerate(pbar):
            # Move inputs to device
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            pixel_values = batch["pixel_values"].cuda()
            image_grid_thw = batch["image_grid_thw"].cuda()
            captions = batch["captions"]
            
    
            # Get image embeddings from Qwen
            image_emb = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw
            )  # [batch_size, 384]
            
            # Get text embeddings from sentence transformer (frozen)
            with torch.no_grad():
                text_emb = sentence_model.encode(
                    captions,
                    convert_to_tensor=True,
                    show_progress_bar=False
                )  # [batch_size, 384]
            
            # Convert to float32 for loss computation
            image_emb = image_emb.float()
            text_emb = text_emb.float()
            
            # Normalize embeddings
            image_emb = F.normalize(image_emb, dim=-1)
            text_emb = F.normalize(text_emb, dim=-1)
            
            # Contrastive loss
            temperature = args.temperature
            logits = image_emb @ text_emb.T / temperature  # [batch_size, batch_size]
            labels = torch.arange(len(logits), device=logits.device)
            
            # Row-wise cross-entropy (not symmetric)
            loss = F.cross_entropy(logits, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            # Logging
            epoch_loss += loss.item()
            global_step += 1
            
            if global_step % args.logging_steps == 0:
                avg_loss = epoch_loss / (batch_idx + 1)
                # Calculate accuracy (how many pairs are correctly matched)
                with torch.no_grad():
                    predictions = logits.argmax(dim=1)
                    accuracy = (predictions == labels).float().mean().item()
                
                wandb.log({
                    "train/loss": loss.item(),
                    "train/epoch_loss": avg_loss,
                    "train/accuracy": accuracy,
                    "train/learning_rate": scheduler.get_last_lr()[0],
                    "train/epoch": epoch,
                    # "step": global_step
                })
                
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "acc": f"{accuracy:.4f}"
                })
            
            # Save checkpoint
            if global_step % args.save_steps == 0:
                checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                os.makedirs(checkpoint_dir, exist_ok=True)
                
                # Save LoRA weights
                model.base_model.save_pretrained(checkpoint_dir)
                # Save projection head and query token info
                torch.save({
                    'projection': model.projection.state_dict(),
                    'query_token_id': model.query_token_id,
                    'global_step': global_step,
                    'epoch': epoch,
                }, os.path.join(checkpoint_dir, "projection_and_config.pt"))
                processor.save_pretrained(checkpoint_dir)
                
                print(f"\n💾 Checkpoint saved to {checkpoint_dir}")
        
        print(f"\n📊 Epoch {epoch+1} completed. Average loss: {epoch_loss / len(dataloader):.4f}")
    
    final_dir = os.path.join(args.output_dir, "final_model")
    os.makedirs(final_dir, exist_ok=True)
    
    model.base_model.save_pretrained(final_dir)
    torch.save({
        'projection': model.projection.state_dict(),
        'query_token_id': model.query_token_id,
    }, os.path.join(final_dir, "projection_and_config.pt"))
    processor.save_pretrained(final_dir)
    
    print("\n✅ Training completed!")
    print(f"📁 Final model saved to: {final_dir}")
    print("📊 Check your wandb dashboard for training logs!")
    
    wandb.finish()