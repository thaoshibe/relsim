"""
Inference script to compute embeddings for all images in the dataset.
Saves embeddings to disk for retrieval tasks.
"""
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import glob
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from PIL import Image
from flash_s3_dataloader.s3_io import load_s3_image

# Import model, dataset and collate_fn from training script
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_score_model import QwenWithQueryToken, ImageTextDataset, create_collate_fn

# =======================
# 0. Arguments
# =======================
parser = argparse.ArgumentParser(description="Compute embeddings for all images")
parser.add_argument(
    "--checkpoint_dir",
    type=str,
    required=True,
    help="Path to trained model checkpoint"
)
parser.add_argument(
    "--json_file",
    type=str,
    required=True,
    help="JSON file to load"
)
parser.add_argument(
    "--output_path",
    type=str,
    default="/mnt/localssd/embeddings.npz",
    help="Output file path for embeddings"
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=64,
    help="Batch size for inference"
)
parser.add_argument(
    "--total_images",
    type=int,
    default=None,
    help="Maximum number of images to process (default: None, process all)"
)
args = parser.parse_args()

print(f"Loading checkpoint from: {args.checkpoint_dir}")
print(f"Output path: {args.output_path}")
os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

# =======================
# 1. Load Model
# =======================
print("\n" + "="*60)
print("Loading model...")
print("="*60)

# Load processor (includes query token from training)
processor = AutoProcessor.from_pretrained(args.checkpoint_dir, trust_remote_code=True)

# Load base model
base_model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    base_model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Resize token embeddings to match the processor (with added query token)
print(f"Resizing token embeddings: {len(processor.tokenizer)} tokens")
base_model.resize_token_embeddings(len(processor.tokenizer))

# Load LoRA weights
print("Loading LoRA weights...")
base_model = PeftModel.from_pretrained(base_model, args.checkpoint_dir)

# Wrap with query token
model = QwenWithQueryToken(base_model, processor, hidden_size=3584)

# Load projection head
print("Loading projection head...")
checkpoint = torch.load(
    os.path.join(args.checkpoint_dir, "projection_and_config.pt"),
    map_location="cpu"
)
model.projection.load_state_dict(checkpoint['projection'])

# Move projection to correct device and dtype
try:
    param = next(base_model.parameters())
    model.projection = model.projection.to(device=param.device, dtype=param.dtype)
    print(f"✅ Projection head loaded: device={param.device}, dtype={param.dtype}")
except StopIteration:
    pass

model.eval()
print("✅ Model loaded successfully!")

# =======================
# 2. Load Dataset
# =======================
print("\n" + "="*60)
print("Loading dataset...")
print("="*60)
# Don't shuffle for inference
dataset = ImageTextDataset(
    [args.json_file],  # Wrap in list since ImageTextDataset expects a list
    processor, 
    shuffle=False, 
    max_samples=args.total_images
)

# Create collate function
collate_fn = create_collate_fn(processor)

dataloader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=False,  # Don't shuffle for inference
    collate_fn=collate_fn,
    num_workers=4,
    pin_memory=True
)

# =======================
# 3. Compute Embeddings
# =======================
print("\n" + "="*60)
print("Computing embeddings...")
print("="*60)

all_embeddings = []
all_image_ids = []
all_captions = []
all_s3_paths = []

with torch.no_grad():
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Computing embeddings")):
        # Move to device
        input_ids = batch["input_ids"].cuda()
        attention_mask = batch["attention_mask"].cuda()
        pixel_values = batch["pixel_values"].cuda()
        image_grid_thw = batch["image_grid_thw"].cuda()
        
        # Get embeddings
        embeddings = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw
        )
        
        # Convert to float32 and normalize
        embeddings = embeddings.float()
        embeddings = F.normalize(embeddings, dim=-1)
        
        # Move to CPU and store
        embeddings_np = embeddings.cpu().numpy()
        all_embeddings.append(embeddings_np)
        all_image_ids.extend(batch["image_ids"])
        all_captions.extend(batch["captions"])
        all_s3_paths.extend(batch["s3_paths"])

# Concatenate all embeddings
all_embeddings = np.concatenate(all_embeddings, axis=0)
print(f"✅ Computed embeddings for {len(all_image_ids)} images")
print(f"   Embedding shape: {all_embeddings.shape}")

# =======================
# 4. Save Embeddings
# =======================
print("\n" + "="*60)
print("Saving embeddings...")
print("="*60)

# Save as numpy
np.savez_compressed(
    args.output_path,
    embeddings=all_embeddings,
    image_ids=np.array(all_image_ids),
    captions=np.array(all_captions),
    s3_paths=np.array(all_s3_paths)
)
print(f"✅ Saved embeddings to: {args.output_path}")

print("\n" + "="*60)
print("🎉 Done!")
print("="*60)
print(f"Total images processed: {len(all_image_ids)}")
print(f"Embedding dimension: {all_embeddings.shape[1]}")
print(f"Output file: {args.output_path}")

