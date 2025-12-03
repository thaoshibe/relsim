import argparse
import json
import torch
import torch.nn.functional as F
import os
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from relsim.relsim_score import relsim

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
# 1. Simple Dataset Wrapper
# =======================
class SimpleImageDataset(Dataset):
    """Simple wrapper that loads images from local folder based on JSONL file"""
    
    def __init__(self, json_file, max_samples=None):
        print(f"Loading {json_file}...")
        # load jsonl file
        data = []
        with open(json_file, "r") as f:
            for line in f:
                data.append(json.loads(line))

        # the loaded data is the list, we need to convert it to a list of tuples
        self.samples = [(info["image_hash"], info["caption"], info["url_link"]) for info in data]
        if max_samples is not None and max_samples < len(self.samples):
            self.samples = self.samples[:max_samples]
            print(f"Total loaded: {len(self.samples)} samples (limited from {len(data)})")
        else:
            print(f"Total loaded: {len(self.samples)} samples")
        
        self.IMAGE_FOLDER = '../data/anonymous_captions_test_images'
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image_id, caption, url_link = self.samples[idx]
        image = Image.open(os.path.join(self.IMAGE_FOLDER, f"{image_id}.png"))

        return {
            'image': image,
            'image_id': image_id,
            'caption': caption,
        }


def collate_fn(batch):
    """Simple collate function that keeps PIL images"""
    images = [item['image'] for item in batch]
    image_ids = [item['image_id'] for item in batch]
    captions = [item['caption'] for item in batch]
    
    return {
        'images': images,
        'image_ids': image_ids,
        'captions': captions,
    }


# =======================
# 2. Load Model
# =======================
print("\n" + "="*60)
print("Loading model...")
print("="*60)

model, preprocess = relsim(pretrained=True, checkpoint_dir=args.checkpoint_dir)
print("✅ Model loaded successfully!")

# =======================
# 3. Load Dataset
# =======================
print("\n" + "="*60)
print("Loading dataset...")
print("="*60)

dataset = SimpleImageDataset(args.json_file, max_samples=args.total_images)

dataloader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=False,  # Don't shuffle for inference
    collate_fn=collate_fn,
    num_workers=4,
    pin_memory=False
)

# =======================
# 4. Compute Embeddings
# =======================
print("\n" + "="*60)
print("Computing embeddings...")
print("="*60)

all_embeddings = []
all_image_ids = []
all_captions = []

with torch.no_grad():
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Computing embeddings")):
        images = batch["images"]
        
        # Get embeddings for each image in batch
        batch_embeddings = []
        for img in images:
            embedding = model.embed(img)
            # Normalize embedding
            embedding = F.normalize(embedding, dim=-1)
            batch_embeddings.append(embedding.cpu().float().numpy())
        
        # Stack batch embeddings
        embeddings_np = np.concatenate(batch_embeddings, axis=0)
        all_embeddings.append(embeddings_np)
        all_image_ids.extend(batch["image_ids"])
        all_captions.extend(batch["captions"])

# Concatenate all embeddings
all_embeddings = np.concatenate(all_embeddings, axis=0)
print(f"✅ Computed embeddings for {len(all_image_ids)} images")
print(f"   Embedding shape: {all_embeddings.shape}")

# =======================
# 5. Save Embeddings
# =======================
print("\n" + "="*60)
print("Saving embeddings...")
print("="*60)

# Save as numpy
np.savez_compressed(
    args.output_path,
    embeddings=all_embeddings,
    image_ids=np.array(all_image_ids),
    captions=np.array(all_captions)
)
print(f"✅ Saved embeddings to: {args.output_path}")

print("\n" + "="*60)
print("🎉 Done!")
print("="*60)
print(f"Total images processed: {len(all_image_ids)}")
print(f"Embedding dimension: {all_embeddings.shape[1]}")
print(f"Output file: {args.output_path}")
