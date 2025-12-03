"""
Inference script to compute baseline embeddings (CLIP, DINOv2, DreamSim) for all images in the dataset.
Saves embeddings to disk for retrieval tasks.

Can also load trained ablation models (CLIP/DINOv2 with LoRA + projection head) by providing checkpoint paths.
"""
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import os
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel, AutoImageProcessor, AutoModel
from PIL import Image
from dreamsim import dreamsim
from peft import PeftModel
from utils import get_text_embedding_from_image

# =======================
# -1. Projection Head (for trained models)
# =======================
class ProjectionHead(nn.Module):
    """Projection head used in ablation training"""
    def __init__(self, input_dim, output_dim=384):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.projection(x)

# =======================
# 0. Arguments
# =======================
parser = argparse.ArgumentParser(description="Compute baseline embeddings for all images")
parser.add_argument(
    "--model_type",
    type=str,
    required=True,
    choices=["clip", "dino", "dreamsim", "qwen_text"],
    help="Model type to use: 'clip', 'dino', 'dreamsim', or 'qwen_text'"
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
    required=True,
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
parser.add_argument(
    "--clip_ablation_ckpt",
    type=str,
    default=None,
    help="Path to trained CLIP checkpoint (for ablation models)"
)
parser.add_argument(
    "--dino_ablation_ckpt",
    type=str,
    default=None,
    help="Path to trained DINOv2 checkpoint (for ablation models)"
)
args = parser.parse_args()

print(f"Model type: {args.model_type}")
print(f"Output path: {args.output_path}")
os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

# =======================
# 1. Load Model
# =======================
print("\n" + "="*60)
print(f"Loading {args.model_type.upper()} model...")
print("="*60)

device = "cuda" if torch.cuda.is_available() else "cpu"
projection_head = None  # Will be set if using trained model

if args.model_type == "clip":
    # Load CLIP model
    clip_model_name = "openai/clip-vit-large-patch14"
    
    if args.clip_ablation_ckpt:
        # Load trained CLIP model with LoRA
        print(f"Loading trained CLIP model from: {args.clip_ablation_ckpt}")
        
        # Load base model
        model = CLIPModel.from_pretrained(clip_model_name)
        
        # Load LoRA weights
        model.vision_model = PeftModel.from_pretrained(model.vision_model, args.clip_ablation_ckpt)
        model = model.to(device)
        model.eval()
        
        # Load projection head
        proj_config = torch.load(os.path.join(args.clip_ablation_ckpt, "projection_and_config.pt"))
        projection_head = ProjectionHead(input_dim=768, output_dim=384)
        projection_head.projection.load_state_dict(proj_config['projection'])
        projection_head = projection_head.to(device)
        projection_head.eval()
        
        processor = CLIPProcessor.from_pretrained(args.clip_ablation_ckpt)
        print(f"✅ Trained CLIP model loaded with LoRA and projection head")
        print(f"   Device: {device}")
        print(f"   Output dimension: 384 (with projection)")
    else:
        # Load pretrained CLIP model
        model = CLIPModel.from_pretrained(clip_model_name)
        processor = CLIPProcessor.from_pretrained(clip_model_name)
        model = model.to(device)
        model.eval()
        print(f"✅ CLIP model loaded: {clip_model_name}")
        print(f"   Device: {device}")
    
elif args.model_type == "dino":
    # Load DINOv2 model
    dino_model_name = "facebook/dinov2-large"
    
    if args.dino_ablation_ckpt:
        # Load trained DINOv2 model with LoRA
        print(f"Loading trained DINOv2 model from: {args.dino_ablation_ckpt}")
        
        # Load base model
        model = AutoModel.from_pretrained(dino_model_name)
        
        # Load LoRA weights
        model = PeftModel.from_pretrained(model, args.dino_ablation_ckpt)
        model = model.to(device)
        model.eval()
        
        # Load projection head
        proj_config = torch.load(os.path.join(args.dino_ablation_ckpt, "projection_and_config.pt"))
        projection_head = ProjectionHead(input_dim=1024, output_dim=384)
        projection_head.projection.load_state_dict(proj_config['projection'])
        projection_head = projection_head.to(device)
        projection_head.eval()
        
        processor = AutoImageProcessor.from_pretrained(args.dino_ablation_ckpt)
        print(f"✅ Trained DINOv2 model loaded with LoRA and projection head")
        print(f"   Device: {device}")
        print(f"   Output dimension: 384 (with projection)")
    else:
        # Load pretrained DINOv2 model
        model = AutoModel.from_pretrained(dino_model_name)
        processor = AutoImageProcessor.from_pretrained(dino_model_name)
        model = model.to(device)
        model.eval()
        
        print(f"✅ DINOv2 model loaded: {dino_model_name}")
        print(f"   Device: {device}")
    
elif args.model_type == "dreamsim":
    # Load DreamSim model
    model, processor = dreamsim(pretrained=True, device=device)
    model.eval()
    
    print(f"✅ DreamSim model loaded")
    print(f"   Device: {device}")
    
elif args.model_type == "qwen_text":
    # For qwen_text, models will be loaded lazily in utils.py
    model = None
    processor = None
    print(f"✅ Qwen-text mode (Qwen caption + CLIP text)")
    print(f"   Models will be loaded on first use")

# =======================
# 2. Simple Dataset Wrapper
# =======================
class SimpleImageDataset(Dataset):
    """Simple wrapper that loads images and returns raw PIL images"""
    
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
        
        self.corrupted_count = 0
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
            'url_link': url_link,
        }

def collate_fn_clip(batch):
    """Collate function for CLIP"""
    images = [item['image'] for item in batch]
    image_ids = [item['image_id'] for item in batch]
    captions = [item['caption'] for item in batch]
    url_links = [item['url_link'] for item in batch]
    
    # Process images with CLIP processor
    inputs = processor(images=images, return_tensors="pt", padding=True)
    
    return {
        'pixel_values': inputs.pixel_values,
        'image_ids': image_ids,
        'captions': captions,
        'url_links': url_links,
    }

def collate_fn_dino(batch):
    """Collate function for DINOv2"""
    images = [item['image'] for item in batch]
    image_ids = [item['image_id'] for item in batch]
    captions = [item['caption'] for item in batch]
    url_links = [item['url_link'] for item in batch]
    
    # Process images with DINOv2 processor
    inputs = processor(images=images, return_tensors="pt")
    
    return {
        'pixel_values': inputs.pixel_values,
        'image_ids': image_ids,
        'captions': captions,
        'url_links': url_links,
    }

def collate_fn_dreamsim(batch):
    """Collate function for DreamSim"""
    images = [item['image'] for item in batch]
    image_ids = [item['image_id'] for item in batch]
    captions = [item['caption'] for item in batch]
    url_links = [item['url_link'] for item in batch]
    
    # Process images with DreamSim processor
    # processor returns (C, H, W), we stack to get (B, C, H, W)
    processed_images = []
    for img in images:
        processed = processor(img)
        # Ensure it's 3D (C, H, W), squeeze if needed
        if processed.dim() == 4:
            processed = processed.squeeze(0)
        processed_images.append(processed)
    
    pixel_values = torch.stack(processed_images)
    
    return {
        'pixel_values': pixel_values,
        'image_ids': image_ids,
        'captions': captions,
        'url_links': url_links,
    }

# =======================
# 3. Load Dataset
# =======================
print("\n" + "="*60)
print("Loading dataset...")
print("="*60)

dataset = SimpleImageDataset(args.json_file, max_samples=args.total_images)

# Choose collate function based on model type
if args.model_type == "clip":
    collate_fn = collate_fn_clip
elif args.model_type == "dino":
    collate_fn = collate_fn_dino
elif args.model_type == "dreamsim":
    collate_fn = collate_fn_dreamsim
elif args.model_type == "qwen_text":
    # For qwen_text, we don't batch process, so we use a simple collate
    def collate_fn_identity(batch):
        return batch
    collate_fn = collate_fn_identity

# For qwen_text, use smaller batch and no multiprocessing to avoid CUDA issues
if args.model_type == "qwen_text":
    dataloader = DataLoader(
        dataset,
        batch_size=1,  # Process one at a time for qwen_text
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False
    )
else:
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )

# =======================
# 4. Compute Embeddings
# =======================
print("\n" + "="*60)
print(f"Computing {args.model_type.upper()} embeddings...")
print("="*60)

all_embeddings = []
all_image_ids = []
all_captions = []
all_urls = []

with torch.no_grad():
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Computing embeddings")):
        if args.model_type == "qwen_text":
            # Special handling for qwen_text: process each item individually
            for item in batch:
                image = item['image']
                image_id = item['image_id']
                caption = item['caption']
                
                # Get embedding using Qwen caption + CLIP text
                embedding = get_text_embedding_from_image(image)
                embedding = embedding.flatten()  # Shape: (embedding_dim,)
                
                all_embeddings.append(embedding[np.newaxis, :])  # Add batch dim
                all_image_ids.append(image_id)
                all_captions.append(caption)
        else:
            # Standard batch processing for other models
            # Move to device
            pixel_values = batch["pixel_values"].to(device)
            
            if args.model_type == "clip":
                # Get CLIP image embeddings
                image_features = model.get_image_features(pixel_values=pixel_values)
                
                # Apply projection head if using trained model
                if projection_head is not None:
                    embeddings = projection_head(image_features)
                else:
                    embeddings = image_features
                
                # Normalize embeddings
                embeddings = F.normalize(embeddings, dim=-1)
                
            elif args.model_type == "dino":
                # Get DINOv2 embeddings
                outputs = model(pixel_values=pixel_values)
                # Use the [CLS] token embedding
                image_features = outputs.last_hidden_state[:, 0]  # Shape: (batch_size, hidden_dim)
                
                # Apply projection head if using trained model
                if projection_head is not None:
                    embeddings = projection_head(image_features)
                else:
                    embeddings = image_features
                
                # Normalize
                embeddings = F.normalize(embeddings, dim=-1)
                
            elif args.model_type == "dreamsim":
                # Get DreamSim embeddings
                # pixel_values shape: (B, C, H, W)
                embeddings = model.embed(pixel_values)
                # embeddings might need reshaping if it has extra dimensions
                if embeddings.dim() > 2:
                    embeddings = embeddings.view(embeddings.size(0), -1)
                # Normalize
                embeddings = F.normalize(embeddings, dim=-1)
            
            # Move to CPU and store
            embeddings_np = embeddings.cpu().float().numpy()
            all_embeddings.append(embeddings_np)
            all_image_ids.extend(batch["image_ids"])
            all_captions.extend(batch["captions"])
            all_urls.extend(batch["url_links"])

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
    captions=np.array(all_captions),
    url_links=np.array(all_urls),
    model_type=args.model_type
)
print(f"✅ Saved embeddings to: {args.output_path}")

print("\n" + "="*60)
print("🎉 Done!")
print("="*60)
print(f"Model type: {args.model_type.upper()}")
print(f"Total images processed: {len(all_image_ids)}")
print(f"Embedding dimension: {all_embeddings.shape[1]}")
print(f"Output file: {args.output_path}")

