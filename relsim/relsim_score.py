import torch
import torch.nn.functional as F
import os
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from PIL import Image
from huggingface_hub import hf_hub_download

# Import model from training script
import sys
# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from .relsim_model import QwenWithQueryToken

class relsim:
    """RelSim perceptual similarity model"""
    
    def __init__(self, base_model, processor):
        self.base_model = base_model
        self.processor = processor
        self.device = next(base_model.parameters()).device
    
    def embed(self, img):
        """
        Extract embedding from a preprocessed image.
        
        Args:
            img: Preprocessed PIL Image
            
        Returns:
            Normalized embedding tensor
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": "<|query|>"}
                ]
            }
        ]
        
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        with torch.no_grad():
            embedding = self.base_model(**inputs)
            embedding = embedding.float()
            embedding = F.normalize(embedding, dim=-1)
        
        return embedding
    
    def __call__(self, img1, img2):
        """
        Compute perceptual similarity between two preprocessed images.
        
        Args:
            img1: First preprocessed PIL Image
            img2: Second preprocessed PIL Image
            
        Returns:
            Similarity score (higher = more similar)
        """
        embedding1 = self.embed(img1)
        embedding2 = self.embed(img2)
        
        # Cosine similarity (embeddings are normalized, so it's dot product)
        similarity = (embedding1 * embedding2).sum().item()
        
        return similarity
    
    def eval(self):
        """Set model to evaluation mode"""
        self.base_model.eval()
        return self


def preprocess_image(img):
    """
    Preprocess function for images.
    
    Args:
        img: PIL Image
        
    Returns:
        Preprocessed PIL Image in RGB format
    """
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    else:
        raise TypeError("Input must be a PIL Image")


# Store reference to class before function shadows the name
_RelSimClass = relsim


def relsim(pretrained=True, checkpoint_dir=None, cache_dir=None):
    """
    Load RelSim model and preprocessing function.
    
    Args:
        pretrained: If True, load pretrained model (requires checkpoint_dir)
        checkpoint_dir: Path to trained model checkpoint OR HuggingFace model ID
                       Examples: "path/to/checkpoint" or "thaoshibe/relsim-qwenvl25-lora"
        cache_dir: Cache directory for model files
        
    Returns:
        model: relsim instance
        preprocess: Preprocessing function
        
    Example:
        >>> from relsim import relsim
        >>> from PIL import Image
        >>> 
        >>> # Load from HuggingFace
        >>> model, preprocess = relsim(pretrained=True, checkpoint_dir="thaoshibe/relsim-qwenvl25-lora")
        >>> 
        >>> # Or load from local path
        >>> model, preprocess = relsim(pretrained=True, checkpoint_dir="path/to/checkpoint")
        >>> 
        >>> # Feature extraction
        >>> img1 = preprocess(Image.open("img1.jpg"))
        >>> embedding = model.embed(img1)
        >>> 
        >>> # Perceptual similarity
        >>> img1 = preprocess(Image.open("img1.jpg"))
        >>> img2 = preprocess(Image.open("img2.jpg"))
        >>> distance = model(img1, img2)
    """
    if not pretrained:
        raise ValueError("Only pretrained=True is supported")
    
    if checkpoint_dir is None:
        raise ValueError("checkpoint_dir must be provided")
    
    print(f"Loading checkpoint from: {checkpoint_dir}")
    
    # Load processor (includes query token from training)
    # Works with both local paths and HuggingFace model IDs
    processor = AutoProcessor.from_pretrained(
        checkpoint_dir, 
        trust_remote_code=True,
        cache_dir=cache_dir
    )
    
    # Load base model
    base_model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
    print(f"Loading base model: {base_model_name}")
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=cache_dir
    )
    
    # Resize token embeddings to match the processor (with added query token)
    print(f"Resizing token embeddings: {len(processor.tokenizer)} tokens")
    base_model.resize_token_embeddings(len(processor.tokenizer))
    
    # Load LoRA weights
    print("Loading LoRA weights...")
    # Load adapter weights to CPU first to avoid device issues
    # Works with both local paths and HuggingFace model IDs
    base_model = PeftModel.from_pretrained(
        base_model, 
        checkpoint_dir,
        torch_device="cpu"
    )
    
    # Wrap with query token
    wrapped_model = QwenWithQueryToken(base_model, processor, hidden_size=3584)
    
    # Load projection head
    print("Loading projection head...")
    # For HuggingFace models, this will be downloaded to cache
    # For local paths, this will be loaded directly
    
    # Check if it's a HuggingFace model ID (contains '/') or local path
    if '/' in checkpoint_dir and not os.path.exists(checkpoint_dir):
        # It's a HuggingFace model ID
        projection_path = hf_hub_download(
            repo_id=checkpoint_dir,
            filename="projection_and_config.pt",
            cache_dir=cache_dir
        )
    else:
        # It's a local path
        projection_path = os.path.join(checkpoint_dir, "projection_and_config.pt")
    
    checkpoint = torch.load(projection_path, map_location="cpu")
    wrapped_model.projection.load_state_dict(checkpoint['projection'])
    
    # Move projection to correct device and dtype
    try:
        param = next(base_model.parameters())
        wrapped_model.projection = wrapped_model.projection.to(device=param.device, dtype=param.dtype)
        print(f"✅ Projection head loaded: device={param.device}, dtype={param.dtype}")
    except StopIteration:
        pass
    
    wrapped_model.eval()
    print("✅ Model loaded successfully!")
    
    # Create API model
    model = _RelSimClass(wrapped_model, processor)
    
    return model, preprocess_image


# For backward compatibility - CLI usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compute similarity score between two images")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--image1",
        type=str,
        required=True,
        help="Path to first image"
    )
    parser.add_argument(
        "--image2",
        type=str,
        required=True,
        help="Path to second image"
    )
    args = parser.parse_args()
    
    print(f"Image 1: {args.image1}")
    print(f"Image 2: {args.image2}")
    print("\n" + "="*60)
    print("Loading model...")
    print("="*60)
    
    # Load model using new API
    model, preprocess = relsim(pretrained=True, checkpoint_dir=args.checkpoint_dir)
    
    # Load and preprocess images
    print("\n" + "="*60)
    print("Loading images...")
    print("="*60)
    
    image1 = preprocess(Image.open(args.image1))
    image2 = preprocess(Image.open(args.image2))
    print("✅ Images loaded")
    
    # Compute similarity
    print("\n" + "="*60)
    print("Computing similarity...")
    print("="*60)
    
    similarity = model(image1, image2)
    
    print(f"✅ Similarity score: {similarity:.4f}")
    
    print("\n" + "="*60)
    print("🎉 Done!")
    print("="*60)