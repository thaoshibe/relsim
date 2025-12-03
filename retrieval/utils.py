import torch
import numpy as np
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image

_qwen_model = None
_qwen_processor = None
_clip_model = None
_clip_preprocess = None


def get_qwen_model():
    """Lazy load Qwen model"""
    global _qwen_model, _qwen_processor
    if _qwen_model is None:
        print("Loading Qwen model...")
        _qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
        )
        _qwen_processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        print("✅ Qwen model loaded")
    return _qwen_model, _qwen_processor


def get_clip_model():
    """Lazy load CLIP model"""
    global _clip_model, _clip_preprocess
    if _clip_model is None:
        print("Loading CLIP model (openai/clip-vit-large-patch14)...")
        from transformers import CLIPModel, CLIPProcessor
        _clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").cuda()
        _clip_preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        print("✅ CLIP model loaded")
    return _clip_model, _clip_preprocess

def generate_caption_with_qwen(image, prompt=None):
    """
    Generate caption for an image using Qwen model.
    
    Args:
        image: PIL Image or image path
        prompt: Text prompt for caption generation
    
    Returns:
        str: Generated caption
    """
    model, processor = get_qwen_model()

    prompt = '''
    Describe this image in an abstract manner, focusing on the relationships and structures within it rather than on specific attributes or semantic content.
    Use no more than 30 words.
    '''
    
    # Handle different image input types
    if isinstance(image, str):
        image_input = image  # Can be local path or URL
    elif isinstance(image, Image.Image):
        # CLIP expects PIL Image, Qwen processor will handle it
        image_input = image
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_input},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    
    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    
    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    return output_text[0]

def get_clip_text_embedding(text):
    """
    Get CLIP text embedding for a given text.
    
    Args:
        text: str, input text
    
    Returns:
        numpy array: Text embedding (normalized)
    """
    model, processor = get_clip_model()
    
    # Tokenize and encode text (truncate to max 77 tokens for CLIP)
    with torch.no_grad():
        inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True, max_length=77).to("cuda")
        text_embedding = model.get_text_features(**inputs)
        # Normalize
        text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
    
    return text_embedding.cpu().numpy()


def get_text_embedding_from_image(image):
    """
    Generate caption from image using Qwen, then get CLIP text embedding.
    
    Args:
        image: PIL Image or image path
    
    Returns:
        numpy array: CLIP text embedding
    """
    # Step 1: Generate caption with Qwen
    caption = generate_caption_with_qwen(image)
    print(f"  Generated caption: {caption}")
    
    # Truncate to 15 words if necessary (to stay within CLIP's 77 token limit) ~ Can be longer, but for now, keep it at 30.
    words = caption.split()
    if len(words) > 30:
        caption = ' '.join(words[:30])
        print(f"  Truncated to 30 words: {caption}")
    
    # Step 2: Get CLIP text embedding
    embedding = get_clip_text_embedding(caption)
    
    return embedding
