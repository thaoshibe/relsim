import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel
from qwen_vl_utils import process_vision_info
import glob, os, json
from tqdm import tqdm
from PIL import Image
import argparse

def load_model(adapter_path):
    """Load base + LoRA adapter and processor"""
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    processor = AutoProcessor.from_pretrained(adapter_path, trust_remote_code=True)
    processor.tokenizer.padding_side = 'left'
    return model, processor

def batch_inference(image_paths, model, processor):
    """Run batch inference on multiple images"""

    question = '''
    You are given a single image.
    Carefully analyze it to understand its underlying logic, layout, structure, or creative concept. Then generate a single, reusable anonymous caption that could describe any image following the same concept.
    The caption must:
    - Fully capture the general logic or analogy of the image.
    - Include placeholders (e.g., {Object}, {Word}, {Character}, {Meaning}, {Color}, etc.) wherever variations can occur.
    - Be concise and standalone.
    Important: Only output the anonymous caption. Do not provide any explanations or additional text.
    '''

    # Load and resize images to match training preprocessing
    target_size = (448, 448)
    images = []
    for img_path in image_paths:
        image = Image.open(img_path).convert("RGB")
        image = image.resize(target_size, Image.Resampling.LANCZOS)
        images.append(image)

    # Build messages for each image
    messages_batch = [[
        {"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": question}
        ]}
    ] for img in images]

    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in messages_batch
    ]
    image_inputs, video_inputs = process_vision_info(messages_batch)

    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding=True,
    ).to(model.device)

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
        trimmed = [
            out[len(inp):] for inp, out in zip(inputs.input_ids, generated)
        ]
        responses = processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

    torch.cuda.empty_cache()
    return responses

def get_args():
    parser = argparse.ArgumentParser(description="Run Qwen LoRA inference on images")
    parser.add_argument("--adapter_path", type=str, default="./anonymous_caption/lora-ckpt/", help="Path to the adapter checkpoint")
    parser.add_argument("--image_path", type=str, default="./anonymous_caption/mam.jpg", help="Path to a single image file or directory containing images")
    parser.add_argument("--start_index", type=int, default=0, help="Start index")
    parser.add_argument("--end_index", type=int, default=1000, help="Number of images to process")
    parser.add_argument("--save_dir", type=str, default="./anonymous_caption/results", help="Path to the save directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    return parser.parse_args()

if __name__ == "__main__":
    print("Loading model...")
    args = get_args()
    adapter_path = args.adapter_path
    model, processor = load_model(adapter_path)
    
    # Handle both single file and directory input
    image_path = args.image_path
    if os.path.isfile(image_path):
        IMAGE_PATHS = [image_path]
    elif os.path.isdir(image_path):
        IMAGE_PATHS = glob.glob(os.path.join(image_path, "*.png"))
    else:
        raise ValueError(f"Path {image_path} is neither a valid file nor directory")
    
    print(f"Found {len(IMAGE_PATHS)} images")

    print(f"Processing {len(IMAGE_PATHS)} images...")
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    batch_size, results = args.batch_size, {}
    for i in tqdm(range(0, len(IMAGE_PATHS), batch_size)):
        try:
            paths = IMAGE_PATHS[i:i+batch_size]
            responses = batch_inference(paths, model, processor)
            results.update(dict(zip(paths, responses)))
        except Exception as e:
            print(f"Error processing {paths[0]}: {e}")
            continue
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Generate appropriate output filename based on input type
    if os.path.isfile(args.image_path):
        # For single file, use the image filename
        base_name = os.path.splitext(os.path.basename(args.image_path))[0]
        output_file = f"{args.save_dir}/anonymous_caption_{base_name}.json"
        print(results)
    else:
        # For directory, use start/end index
        output_file = f"{args.save_dir}/anonymous_caption_{args.start_index}_{args.end_index}.json"
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_file}")
    