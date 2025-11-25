import os
import json
import argparse
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time


def download_image(item, save_dir):
    """Download a single image from the given URL."""
    image_url = item['url_link']
    image_hash = item['image_hash']
    
    try:
        image_path = os.path.join(save_dir, f'{image_hash}.png')
        if os.path.exists(image_path):
            return f"Skipped {image_hash} (already exists)"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(image_url, timeout=15, headers=headers)
        response.raise_for_status()
        
        image = Image.open(BytesIO(response.content))
        
        if image.mode in ('RGBA', 'LA', 'P'):
            if image.mode == 'P':
                image = image.convert('RGBA')
        elif image.mode != 'RGB':
            image = image.convert('RGB')
        
        image.save(image_path, 'PNG')
        
        return f"Downloaded {image_hash}.png"
    
    except requests.exceptions.RequestException as e:
        return f"Failed to download {image_hash}: Network error - {str(e)}"
    except Exception as e:
        return f"Failed to process {image_hash}: {str(e)}"


def main():
    parser = argparse.ArgumentParser(
        description='Download images from a JSONL file containing image URLs and hashes.'
    )
    parser.add_argument(
        '--json_file',
        type=str,
        required=True,
        help='Path to the JSONL file containing image data (url_link and image_hash fields)'
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        required=True,
        help='Directory where downloaded images will be saved'
    )
    parser.add_argument(
        '--max_workers',
        type=int,
        default=16,
        help='Maximum number of concurrent download threads (default: 128)'
    )
    
    args = parser.parse_args()
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load data from JSONL file
    data = []
    with open(args.json_file, 'r') as f:
        for line in f:
            data.append(json.loads(line))

    print("-"*30)
    print(f'Found {len(data)} images in {args.json_file}')
    print(f"Image will be saved to {args.save_dir}")
    print("-"*30)
    
    successful = 0
    failed = 0
    
    # Download images with thread pool
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(download_image, item, args.save_dir): item for item in data}
        
        with tqdm(total=len(data), desc='Downloading images') as pbar:
            for future in as_completed(futures):
                result = future.result()
                if "Downloaded" in result or "Skipped" in result:
                    successful += 1
                else:
                    failed += 1
                
                rate = successful / (successful + failed) if (successful + failed) > 0 else 0
                pbar.set_postfix(success=successful, failed=failed, rate=f"{rate:.2%}")
                pbar.update(1)
    
    print(f"\nDownload complete: {successful} successful, {failed} failed, success rate: {successful / (successful + failed):.2%}")
    print('--- Check the images in the directory: ', args.save_dir)

if __name__ == "__main__":
    main()

