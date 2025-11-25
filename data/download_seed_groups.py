# Download seed groups images
import json
import os
import argparse
import hashlib
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


def compute_hash(url):
    """Compute hash from URL for consistent naming."""
    return hashlib.md5(url.encode()).hexdigest()


def download_image(image_url, save_path):
    """Download a single image from the given URL."""
    try:
        # Skip if already exists
        if os.path.exists(save_path):
            return f"Skipped {os.path.basename(save_path)} (already exists)"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(image_url, timeout=15, headers=headers)
        response.raise_for_status()
        
        # Process image with PIL
        image = Image.open(BytesIO(response.content))
        
        # Handle different image modes
        if image.mode in ('RGBA', 'LA', 'P'):
            if image.mode == 'P':
                image = image.convert('RGBA')
        elif image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Save as PNG
        image.save(save_path, 'PNG')
        
        return f"Downloaded {os.path.basename(save_path)}"
    
    except requests.exceptions.RequestException as e:
        return f"Failed to download {os.path.basename(save_path)}: Network error - {str(e)}"
    except Exception as e:
        return f"Failed to process {os.path.basename(save_path)}: {str(e)}"


def main():
    parser = argparse.ArgumentParser(
        description='Download seed group images from a JSON file containing image URLs.'
    )
    parser.add_argument(
        '--json_file',
        type=str,
        required=True,
        help='Path to the JSON file containing seed group data (anonymous_caption and image_urls fields)'
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
        help='Maximum number of concurrent download threads (default: 16)'
    )
    parser.add_argument(
        '--use_hash',
        action='store_true',
        help='Save images by hash instead of original filename'
    )
    
    args = parser.parse_args()
    
    # Load data from JSON file
    with open(args.json_file, 'r') as f:
        data = json.load(f)
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("-" * 50)
    print(f"Found {len(data)} seed groups in {args.json_file}")
    print(f"Images will be saved to {args.save_dir}")
    print(f"Naming strategy: {'hash' if args.use_hash else 'original filename'}")
    print("-" * 50)
    
    # Prepare all download tasks
    download_tasks = []
    for index, item in enumerate(data):
        image_urls = item['image_urls']
        this_group_dir = os.path.join(args.save_dir, f"group_{index}")
        os.makedirs(this_group_dir, exist_ok=True)
        
        for image_url in image_urls:
            if args.use_hash:
                # Use hash for filename
                image_hash = compute_hash(image_url)
                image_name = f"{image_hash}.png"
            else:
                # Use original filename
                image_name = os.path.basename(image_url)
                # Ensure .png extension
                if not image_name.lower().endswith('.png'):
                    image_name = os.path.splitext(image_name)[0] + '.png'
            
            save_path = os.path.join(this_group_dir, image_name)
            download_tasks.append((image_url, save_path))
    
    print(f"Total images to download: {len(download_tasks)}")
    
    successful = 0
    failed = 0
    
    # Download images concurrently
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(download_image, url, path): (url, path) for url, path in download_tasks}
        
        with tqdm(total=len(download_tasks), desc='Downloading images') as pbar:
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
    print(f"Check the images in the directory: {args.save_dir}")


if __name__ == "__main__":
    main()