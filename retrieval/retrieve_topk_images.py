# open the precomputed/relsim.npz file
# open top 1000 images, and retrive the top 10 images to the query image based on the cosine similarity
# retrive the top 10 images to the query image

import numpy as np
from tqdm import tqdm
import json
import argparse
import os
import glob

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--output_file", type=str, default="retrieved_images.json")
    parser.add_argument("--num_images", type=int, default=1000)
    parser.add_argument("--image_dir", type=str, default="./images")
    parser.add_argument("--precomputed_dir", type=str, default="./precomputed")
    return parser.parse_args()

def retrieve_topk_images(topk, image_indices, all_embeddings, all_image_ids, image_dir):
    # Compute similarity scores once
    similarity_scores = all_embeddings[image_indices] @ all_embeddings.T
    # Get top k+1 results, then skip the first one (the image itself)
    topk_plus_one_indices = np.argsort(similarity_scores, axis=1)[:, ::-1][:, :topk+1]
    topk_indices = topk_plus_one_indices[:, 1:]  # Skip first column (self-match)
    topk_scores = np.round(np.take_along_axis(similarity_scores, topk_indices, axis=1), 3)
    
    result_dict = []

    for i in range(len(image_indices)):
        result_dict.append({
            "query_image_id": all_image_ids[image_indices[i]],
            "query_image_path": os.path.join(image_dir, f"{all_image_ids[image_indices[i]]}.png"),
            "top_k_image_ids": [all_image_ids[index] for index in topk_indices[i]],
            "top_k_image_paths": [os.path.join(image_dir, f"{all_image_ids[index]}.png") for index in topk_indices[i]],
            "top_k_scores": topk_scores[i].tolist(),
        })
    return result_dict

if __name__ == "__main__":
    args = parse_args()
    topk = args.topk
    output_file = args.output_file
    num_images = args.num_images
    image_dir = args.image_dir
    precomputed_dir = args.precomputed_dir

    # list all the npz files in the precomputed directory
    npz_files = glob.glob(os.path.join(precomputed_dir, "*.npz"))
    print(f"Found {len(npz_files)} npz files")

    # Load the first npz file to get the total number of images
    if len(npz_files) == 0:
        raise ValueError(f"No npz files found in {precomputed_dir}")
    
    first_data = np.load('./precomputed/relsim.npz')
    total_images = len(first_data['embeddings'])
    
    # select the indices of intersting images to retrieve
    start_index = 14881
    available_indices = np.arange(start_index, total_images)
    random_indices = np.random.choice(available_indices, min(num_images, len(available_indices)), replace=False)
    print(f"Sampled {len(random_indices)} random indices from {total_images} total images")
    print(random_indices)

    save_dict = {}
    for npz_file in tqdm(npz_files, desc="Processing npz files"):
        data = np.load(npz_file)
        all_embeddings = data['embeddings']
        all_image_ids = data['image_ids']
        all_captions = data['captions']
        retrieved_result = retrieve_topk_images(topk, random_indices, all_embeddings, all_image_ids, image_dir)
        save_dict[f'{npz_file.split("/")[-1].split(".")[0]}'] = retrieved_result
    
    with open(output_file, "w") as f:
        json.dump(save_dict, f, indent=2)
    print(f"Saved {len(save_dict)} retrieved results to {output_file}")
