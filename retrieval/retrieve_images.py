# open the precomputed/relsim.npz file
# open top 1000 images, and retrive the top 10 images to the query image based on the cosine similarity
# retrive the top 10 images to the query image

import numpy as np
import torch
from torch.nn.functional import cosine_similarity
from tqdm import tqdm
import json

IMAGAE_DIR = "./images"

RELSIM_EMBEDDINGS_PATH = "./precomputed/relsim.npz"
CLIP_EMBEDDINGS_PATH = "./precomputed/clip.npz"
DINO_EMBEDDINGS_PATH = "./precomputed/dino.npz"


relsim_embeddings = np.load(RELSIM_EMBEDDINGS_PATH)
all_relsim_embeddings = relsim_embeddings["embeddings"]

clip_embeddings = np.load(CLIP_EMBEDDINGS_PATH)
all_clip_embeddings = clip_embeddings["embeddings"]

dino_embeddings = np.load(DINO_EMBEDDINGS_PATH)
all_dino_embeddings = dino_embeddings["embeddings"]

all_urls = dino_embeddings["url_links"]
# here, we sample 1000 random indices from all_relsim_embeddings
random_indices = np.random.choice(len(all_relsim_embeddings), 1000, replace=False)

relsim_scores = all_relsim_embeddings[random_indices] @ all_relsim_embeddings.T
clip_scores = all_clip_embeddings[random_indices] @ all_clip_embeddings.T
dino_scores = all_dino_embeddings[random_indices] @ all_dino_embeddings.T

# for each row, sort the scores and get the top 10 most similar

top_10_indices_relsim = np.argsort(relsim_scores, axis=1)[:, ::-1][:, :10]
top_10_indices_clip = np.argsort(clip_scores, axis=1)[:, ::-1][:, :10]
top_10_indices_dino = np.argsort(dino_scores, axis=1)[:, ::-1][:, :10]
# get the scores of these top 10 indices # get 3 decimal places
relsim_top_10_scores = np.round(np.sort(relsim_scores, axis=1)[:, ::-1][:, :10], 3)
clip_top_10_scores = np.round(np.sort(clip_scores, axis=1)[:, ::-1][:, :10], 3)
dino_top_10_scores = np.round(np.sort(dino_scores, axis=1)[:, ::-1][:, :10], 3)

# save result in json
result_dict = []
for i in range(len(random_indices)):
    result_dict.append({
        "query_image": int(random_indices[i]),
        "query_url": all_urls[random_indices[i]],
        "relsim_top_10": [all_urls[index] for index in top_10_indices_relsim[i]],
        "relsim_top_10_scores": str(relsim_top_10_scores[i].tolist()),
        "clip_top_10": [all_urls[index] for index in top_10_indices_clip[i]],
        "clip_top_10_scores": str(clip_top_10_scores[i].tolist()),
        "dino_top_10": [all_urls[index] for index in top_10_indices_dino[i]],
        "dino_top_10_scores": str(dino_top_10_scores[i].tolist()),
    })

# save result_dict to json
with open("1000_images_retrieved_images.json", "w") as f:
    json.dump(result_dict, f)
    
