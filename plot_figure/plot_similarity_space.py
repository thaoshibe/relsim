import json
import os
import random
from collections import defaultdict

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from tqdm import tqdm

# -------------------------
# Configurations
# -------------------------
JSON_FILE = "./xy.json"
IMAGE_FOLDER = "xy_images"
FIG_SIZE = (17, 5)
MAX_HEIGHT = 15  # max height of images in display coordinates
MAX_SHOW = 4000  # maximum number of images to display
N_COLS = 20      # number of grid cells along x-axis
N_ROWS = 10      # number of grid cells along y-axis
MAX_POINTS_PER_CELL = 5  # maximum points per grid cell

# -------------------------
# Load data
# ------------------------


with open(JSON_FILE, 'r') as f:
    data = json.load(f)

clip_scores = np.array([x['clip_score'] for x in data.values()])
our_scores  = np.array([x['our_score']  for x in data.values()])
image_paths = [x['our_path'] for x in data.values()]

# copy top 40 clip score to another folder
os.makedirs('top_clip_images', exist_ok=True)
top_50_indices = np.argsort(clip_scores)[-50:]
for i in top_50_indices:
    import shutil
    shutil.copy(os.path.join(IMAGE_FOLDER, image_paths[i]), os.path.join('top_clip_images', image_paths[i].split('/')[-1]))

# -------------------------
# Normalize scores to [0,1]
# -------------------------
clip_scores_norm = (clip_scores - clip_scores.min()) / (clip_scores.max() - clip_scores.min())
our_scores_norm  = (our_scores  - our_scores.min())  / (our_scores.max()  - our_scores.min())

# -------------------------
# Define grid bins
# -------------------------
x_edges = np.linspace(0, 1, N_COLS + 1)
y_edges = np.linspace(0, 1, N_ROWS + 1)

x_idx = np.digitize(clip_scores_norm, x_edges) - 1  # 0-based
y_idx = np.digitize(our_scores_norm,  y_edges) - 1  # 0-based

# -------------------------
# Collect points per cell
# -------------------------
cell_points = defaultdict(list)
for i, (xi, yi) in enumerate(zip(x_idx, y_idx)):
    cell_points[(xi, yi)].append(i)

# Sample points per cell
sampled_indices = []
for key, indices in cell_points.items():
    if len(indices) > MAX_POINTS_PER_CELL:
        sampled_indices.extend(random.sample(indices, MAX_POINTS_PER_CELL))
    else:
        sampled_indices.extend(indices)

# Limit total points
sampled_indices = sampled_indices[:MAX_SHOW]

# copy the sampled_images into a new folder
# NEW_FOLDER = 'sampled_images'
# os.makedirs(NEW_FOLDER, exist_ok=True)
# for i in sampled_indices:
#     import shutil
#     shutil.copy(os.path.join(IMAGE_FOLDER, image_paths[i]), os.path.join(NEW_FOLDER, image_paths[i].split('/')[-1]))
# -------------------------
# Sort so higher-score images are plotted later (appear on top)
# -------------------------
# Sorting ascending so high-score images go last → appear on top
sampled_indices = sorted(sampled_indices, key=lambda i: our_scores[i])

# print("Top 2 image with highest CLIP score: im
# print("Image with highest our score:", image_paths[np.argmax(our_scores)])
# -------------------------
# Plot
# -------------------------
fig, ax = plt.subplots(figsize=FIG_SIZE)

# Draw grid lines
for xe in x_edges:
    ax.axvline(x=xe, color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
for ye in y_edges:
    ax.axhline(y=ye, color='gray', linestyle='--', linewidth=0.5, alpha=0.3)

# Make sure grid is behind images
ax.set_axisbelow(True)

# Plot sampled images
for i in tqdm(sampled_indices):
    x, y, img_path = clip_scores_norm[i], our_scores_norm[i], image_paths[i]
    try:
        img = mpimg.imread(os.path.join(IMAGE_FOLDER, img_path))
        img_height, img_width = img.shape[:2]
        zoom = MAX_HEIGHT / img_height
        im = OffsetImage(img, zoom=zoom)
        # ab = AnnotationBbox(im, (x, y), frameon=False)
                # Create AnnotationBbox with dark border
        ab = AnnotationBbox(
            im,
            (x, y),
            frameon=True,  # enable border
            bboxprops=dict(
                edgecolor='black',  # border color
                linewidth=0.7,      # border thickness
                boxstyle='square,pad=0'  # optional padding
            )
        )
        ax.add_artist(ab)
        ax.add_artist(ab)
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")

# -------------------------
# Axis setup
# -------------------------
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
# Hide ticks
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)


# ax.set_title('Attr vs CLIP Scores (Normalized & Sampled per Cell)', pad=20)

plt.savefig('similarity_space.png', dpi=300, bbox_inches='tight', transparent=True)
# plt.show()
