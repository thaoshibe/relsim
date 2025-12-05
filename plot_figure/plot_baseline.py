import matplotlib.pyplot as plt
import numpy as np

# Academic-style settings
plt.rcParams.update({
    'font.size': 25,
    'axes.labelsize': 20,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})

# Models and scores
models = [
    "LPIPS", 
    "dreamsim", 
    "DINO", 
    "CLIP-I", "CLIP-T", 
    "Qwen-T",
    "Tuned DINO", "Tuned CLIP", 'Ours'
]

scores = [
    4.56,
    5.76,
    4.68,
    5.91, 5.33,
    4.86,
    5.62, 6.02, 6.77
]

# Colors
colors = [
    "#8172B2", # LPIPS
    "#DD8452", # dreamsim
    "#55A868", 
    "#D46A6D", "#C44E52", # CLIP
    "#937860",
    "#87A7CF", "#6A8DC0", "#4C72B0",
]

# Figure
fig, ax = plt.subplots(figsize=(10, 4))

# Custom x positions for nicer separation
x = np.arange(len(models), dtype=float)
x_shift = 0.5  # amount to shift right group
tuned_dino_index=6
x[tuned_dino_index:] += x_shift  # shift bars from 'Tuned DINO' onwards

bars = ax.bar(x, scores, color=colors, edgecolor='black', alpha=0.9)

# Add value labels on top
for xi, bar in zip(x, bars):
    height = bar.get_height()
    ax.text(xi, height + 0.1, f'{height:.2f}',
            ha='center', va='bottom', fontsize=20)

# Rotate x-axis labels
plt.xticks(x, models, rotation=20, ha='right')

# Vertical dashed line separating groups
clip_i_index = 5
tuned_dino_index = 6
line_x = (x[clip_i_index] + x[tuned_dino_index]) / 2

ax.axvline(x=line_x, color='#b0b0b0', linestyle='--', linewidth=2)
# Labels
ax.set_ylabel('GPT Score')
ax.set_ylim(4.4, max(scores) + 0.4)
ax.set_title('')  # academic style
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('baselines_vs_ours.pdf', dpi=300, bbox_inches='tight')
plt.show()
