import matplotlib.pyplot as plt

# =========================
# Adjustable style settings
# =========================
FIG_WIDTH = 11
FIG_HEIGHT = 4
BAR_HEIGHT = 0.6
BAR_ALPHA = 0.9
HSPACE = 0         # vertical spacing between rows
X_MARGIN = 10       # margin for text labels
RIGHT_MARGIN = 100  # extra space for method names

# Font sizes
FONT_SIZE = 25
LABEL_FONT_SIZE = 25
TICK_FONT_SIZE = 18
PCT_FONT_SIZE = 25
METHOD_FONT_SIZE = 25
TITLE_FONT_SIZE = 30

# Colors
colors = {
    'ours': '#4C72B0',      # Blue
    'tie': '#BFC5CE',       # Gray
}

# =========================
# Baseline colors per row (top to bottom)
# =========================
# baseline_colors = ['#8c564b','#d62728', '#ffa94d', '#ff7f0e', '#2ca02c']  # Orange, Green, Red, Purple, Brown
baseline_colors = ["#DD8452", "#55A868", "#D46A6D", "#C44E52", "#937860"]
# =========================
# Academic-style settings
# =========================
plt.rcParams.update({
    'font.size': FONT_SIZE,
    'axes.labelsize': LABEL_FONT_SIZE,
    'axes.titlesize': FONT_SIZE,
    'xtick.labelsize': TICK_FONT_SIZE,
    'ytick.labelsize': TICK_FONT_SIZE,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linestyle': '--',
})

# =========================
# Comparative evaluation data
# =========================
comparisons = [
    {"method": "DreamSim", "ours": 427, "tie": 150, "baseline": 303, "total": 880},
    {"method": "DINO", "ours": 477, "tie": 189, "baseline": 214, "total": 880},
    {"method": "CLIP-I", "ours": 340, "tie": 205, "baseline": 255, "total": 800},
    {"method": "CLIP-T", "ours": 525, "tie": 111, "baseline": 244, "total": 880},
    {"method": "Qwen-T", "ours": 534, "tie": 154, "baseline": 192, "total": 880},
]

# =========================
# Create figure
# =========================
fig, axes = plt.subplots(len(comparisons), 1, figsize=(FIG_WIDTH, FIG_HEIGHT), sharex=True)
fig.subplots_adjust(hspace=HSPACE, top=0.93, bottom=0.1, left=0.15, right=0.95)

for i, (ax, comp) in enumerate(zip(axes, comparisons)):
    counts = [comp['ours'], comp['tie'], comp['baseline']]
    labels = ['Ours', 'Tie', 'Baseline']
    total = comp['total']

    left = 0
    for count, label in zip(counts, labels):
        if label == 'Baseline':
            color = baseline_colors[i]  # pick the color for this row
        else:
            color = colors[label.lower()]
        
        ax.barh(0, count, left=left, color=color, height=BAR_HEIGHT, edgecolor='black', alpha=BAR_ALPHA)
        
        # Annotate percentages
        pct = (count / total) * 100
        if pct > 5:
            ax.text(left + count / 2, 0, f'{pct:.1f}%', ha='center', va='center', color='white', fontsize=PCT_FONT_SIZE)
        
        # Label "Ours" explicitly on the left side
        if label == 'Ours':
            ax.text(left - X_MARGIN, 0, 'Ours', ha='right', va='center', fontsize=PCT_FONT_SIZE)
        
        left += count

    # Method label at the right end
    ax.text(total + X_MARGIN, 0, comp['method'], va='center', ha='left', fontsize=METHOD_FONT_SIZE)

    ax.set_xlim(0, total + RIGHT_MARGIN)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.grid(axis='x', alpha=0.25, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)

# X-axis label only on the last plot
axes[-1].set_xlabel('Number of User Responses', fontsize=LABEL_FONT_SIZE)
axes[-1].tick_params(axis='x', which='both', length=5, width=1)

# Overall title
# fig.suptitle('Human Preference Study: Pairwise Comparisons', fontsize=TITLE_FONT_SIZE, y=0.97)
plt.savefig("user_study.pdf", dpi=300, bbox_inches='tight')
plt.show()
