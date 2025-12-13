from relsim.relsim_score import relsim
from PIL import Image

# Load model
model, preprocess = relsim(pretrained=True, checkpoint_dir="thaoshibe/relsim-qwenvl25-lora")

img1 = preprocess(Image.open("./anonymous_caption/mam.jpg"))
img2 = preprocess(Image.open("./anonymous_caption/mam2.jpg"))
similarity = model(img1, img2)  # Returns similarity score (higher = more similar)
print(f"Similarity score: {similarity:.3f}")