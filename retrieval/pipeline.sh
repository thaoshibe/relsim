###########################################
#
#        Get Precomputed Embeddings
#
###########################################
JSON_FILE="combined.jsonl"
# # Get embedding for CLIP
python get_embedding_baseline.py \
    --json_file $JSON_FILE \
    --model_type clip \
    --output_path ./precomputed/clip.npz \
    --batch_size 16

# Get embedding for DINO
python get_embedding_baseline.py \
    --model_type dino \
    --json_file $JSON_FILE \
    --output_path ./precomputed/dino.npz \
    --batch_size 16

# Get embedding for relsim
python get_embedding_our.py \
    --checkpoint_dir thaoshibe/relsim-qwenvl25-lora \
    --json_file $JSON_FILE \
    --output_path ./precomputed/relsim.npz \
    --batch_size 16

# Optional: You can also get embedding for your own model
# python get_embedding_our.py \
#     --checkpoint_dir $PATH_TO_YOUR_MODEL \
#     --json_file $JSON_FILE \
#     --output_path ./precomputed/your_model.npz \
#     --batch_size 16

###########################################
#
#        Retrieve Top-K Images
#
###########################################
python retrieve_topk_images.py \
    --precomputed_dir ./precomputed \
    --output_file retrieved_images.json \
    --topk 10 \
    --num_images 1000 \
    --image_dir ./images

###########################################
#
#        GPTScore Evaluation
#
###########################################
# Evaluate ALL methods using top-1 retrieved images
# python gptscore.py \
#     --json retrieved_images.json \
#     --output gpt_scores.json \
#     --top-k 1 \
#     --workers 64
    # --methods clip relsim relsim_iter1000

# Evaluate only specific methods
# python gptscore.py --json retrieved_images.json --methods dino clip relsim