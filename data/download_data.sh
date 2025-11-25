########################################################
#
# Script to download the data from the JSONL file
#
########################################################

MAX_WORKERS=64

# Download anonymous captions images
echo "Downloading anonymous captions test images..."
python data/download_data.py --json_file data/anonymous_captions_test.jsonl \
    --save_dir data/anonymous_captions_test_images \
    --max_workers $MAX_WORKERS

python data/download_data.py --json_file data/anonymous_captions_train.jsonl \
    --save_dir data/anonymous_captions_train_images \
    --max_workers $MAX_WORKERS

# Download seed groups images
echo "Downloading seed groups images..."
python data/download_seed_groups.py --json_file data/seed_group.json \
    --save_dir data/seed_groups_images \
    --max_workers $MAX_WORKERS \
    --use_hash