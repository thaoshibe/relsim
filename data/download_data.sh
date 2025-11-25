########################################################
#
# Script to download the data from the JSONL file
#
########################################################

MAX_WORKERS=64

python data/download_data.py --json_file data/anonymous_captions_test.jsonl \
    --save_dir data/anonymous_captions_test_images \
    --max_workers $MAX_WORKERS

python data/download_data.py --json_file data/anonymous_captions_train.jsonl \
    --save_dir data/anonymous_captions_train_images \
    --max_workers $MAX_WORKERS