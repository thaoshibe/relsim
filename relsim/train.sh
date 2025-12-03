# export WANDB_API_KEY='your_wandb_api_key'
# export HF_TOKEN='your_hf_token'

CUDA_VISIBLE_DEVICES=4,5,6,7 python train_relsim.py \
    --json_files ../data/anonymous_captions_train.jsonl \
    --image_folder ../data/anonymous_captions_train_images \
    --output_dir /mnt/localssd/ \
    --run_name bs64 \
    --batch_size 64 \
    --learning_rate 1e-4 \
    --num_epochs 3 \
    --temperature 0.07 \
    --save_steps 1000 \
    --logging_steps 1