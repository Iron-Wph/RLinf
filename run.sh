export HF_DATASETS_CACHE=/mnt/mnt/public/liuzhihao/hf_cache
export TRANSFORMERS_CACHE=/mnt/mnt/public/liuzhihao/transformers_cache
export HF_LEROBOT_HOME="/mnt/mnt/public/liuzhihao/RoboTwin-main/data"
torchrun --standalone --nnodes=1 --nproc_per_node=8 scripts/train_pytorch.py \
    --config_name place_empty_cup_clean \
    --exp_name place_empty_cup_clean