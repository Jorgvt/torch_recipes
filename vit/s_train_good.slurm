#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-sxm4:4
#SBATCH --job-name=train_vit_saving
#SBATCH --output=log_training.log

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12355

uv run torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train.py \
    --model vit_b_16 \
    --epochs 4 \
    --batch-size 1024 \
    --opt adamw \
    --lr 0.003 \
    --wd 0.3 \
    --lr-scheduler cosineannealinglr \
    --lr-warmup-method linear \
    --lr-warmup-epochs 30 \
    --lr-warmup-decay 0.033 \
    --amp \
    --label-smoothing 0.11 \
    --mixup-alpha 0.2 \
    --auto-augment ra \
    --clip-grad-norm 1 \
    --ra-sampler \
    --cutmix-alpha 1.0 \
    --model-ema \
    --data-path /media/disk/vista/BBDD_video_image/imagenet_complete/ILSVRC/Data/CLS-LOC \
    --output-dir /media/disk/vista/BBDD_video_image/Modelos_epochs/vit/