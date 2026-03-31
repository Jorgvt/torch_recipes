#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-sxm4:4
#SBATCH --job-name=train_model_saving
#SBATCH --output=log_training.log

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12355

uv run torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train.py \
    --model alexnet \
    --epochs 4 \
    --lr 1e-2 \
    --batch-size 64 \
    --data-path /media/disk/vista/BBDD_video_image/imagenet_complete/ILSVRC/Data/CLS-LOC \
    --output-dir /media/disk/vista/BBDD_video_image/Modelos_epochs/alexnet