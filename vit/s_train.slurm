#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-sxm4:1
#SBATCH --job-name=test
#SBATCH --output=log_training.log

# export MASTER_PORT=$(get_free_port)
# export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_PORT=12355
echo "MASTER_PORT"=$MASTER_PORT

# export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
# echo "WORLD_SIZE="$WORLD_SIZE

# master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# export MASTER_ADDR=$master_addr
export MASTER_ADDR=localhost
echo "MASTER_ADDR="$MASTER_ADDR

uv run torchrun --nproc_per_node=1 train.py --model vit_b_16 --epochs 300 --batch-size 32 --opt adamw --lr 0.003 --wd 0.3 --lr-scheduler cosineannealinglr --lr-warmup-method linear --lr-warmup-epochs 30 --lr-warmup-decay 0.033 --amp --label-smoothing 0.11 --mixup-alpha 0.2 --auto-augment ra --clip-grad-norm 1 --ra-sampler --cutmix-alpha 1.0 --model-ema --data-path /media/disk/vista/BBDD_video_image/imagenet_complete/ILSVRC/Data/CLS-LOC

