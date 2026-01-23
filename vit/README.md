To run:
```
uv run torchrun --nproc_per_node=1 train.py    --model vit_b_16 --epochs 300 --batch-size 32 --opt adamw --lr 0.003 --wd 0.3    --lr-scheduler cosineannealinglr --lr-warmup-method linear --lr-warmup-epochs 30    --lr-warmup-decay 0.033 --amp --label-smoothing 0.11 --mixup-alpha 0.2 --auto-augment ra    --clip-grad-norm 1 --ra-sampler --cutmix-alpha 1.0 --model-ema --data-path /media/disk/vista/BBDD_video_image/imagenet_complete/ILSVRC/Data/CLS-LOC
```
