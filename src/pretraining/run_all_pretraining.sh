#!/bin/bash
DATA_ROOT="~/Desktop/imageNet100"
EPOCHS=30
BATCH_SIZE=128

BACKBONES=("resnet18" "resnet34" "resnet50" "resnet101")

for R in "${BACKBONES[@]}"; do
    echo "============================================"
    echo " Pretraining $R on ImageNet-100"
    echo "============================================"

    python pretrain_resnet_imagenet100.py \
        --data_root "$DATA_ROOT" \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --resnet_type "$R" \
        --save_path "weights/${R}_imagenet100_pretrained.pth"

done
