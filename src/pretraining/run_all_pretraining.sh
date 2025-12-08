#!/bin/bash
DATA_ROOT="~/Desktop/imageNet100"
EPOCHS=30
BATCH_SIZE=128
LR=0.002
NUM_WORKERS=12

BACKBONES=("resnet18" "resnet34" "resnet50" "resnet101")

for R in "${BACKBONES[@]}"; do
    echo "============================================"
    echo " Pretraining $R on ImageNet-100"
    echo "============================================"

    python resnet_pretrain_imagenet100.py \
        --data_root "$DATA_ROOT" \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --resnet_type "$R" \
        --save_path "/home/yoann/Desktop/rework/project-2-roadseg_nsy/pretrained_resnets/$R-imagenet100.pth" \
        --lr "$LR" \
        --num_workers "$NUM_WORKERS"

done
