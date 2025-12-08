#!/bin/bash


CONFIG_LIST=(
    "src/configs/baseline/kfold_baseline.yaml"
    "src/configs/resnet/kfold_resnet18.yaml"
    "src/configs/resnet/kfold_resnet34.yaml"
    "src/configs/resnet/kfold_resnet50.yaml"
    "src/configs/resnet/kfold_resnet101.yaml"
    "src/configs/resnet/kfold_resnet18_pretrained.yaml"
    "src/configs/resnet/kfold_resnet34_pretrained.yaml"
    "src/configs/resnet/kfold_resnet50_pretrained.yaml"
    "src/configs/resnet/kfold_resnet101_pretrained.yaml"
    "src/configs/improvement_module/kfold_features_improve_joint.yaml"
    "src/configs/improvement_module/kfold_features_improve_sep.yaml"
    "src/configs/improvement_module/kfold_improve_joint.yaml"
    "src/configs/improvement_module/kfold_improve_sep.yaml"
)


for CONFIG_PATH in "${CONFIG_LIST[@]}"; do
    echo "============================================"
    echo " Running training with config: $CONFIG_PATH"
    echo "============================================"

    python -m src.run_config -c "$CONFIG_PATH"

done
