# Architectural Improvements to U-Net for Road Segmentation
Testing architectural modifications to the U-Net framework to enhance performance for road segmentation tasks.

## Overview

This project applies and evaluates variants of the U-Net architecture for the task of road segmentation.

More specifically, an in-depth study is conducted on integrating a ResNet backbone into U-Net, both with and without pretraining. In addition, the impact of adding a refinement module at the output of the U-Net is evaluated.

## 🚀 Quick Setup

The following instructions describe the minimum steps required to reproduce the AIcrowd submission.

### 1. Code Location

The code used to generate the submission is located at the repository root in `run.py`.

### 2. Environment

The project uses Python 3.10 and depends on common machine learning libraries listed in `requirements.txt`.
To install the required dependencies, run the following command from the repository root:

```bash
pip install -r requirements.txt
```

A justification for the chosen libraries is provided further below.

### 3. Dataset Path

In `run.py`, set the `TEST_IMGS_DIR` variable to the path of the directory containing the test images (organized in the same structure as provided by AIcrowd). For example:

```python
TEST_IMGS_DIR="/path/to/data/test_set_images/"
```

### 4. Generating the Submission

To generate the submission, run the following command from the repository root:

```bash
python -m run
```

The predicted mask images will be saved in the `generated_test/` directory, and the corresponding submission file will be created at the repository root as `submission.csv`.

### (Optional) 5. Training & Results Reproducibility

To reproduce the training runs and results reported in the study, the provided configuration framework can be used. First, set the path to the training dataset in `src/configs/base.yaml` as follows:

```yaml
data: 
  train_dir: /path/to/data/training/
```

Then, from the repository root, run the following script to execute all training configurations used to produce the reported results and store the outputs in the `out/` directory:

```bash
./src/configs/run_all_configs.sh
```

This script automatically runs all cross-validation experiments using the same random seed as in the report. Note that minor variations may still occur due to the non-deterministic behavior of NVIDIA GPUs.

## 📝 Methodology

The methodology used to address this problem is described in detail in the accompanying report.

## 📚 External Libraries and Datasets

The following section provides the sources and justification for the external libraries and datasets used throughout the project.

### Datasets

**ImageNet100:** To assess the effect of ResNet pretraining before using it as a backbone for U-Net, access to a labeled image dataset was required. For practical and resource-related reasons, a subset of the ImageNet-1k dataset containing 100 randomly sampled classes was selected. This dataset is publicly available on Kaggle [here](https://www.kaggle.com/datasets/ambityga/imagenet100).

Minor formatting adjustments were applied for code compatibility:

* `train.X1/, train.X2/, train.X3/, train.X4/` were merged into `train/`
* `val.X/` was renamed to `val/`


### Libraries

#### Maths and Statistics

* `numpy`: for numerical computations and array handling.
* `scikit-learn`: for computing validation metrics.

#### Computer Vision & Data Augmentation

* `albumentations`: for applying image transformations while preserving mask consistency.
* `opencv-python-headless`: for efficient image processing without GUI dependencies.
* `Pillow`: for basic image loading, saving, and manipulation operations.

#### Plotting

* `matplotlib`: for visualizing training curves, results, and sample predictions.

#### Deep Learning Frameworks

* `torch`: for model implementation, training, and inference.
* `torchvision`: for pretrained models, dataset utilities, and common computer vision transforms.

#### Utilities

* `tqdm`: for progress bar visualization during training and evaluation.
* `wandb`: for experiment tracking, logging, and result visualization.
* `omegaconf`: for configuration management.

