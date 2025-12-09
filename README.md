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


## 🧩 Structure

This section describes and explains the codebase structure, covering all processing stages including data preparation, training, evaluation, and cross-validation. The main codebase is located under `src/` and is organized into the following categories.

### 1. Configs

To efficiently test different approaches, hyperparameters, and model variants, a versatile configuration-based framework was implemented to execute various tasks based on provided configuration files.

The configuration system operates as follows:

* A base configuration file (`base.yaml`) serves as the default template and is merged at runtime with any task-specific configuration file.
* Task-specific configuration files are grouped into subdirectories.
* Two primary tasks are supported: `train` (standard training) and `kfolds` (k-fold cross-validation). Example configurations for both tasks are available under `configs/sample/`.
* To run a configuration, execute the following command from the repository root:

```bash
python -m src.run_config -c /path/to/config-file.yaml
```

The current codebase includes configuration sets for the following experiments:

* Baseline 5-fold cross-validation (`baseline/`)
* 5-fold cross-validation for ResNet-backed U-Nets, with and without pretrained weights (`resnet/`)
* 5-fold cross-validation for U-Net models augmented with a refinement module (`improvement_module/`)

Additionally, a convenience Bash script, `run_all_configs.sh`, is provided to launch all configurations used to produce the results reported in the study.

### 2. Pre-processing

To improve training efficiency and maximize performance on the relatively small dataset, a data augmentation–based preprocessing strategy was implemented. This stage is handled under the `preprocessing/` directory.

* A custom PyTorch dataset class is defined in `dataset.py` to wrap the dataset in a PyTorch-compatible format and to apply the augmentations described below. This module also provides utilities for loading the data either as a standard split or within a k-fold cross-validation setup.

* To mitigate overfitting and improve model robustness, an image augmentation pipeline was implemented. The `transform.py` file defines the training transformation `augmented_transform`, which applies a combination of spatial transformations (scaling, rotation, affine) and photometric transformations (hue variation, brightness adjustment, compression), each applied with a given probability. The pipeline concludes with image normalization.

* The validation and test transformation, `default_transform`, applies normalization only.

* Dataset statistics (mean and standard deviation) are computed once in `compute_mean_std` over the full dataset. Although this technically introduces minimal information leakage into the validation folds, the practical impact is considered negligible.


### 3. Model

The models used in this project are implemented under the `model/` directory. This includes a standard U-Net architecture as well as a U-Net variant with a ResNet backbone.

#### U-Net

A basic U-Net architecture is implemented in `unet.py`. It follows a standard 4-level design similar to the model proposed in *[U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)*.

One modification concerns the downsampling stages: strided convolutions are used instead of max-pooling layers to allow for improved feature retention during spatial reduction.

#### ResNet-Backbone U-Net

The U-Net with a ResNet backbone is implemented in `resnet_unet.py`. In this variant, the standard U-Net encoder is replaced by the encoding layers of a ResNet architecture.

A general `ResNetUnet` class is defined, supporting ResNet-18, ResNet-34, ResNet-50, and ResNet-101 backbones. The implementation allows either the use of pretrained weights from `torchvision` (used for testing purposes only) or loading weights from externally pretrained ResNet models (used for the experiments reported in the study).

For convenience, specialized wrapper classes are also provided: `ResNetUnet18`, `ResNetUnet34`, `ResNetUnet50`, and `ResNetUnet101`.


### 4. Training and Evaluation

This section summarizes the training and evaluation pipeline implemented under the `train/` directory.

---

#### Loss & Data Loading

* **Loss** functions are defined in `loss.py` (binary cross-entropy is currently supported).
* **Batch collation** is handled in `collate_fn.py`, which concatenates images and masks into PyTorch tensors compatible with the DataLoader.

---

#### Training Epoch

Per-epoch training logic is implemented in `train_epoch.py`.
A standard training routine (`train_epoch`) is provided for all base models, along with specialized procedures for U-Nets extended with a refinement module.

Two training strategies are available for the refinement setup:

* **Joint training (`train_epoch_improve_module_joint`)**
  A single forward–backward pass propagates gradients through both the refinement module and the base U-Net using a single loss.

* **Separate training (`train_epoch_improve_module_sep`)**
  The base U-Net and refinement module are optimized independently, each with its own loss, optimizer, and scheduler.
  Predictions from the base U-Net are detached before being passed to the refinement module, and backpropagation is run separately for both components.

Both strategies optionally allow forwarding intermediate U-Net feature maps to the refinement module using `forward_features=true`.

---

#### Evaluation Epoch

Model evaluation is implemented in `eval_epoch.py`.

* The validation set is processed to produce binary masks using a **0.5 threshold**.
* Metrics computed include **Accuracy, F1-score, Precision, and Recall**.
* Optional image and mask logging to **Weights & Biases** can be enabled via `log_wandb=true`.

Two evaluation functions are provided:

* `evaluate_epoch` for standard one-pass models (U-Net, ResNet U-Net).
* `evaluate_epoch_improve_module` for models with refinement modules.

---

#### Training Loop

The complete training loop is implemented in `unet_train.py` and is fully driven by the configuration framework described earlier.

* Handles dataloaders, optimizers, schedulers, and validation.
* Supports both:

  * **Single-pass models:** `train_model_on_ds`
  * **Refinement models:** `train_improve_model_on_ds`

Extensive logging is performed:

* **Locally**: checkpoints, metrics, and configs saved to the configured output directory.
* **Online (optional)**: experiment tracking via Weights & Biases.

Training can be launched from any configuration file using the `train_from_cfg` entry point.

---

#### K-Fold Cross-Validation

K-fold evaluation is implemented in `kfolds.py`.

* Trains models across all folds defined in a configuration file.
* Aggregates final metrics across folds to produce averaged performance results.
* Saves all results via local logging for reproducibility and comparison.

