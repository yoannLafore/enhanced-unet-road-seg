# Architectural Improvements to U-Net for Road Segmentation
Testing architectural modifications to the U-Net framework to enhance performance for road segmentation tasks.

## Overview

This project applies and evaluates variants of the U-Net architecture for the task of road segmentation.

More specifically, an in-depth study is conducted on integrating a ResNet backbone into U-Net, both with and without pretraining. In addition, the impact of adding a refinement module at the output of the U-Net is evaluated.

## 🚀 Quick Setup for generating submission

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
