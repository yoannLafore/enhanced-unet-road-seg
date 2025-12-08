from src.model.unet import *
from src.model.resnet_unet import *
from src.preprocessing.dataset import *
import os
import torch
from tqdm import tqdm
import wandb
from src.utils import *
import json
import cv2

from src.train.train_epoch import *
from src.helper.mask_to_submission import create_submission

# TODO: Set the correct paths before running
MODEL_PATH = ""
TEST_IMGS_DIR = ""


# Script to run in order to generate the submission of our best model on aicrowd
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = Unet()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    model.to(device)
    print("Model loaded successfully.")

    # Create output directory
    OUT_DIR = "generated_test/"
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load test dataset
    test_transform = default_transform()

    test_imgs = [
        os.path.join(TEST_IMGS_DIR, f"test_{i}/test_{i}.png") for i in range(1, 51)
    ]
    print(f"Found {len(test_imgs)} test images.")

    for i, img_path in tqdm(enumerate(test_imgs), total=len(test_imgs)):
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Apply transformations
        augmented = test_transform(image=img_rgb)
        input_tensor = augmented["image"].unsqueeze(0)  # Add batch dimension
        input_tensor = input_tensor.to(device)

        # Inference
        with torch.no_grad():
            output, preds = model(input_tensor)
            output_mask = (preds > 0.5).float()

        # Save output mask
        output_mask_np = output_mask.squeeze().cpu().numpy() * 255
        output_mask_np = output_mask_np.astype(np.uint8)

        output_path = os.path.join(OUT_DIR, f"test_{i+1:03d}.png")
        cv2.imwrite(output_path, output_mask_np)
    print(f"Generated masks saved to: {OUT_DIR}")

    # Create submission file
    submission_filename = "submission.csv"
    create_submission(OUT_DIR, "test_", submission_filename)
    print(f"Submission file created: {submission_filename}")


if __name__ == "__main__":
    main()
