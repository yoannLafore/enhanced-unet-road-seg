from src.model.unet import *
from src.model.resnet_unet import *
from src.preprocessing.dataset import *
import os
import torch
from tqdm import tqdm
from src.utils import *
import cv2

from src.train.epoch.train_epoch import *
from src.helper.mask_to_submission import create_submission

# TODO: Set the correct paths before running

# You should extract the "compet_resnet_unet.zip" archive at the repository root and the path should be already correct
MODEL_PATH = "competition_model/compet_resnet_unet.pth"
TEST_IMGS_DIR = ""  # Set the path to your test images directory: Ex: "/path/to/data/test_set_images/"

THRESHOLD = 0.4
SUBMISSION_FILENAME = "submission_compet.csv"


def flip_rotate_images(images: torch.Tensor) -> torch.Tensor:
    """Generate augmented versions of the input images by flipping and rotating.

    Args:
        images (torch.Tensor): Batch of images of shape (B, C, H, W).

    Returns:
        torch.Tensor: Batch of augmented images: each images is transformed into 8 images
                      (original, h-flip, v-flip, hv-flip, rot90, rot90+h-flip,
                       rot90+v-flip, rot90+hv-flip): (B*8, C, H, W).
    """
    augmented_images = []
    for img in images:
        # Original
        augmented_images.append(img)
        # Horizontal flip
        augmented_images.append(torch.flip(img, dims=[2]))
        # Vertical flip
        augmented_images.append(torch.flip(img, dims=[1]))
        # Horizontal and vertical flip
        augmented_images.append(torch.flip(img, dims=[1, 2]))
        # Rotate 90 degrees
        augmented_images.append(torch.rot90(img, k=1, dims=[1, 2]))
        # Rotate 90 degrees and horizontal flip
        augmented_images.append(
            torch.flip(torch.rot90(img, k=1, dims=[1, 2]), dims=[2])
        )
        # Rotate 90 degrees and vertical flip
        augmented_images.append(
            torch.flip(torch.rot90(img, k=1, dims=[1, 2]), dims=[1])
        )
        # Rotate 90 degrees and horizontal and vertical flip
        augmented_images.append(
            torch.flip(torch.rot90(img, k=1, dims=[1, 2]), dims=[1, 2])
        )

    return torch.stack(augmented_images)


def reverse_augmentations(augmented_masks: torch.Tensor) -> torch.Tensor:
    """Reverse the augmentations applied to the masks.

    Args:
        augmented_masks (torch.Tensor): Batch of augmented masks of shape (B*8, 1, H, W).

    Returns:
        torch.Tensor: Batch of original masks of shape (B, 1, H, W) obtained by averaging
                      the reversed augmentations.
    """
    B = augmented_masks.shape[0] // 8
    original_masks = []

    for i in range(B):
        aug_masks = augmented_masks[i * 8 : (i + 1) * 8]

        rev_masks = []

        # Original
        rev_masks.append(aug_masks[0])

        # Horizontal flip
        rev_masks.append(torch.flip(aug_masks[1], dims=[2]))

        # Vertical flip
        rev_masks.append(torch.flip(aug_masks[2], dims=[1]))

        # Horizontal + vertical flip
        rev_masks.append(torch.flip(aug_masks[3], dims=[1, 2]))

        # Rotate 90 degrees
        rev_masks.append(torch.rot90(aug_masks[4], k=3, dims=[1, 2]))  # R^{-1}

        # Rotate 90 degrees + horizontal flip
        rev_masks.append(
            torch.rot90(torch.flip(aug_masks[5], dims=[2]), k=3, dims=[1, 2])
        )

        # Rotate 90 degrees + vertical flip
        rev_masks.append(
            torch.rot90(torch.flip(aug_masks[6], dims=[1]), k=3, dims=[1, 2])
        )

        # Rotate 90 degrees + horizontal + vertical flip
        rev_masks.append(
            torch.rot90(torch.flip(aug_masks[7], dims=[1, 2]), k=3, dims=[1, 2])
        )

        # Average the reversed masks to get the final mask
        mask_mean = torch.stack(rev_masks).mean(dim=0)
        original_masks.append(mask_mean)

    return torch.stack(original_masks)


# Script to run in order to generate the submission of our best model on aicrowd
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = ResNet34Unet()
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
        # Load and transform image
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        augmented_img = test_transform(image=img_rgb)["image"].unsqueeze(0).to(device)
        augmented_img = flip_rotate_images(augmented_img)

        # Inference
        with torch.no_grad():
            # Because we don't know the memory available, we process each augmentation separately
            all_preds = []
            for j in range(augmented_img.shape[0]):
                output, pred = model(augmented_img[j : j + 1])
                all_preds.append(pred)
            all_preds = torch.cat(all_preds, dim=0)
            preds = reverse_augmentations(all_preds)

            output_mask = (preds > THRESHOLD).float().squeeze()

        # Save output mask
        output_mask_np = output_mask.cpu().numpy() * 255
        output_mask_np = output_mask_np.astype(np.uint8)

        output_path = os.path.join(OUT_DIR, f"test_{i+1:03d}.png")
        cv2.imwrite(output_path, output_mask_np)

    print(f"Generated masks saved to: {OUT_DIR}")

    # Create submission file
    create_submission(OUT_DIR, "test_", SUBMISSION_FILENAME)
    print(f"Submission file created: {SUBMISSION_FILENAME}")


if __name__ == "__main__":
    main()
