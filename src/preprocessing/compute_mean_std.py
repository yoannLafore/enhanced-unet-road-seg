# Simple code to compute mean and std of dataset
# Need to only be run once
import os
import numpy as np
from PIL import Image
from tqdm import tqdm


def main():
    img_dir = "/home/yoann/Desktop/project-2-roadseg_nsy/data/training/images/"
    all_imgs = sorted(
        os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".png")
    )

    mean = np.zeros(3)
    std = np.zeros(3)
    for img_path in tqdm(all_imgs, desc="Computing mean and std"):
        img = Image.open(img_path).convert("RGB")
        img = np.array(img) / 255.0
        mean += img.mean(axis=(0, 1))
        std += img.std(axis=(0, 1))

    mean /= len(all_imgs)
    std /= len(all_imgs)

    print(f"Mean: {mean}, Std: {std}")


if __name__ == "__main__":
    main()


# Output:
# Mean: [0.33298134 0.33009373 0.29579783], Std: [0.18409964 0.17780256 0.17631003]
