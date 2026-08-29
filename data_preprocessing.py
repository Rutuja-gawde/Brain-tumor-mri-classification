"""
data_preprocessing.py

Loads the Brain Tumor MRI dataset from the `data/Training` and `data/Testing`
folders, resizes all images, normalizes pixel values, and caches everything
as numpy arrays in `processed/` so the training scripts don't need to
re-read images from disk every time.

Expected folder layout (from the Kaggle dataset):
    data/Training/glioma/*.jpg
    data/Training/meningioma/*.jpg
    data/Training/notumor/*.jpg
    data/Training/pituitary/*.jpg
    data/Testing/<same 4 folders>
"""

import os
import cv2
import numpy as np
import json
from pathlib import Path

# ---------------- Config ----------------
DATA_DIR = Path("data")
TRAIN_DIR = DATA_DIR / "Training"
TEST_DIR = DATA_DIR / "Testing"
OUT_DIR = Path("processed")
IMG_SIZE = 128  # resize all images to IMG_SIZE x IMG_SIZE
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
# -----------------------------------------


def load_images_from_folder(folder: Path, img_size: int):
    """Reads every image under folder/<class_name>/, resizes, returns X, y."""
    X, y = [], []
    for label_idx, class_name in enumerate(CLASSES):
        class_dir = folder / class_name
        if not class_dir.exists():
            raise FileNotFoundError(
                f"Expected folder not found: {class_dir}. "
                f"Check that you extracted the dataset with the right structure."
            )
        files = list(class_dir.glob("*"))
        print(f"  {class_name}: {len(files)} images")
        for file_path in files:
            img = cv2.imread(str(file_path))
            if img is None:
                continue  # skip unreadable/corrupt files
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (img_size, img_size))
            X.append(img)
            y.append(label_idx)
    return np.array(X, dtype=np.uint8), np.array(y, dtype=np.int64)


def main():
    OUT_DIR.mkdir(exist_ok=True)

    print("Loading training images...")
    X_train, y_train = load_images_from_folder(TRAIN_DIR, IMG_SIZE)

    print("Loading testing images...")
    X_test, y_test = load_images_from_folder(TEST_DIR, IMG_SIZE)

    print(f"\nTrain set: {X_train.shape}, labels: {y_train.shape}")
    print(f"Test set:  {X_test.shape}, labels: {y_test.shape}")

    # Normalize pixel values to [0, 1] for the CNN; save raw uint8 too
    # (classical ML feature extraction works from raw images, CNN script
    # will normalize on load).
    np.save(OUT_DIR / "X_train.npy", X_train)
    np.save(OUT_DIR / "y_train.npy", y_train)
    np.save(OUT_DIR / "X_test.npy", X_test)
    np.save(OUT_DIR / "y_test.npy", y_test)

    label_map = {i: c for i, c in enumerate(CLASSES)}
    with open(OUT_DIR / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)

    print(f"\nSaved processed arrays + label_map.json to '{OUT_DIR}/'")
    print("Class balance (train):")
    for i, c in enumerate(CLASSES):
        print(f"  {c}: {(y_train == i).sum()}")


if __name__ == "__main__":
    main()