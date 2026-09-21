"""
Download the Face Mask Dataset from Kaggle and organise it into:
    dataset/
        with_mask/
        without_mask/

Requirements:
    pip install kaggle
    Set up your Kaggle API key:
        1. Go to https://www.kaggle.com/settings  → API → Create New Token
        2. Save kaggle.json to  ~/.kaggle/kaggle.json  (Linux/Mac)
                             or  C:\\Users\\YOU\\.kaggle\\kaggle.json  (Windows)
        3. chmod 600 ~/.kaggle/kaggle.json   (Linux/Mac only)

Then run:
    python download_dataset.py
"""

import os
import zipfile
import shutil
import sys

DATASET_SLUG = "omkargurav/face-mask-dataset"
ZIP_NAME     = "face-mask-dataset.zip"
OUT_DIR      = "dataset"

# Expected subfolder names inside the zip
SRC_WITH    = os.path.join("data", "with_mask")
SRC_WITHOUT = os.path.join("data", "without_mask")
DST_WITH    = os.path.join(OUT_DIR, "with_mask")
DST_WITHOUT = os.path.join(OUT_DIR, "without_mask")


def check_kaggle():
    try:
        import kaggle  # noqa: F401
    except ImportError:
        print("[ERROR] kaggle package not found.")
        print("        Run:  pip install kaggle")
        sys.exit(1)


def download():
    print(f"[INFO] Downloading '{DATASET_SLUG}' from Kaggle …")
    os.system(f"kaggle datasets download -d {DATASET_SLUG} -p . --unzip")


def organise():
    """Move images into dataset/with_mask/ and dataset/without_mask/."""
    # After --unzip the images land directly in with_mask / without_mask
    # relative to cwd.  Handle both possible layouts.
    os.makedirs(DST_WITH,    exist_ok=True)
    os.makedirs(DST_WITHOUT, exist_ok=True)

    for src, dst in [(SRC_WITH, DST_WITH), (SRC_WITHOUT, DST_WITHOUT)]:
        if os.path.isdir(src):
            for f in os.listdir(src):
                shutil.move(os.path.join(src, f), dst)
            shutil.rmtree(src, ignore_errors=True)
        # If already in the right place, nothing to do
        elif os.path.isdir(dst) and os.listdir(dst):
            pass
        else:
            print(f"[WARNING] Expected folder '{src}' not found — "
                  "check the zip contents manually.")

    # Clean up leftover data/ folder
    if os.path.isdir("data"):
        shutil.rmtree("data", ignore_errors=True)


def report():
    for label in ["with_mask", "without_mask"]:
        folder = os.path.join(OUT_DIR, label)
        count  = len(os.listdir(folder)) if os.path.isdir(folder) else 0
        print(f"  {label:20s}: {count} images")


if __name__ == "__main__":
    check_kaggle()
    download()
    organise()
    print("\n[INFO] Dataset ready:")
    report()
    print("\nNext step → run:  python train.py")
