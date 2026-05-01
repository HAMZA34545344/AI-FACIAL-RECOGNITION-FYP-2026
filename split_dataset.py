import shutil
import random
from pathlib import Path

random.seed(42)

base_path = Path(r"C:\face-recognition-fyp\data")
source_path = base_path / "final_dataset_300"
split_path = base_path / "dataset_split"

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def get_images(folder):
    return [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in VALID_EXTS]

def clear_folder(folder):
    if folder.exists():
        for item in folder.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
    else:
        folder.mkdir(parents=True)

for split in ["train", "val", "test"]:
    clear_folder(split_path / split)

identity_folders = [f for f in source_path.iterdir() if f.is_dir()]

for identity in identity_folders:
    images = get_images(identity)
    random.shuffle(images)

    n = len(images)

    if n >= 10:
        train_imgs = images[:7]
        val_imgs = images[7:8]
        test_imgs = images[8:10]
    elif n == 9:
        train_imgs = images[:7]
        val_imgs = images[7:8]
        test_imgs = images[8:9]
    elif n == 8:
        train_imgs = images[:6]
        val_imgs = images[6:7]
        test_imgs = images[7:8]
    else:
        print(f"Skipped {identity.name} because it has only {n} images")
        continue

    for split_name, split_images in [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]:
        split_identity_folder = split_path / split_name / identity.name
        split_identity_folder.mkdir(parents=True, exist_ok=True)

        for img in split_images:
            shutil.copy2(img, split_identity_folder / img.name)

print("Dataset split completed successfully.")