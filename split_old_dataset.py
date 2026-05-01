from pathlib import Path
import shutil
import random

# =========================
# CONFIG
# =========================
SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# =========================
# PATHS
# =========================
project_root = Path.cwd()
source_dir = project_root / "data" / "old_dataset"
output_dir = project_root / "data" / "split_dataset"

train_dir = output_dir / "train"
val_dir = output_dir / "val"
test_dir = output_dir / "test"

# =========================
# HELPERS
# =========================
def get_image_files(folder):
    return sorted([
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    ])

def copy_files(files, destination_folder):
    destination_folder.mkdir(parents=True, exist_ok=True)
    for file_path in files:
        shutil.copy2(file_path, destination_folder / file_path.name)

# =========================
# MAIN
# =========================
def main():
    random.seed(SEED)

    print("Current working directory:", project_root)
    print("Looking for source folder:", source_dir)
    print("Output folder will be:", output_dir)
    print("-" * 60)

    if not source_dir.exists():
        print("Source folder not found.")
        print("Make sure you are running this script inside face-recognition-fyp.")
        return

    identity_folders = sorted([f for f in source_dir.iterdir() if f.is_dir()])
    print(f"Total identity folders found: {len(identity_folders)}")
    print("-" * 60)

    total_train = 0
    total_val = 0
    total_test = 0
    skipped_folders = 0

    for identity_folder in identity_folders:
        image_files = get_image_files(identity_folder)

        if len(image_files) < 3:
            print(f"Skipping {identity_folder.name} because it has less than 3 images.")
            skipped_folders += 1
            continue

        random.shuffle(image_files)

        total_images = len(image_files)

        train_count = int(total_images * TRAIN_RATIO)
        val_count = int(total_images * VAL_RATIO)
        test_count = total_images - train_count - val_count

        # Make sure val and test each get at least 1 image
        if val_count == 0:
            val_count = 1
            train_count -= 1

        if test_count == 0:
            test_count = 1
            train_count -= 1

        # Safety check
        if train_count <= 0:
            print(f"Skipping {identity_folder.name} because not enough images after split.")
            skipped_folders += 1
            continue

        train_files = image_files[:train_count]
        val_files = image_files[train_count:train_count + val_count]
        test_files = image_files[train_count + val_count:]

        copy_files(train_files, train_dir / identity_folder.name)
        copy_files(val_files, val_dir / identity_folder.name)
        copy_files(test_files, test_dir / identity_folder.name)

        total_train += len(train_files)
        total_val += len(val_files)
        total_test += len(test_files)

        print(
            f"{identity_folder.name} -> "
            f"train: {len(train_files)}, "
            f"val: {len(val_files)}, "
            f"test: {len(test_files)}"
        )

    print("-" * 60)
    print("Done splitting dataset.")
    print(f"Total train images: {total_train}")
    print(f"Total val images: {total_val}")
    print(f"Total test images: {total_test}")
    print(f"Skipped folders: {skipped_folders}")
    print(f"Saved in: {output_dir}")

if __name__ == "__main__":
    main()