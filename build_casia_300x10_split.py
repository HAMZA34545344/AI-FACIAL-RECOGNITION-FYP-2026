import random
import shutil
from pathlib import Path
from pathlib import Path

source_dir = Path(r"C:\face-recognition-fyp\data\Gallery")
folders = [p for p in source_dir.iterdir() if p.is_dir()]

print("Gallery exists:", source_dir.exists())
print("Gallery path:", source_dir)
print("Total folders found:", len(folders))
print("First 20 folders:", sorted([p.name for p in folders])[:20])
print("Last 20 folders:", sorted([p.name for p in folders])[-20:])

SOURCE_DIR = Path(r"C:\face-recognition-fyp\data\Gallery")
OUTPUT_BASE = Path(r"C:\face-recognition-fyp\data\casia_300x10")

MAX_IDENTITIES = 300
IMAGES_PER_IDENTITY = 10
TRAIN_COUNT = 6
VAL_COUNT = 2
TEST_COUNT = 2
SEED = 42

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

assert TRAIN_COUNT + VAL_COUNT + TEST_COUNT == IMAGES_PER_IDENTITY, \
    "TRAIN_COUNT + VAL_COUNT + TEST_COUNT must equal IMAGES_PER_IDENTITY"

random.seed(SEED)

train_dir = OUTPUT_BASE / "train"
val_dir = OUTPUT_BASE / "val"
test_dir = OUTPUT_BASE / "test"

for d in [train_dir, val_dir, test_dir]:
    d.mkdir(parents=True, exist_ok=True)

if not SOURCE_DIR.exists():
    raise FileNotFoundError(f"Source folder not found: {SOURCE_DIR}")

identity_folders = sorted([p for p in SOURCE_DIR.iterdir() if p.is_dir()])
selected_identities = identity_folders[:MAX_IDENTITIES]

print(f"Source folder: {SOURCE_DIR}")
print(f"Found identity folders: {len(identity_folders)}")
print(f"Using identity folders: {len(selected_identities)}")

used = 0
skipped = []

for identity_path in selected_identities:
    images = [p for p in identity_path.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS]

    if len(images) < IMAGES_PER_IDENTITY:
        skipped.append((identity_path.name, len(images)))
        continue

    random.shuffle(images)
    chosen = images[:IMAGES_PER_IDENTITY]

    train_imgs = chosen[:TRAIN_COUNT]
    val_imgs = chosen[TRAIN_COUNT:TRAIN_COUNT + VAL_COUNT]
    test_imgs = chosen[TRAIN_COUNT + VAL_COUNT:TRAIN_COUNT + VAL_COUNT + TEST_COUNT]

    for split_dir, split_imgs in [
        (train_dir / identity_path.name, train_imgs),
        (val_dir / identity_path.name, val_imgs),
        (test_dir / identity_path.name, test_imgs),
    ]:
        split_dir.mkdir(parents=True, exist_ok=True)
        for img in split_imgs:
            shutil.copy2(img, split_dir / img.name)

    used += 1

print("\nDone.")
print(f"Usable identities copied: {used}")
print(f"Skipped identities: {len(skipped)}")

if skipped:
    print("\nFirst skipped identities:")
    for name, count in skipped[:20]:
        print(f"{name}: {count} images")

print(f"\nCreated dataset at: {OUTPUT_BASE}")
print(f"Train path: {train_dir}")
print(f"Val path:   {val_dir}")
print(f"Test path:  {test_dir}")