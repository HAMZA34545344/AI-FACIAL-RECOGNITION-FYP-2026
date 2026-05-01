import os
import shutil
import random

SOURCE_DIR = r"C:\face-recognition-fyp\data\subset_200_7images"
DEST_DIR = r"C:\face-recognition-fyp\data\subset_200_7images_split"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42

random.seed(SEED)

train_dir = os.path.join(DEST_DIR, "train")
val_dir = os.path.join(DEST_DIR, "val")
test_dir = os.path.join(DEST_DIR, "test")

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

identity_folders = [
    d for d in os.listdir(SOURCE_DIR)
    if os.path.isdir(os.path.join(SOURCE_DIR, d))
]

print(f"Found {len(identity_folders)} identities in source folder.\n")

for identity in sorted(identity_folders):
    src_identity_dir = os.path.join(SOURCE_DIR, identity)
    images = [
        f for f in os.listdir(src_identity_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if len(images) == 0:
        print(f"[SKIP] {identity} has no images.")
        continue

    random.shuffle(images)

    n = len(images)
    n_train = max(1, int(n * TRAIN_RATIO))
    n_val = max(1, int(n * VAL_RATIO))
    n_test = n - n_train - n_val

    if n_test < 1:
        n_test = 1
        if n_train > n_val:
            n_train -= 1
        else:
            n_val -= 1

    train_images = images[:n_train]
    val_images = images[n_train:n_train + n_val]
    test_images = images[n_train + n_val:]

    for split_name, split_images, split_root in [
        ("train", train_images, train_dir),
        ("val", val_images, val_dir),
        ("test", test_images, test_dir),
    ]:
        split_identity_dir = os.path.join(split_root, identity)
        os.makedirs(split_identity_dir, exist_ok=True)

        for img_name in split_images:
            src_path = os.path.join(src_identity_dir, img_name)
            dst_path = os.path.join(split_identity_dir, img_name)
            shutil.copy2(src_path, dst_path)

    print(
        f"{identity}: total={n}, "
        f"train={len(train_images)}, val={len(val_images)}, test={len(test_images)}"
    )

print("\nDataset split complete.")
print(f"Output saved to: {DEST_DIR}")