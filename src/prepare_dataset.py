import os
import shutil
import random
from collections import defaultdict

random.seed(42)

IMAGE_DIR = "data/raw/img_align_celeba/img_align_celeba"
IDENTITY_FILE = "data/raw/identity_CelebA.txt"

TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
TEST_DIR = "data/test"

TARGET_IDENTITIES = 100
MIN_IMAGES_PER_IDENTITY = 20


def reset_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def load_identity_map(identity_file):
    identity_map = defaultdict(list)

    with open(identity_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            image_name, identity = parts
            identity_map[identity].append(image_name)

    return identity_map


def split_images(images):
    random.shuffle(images)
    n = len(images)

    if n < 3:
        return images, [], []

    train_count = max(1, int(n * 0.7))
    val_count = max(1, int(n * 0.15))
    test_count = n - train_count - val_count

    if test_count < 1:
        test_count = 1
        if train_count > val_count and train_count > 1:
            train_count -= 1
        elif val_count > 1:
            val_count -= 1

    train_imgs = images[:train_count]
    val_imgs = images[train_count:train_count + val_count]
    test_imgs = images[train_count + val_count:]

    if len(val_imgs) == 0 and len(train_imgs) > 1:
        val_imgs = [train_imgs.pop()]
    if len(test_imgs) == 0 and len(train_imgs) > 1:
        test_imgs = [train_imgs.pop()]

    return train_imgs, val_imgs, test_imgs


def copy_images(image_names, identity_label, target_root):
    class_dir = os.path.join(target_root, identity_label)
    os.makedirs(class_dir, exist_ok=True)

    for image_name in image_names:
        src = os.path.join(IMAGE_DIR, image_name)
        dst = os.path.join(class_dir, image_name)

        if os.path.exists(src):
            shutil.copy2(src, dst)


def main():
    reset_dir(TRAIN_DIR)
    reset_dir(VAL_DIR)
    reset_dir(TEST_DIR)

    identity_map = load_identity_map(IDENTITY_FILE)

    eligible = {
        identity: imgs
        for identity, imgs in identity_map.items()
        if len(imgs) >= MIN_IMAGES_PER_IDENTITY
    }

    selected_identities = list(eligible.keys())[:TARGET_IDENTITIES]

    print(f"Total identities found: {len(identity_map)}")
    print(f"Eligible identities with at least {MIN_IMAGES_PER_IDENTITY} images: {len(eligible)}")
    print(f"Selected identities: {len(selected_identities)}")

    for identity in selected_identities:
        images = eligible[identity]
        train_imgs, val_imgs, test_imgs = split_images(images)

        if len(train_imgs) == 0 or len(val_imgs) == 0 or len(test_imgs) == 0:
            continue

        copy_images(train_imgs, identity, TRAIN_DIR)
        copy_images(val_imgs, identity, VAL_DIR)
        copy_images(test_imgs, identity, TEST_DIR)

    print("Dataset preparation completed successfully.")


if __name__ == "__main__":
    main()