import shutil
import random
from pathlib import Path

random.seed(42)

base_path = Path(r"C:\face-recognition-fyp\data")
gallery_path = base_path / "Gallery"
new_data_path = base_path / "CASIA-WebFace_cropped"
final_path = base_path / "final_dataset_300"

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
NUM_NEW_IDENTITIES = 163
MIN_IMAGES = 9

def get_image_files(folder):
    return [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in VALID_EXTS]

def get_identity_folders(folder):
    return [f for f in folder.iterdir() if f.is_dir()]

def copy_folder(src, dst_root, new_name=None):
    dst_name = new_name if new_name else src.name
    dst = dst_root / dst_name
    if dst.exists():
        return False
    shutil.copytree(src, dst)
    return True

# optional: clear final_dataset_300 first
if final_path.exists():
    for item in final_path.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()
else:
    final_path.mkdir(parents=True)

gallery_folders = get_identity_folders(gallery_path)
existing_names = set()

copied_gallery = 0
for folder in gallery_folders:
    if copy_folder(folder, final_path):
        existing_names.add(folder.name)
        copied_gallery += 1

print(f"Copied {copied_gallery} identities from Gallery")

new_folders = get_identity_folders(new_data_path)
eligible_folders = []

for folder in new_folders:
    image_count = len(get_image_files(folder))
    if image_count >= MIN_IMAGES:
        eligible_folders.append(folder)

print(f"Found {len(eligible_folders)} folders with at least {MIN_IMAGES} images in CASIA-WebFace_cropped")

if len(eligible_folders) < NUM_NEW_IDENTITIES:
    print(f"Not enough folders. Needed {NUM_NEW_IDENTITIES}, found {len(eligible_folders)}")
else:
    selected_folders = random.sample(eligible_folders, NUM_NEW_IDENTITIES)

    copied_new = 0
    for folder in selected_folders:
        folder_name = folder.name
        if folder_name in existing_names:
            folder_name = f"casia_{folder_name}"

        if copy_folder(folder, final_path, folder_name):
            existing_names.add(folder_name)
            copied_new += 1

    print(f"Copied {copied_new} identities from CASIA-WebFace_cropped")
    print(f"Total identities in final_dataset_300: {len([f for f in final_path.iterdir() if f.is_dir()])}")