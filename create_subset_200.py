from pathlib import Path
import shutil

# project root = the folder where this script exists
project_root = Path(__file__).resolve().parent

# source and destination
source_dir = project_root / "data" / "new dataset"
destination_dir = project_root / "data" / "subset_200_7images"

# image extensions we want to allow
image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

# check source folder
if not source_dir.exists():
    print("Source folder does not exist.")
    print("Checked path:", source_dir)
    exit()

# create destination folder
destination_dir.mkdir(parents=True, exist_ok=True)

# get identity folders only
identity_folders = sorted([folder for folder in source_dir.iterdir() if folder.is_dir()])

# take only first 200 identities
selected_identities = identity_folders[:200]

print("Total identity folders found:", len(identity_folders))
print("Selected identity folders:", len(selected_identities))
print("Destination folder:", destination_dir)
print("-" * 50)

for identity_folder in selected_identities:
    # create matching folder in destination
    target_identity_folder = destination_dir / identity_folder.name
    target_identity_folder.mkdir(parents=True, exist_ok=True)

    # get image files only from this identity folder
    image_files = sorted([
        file for file in identity_folder.iterdir()
        if file.is_file() and file.suffix.lower() in image_extensions
    ])

    # take only first 7 images
    selected_images = image_files[:7]

    print(f"{identity_folder.name}: found {len(image_files)} images, copying {len(selected_images)}")

    for image_file in selected_images:
        target_file = target_identity_folder / image_file.name
        shutil.copy2(image_file, target_file)

print("-" * 50)
print("Done! Subset created successfully.")