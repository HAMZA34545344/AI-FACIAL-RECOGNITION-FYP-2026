import cv2
import pickle
import numpy as np
from pathlib import Path
from insightface.app import FaceAnalysis

project_root = Path(__file__).resolve().parent.parent
data_root = project_root / "data"
gallery_path = project_root / "outputs" / "buffalo_gallery" / "gallery.pkl"


dataset_dirs = [
    data_root / "old_dataset",
    data_root / "new_dataset"
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_IMAGES_PER_IDENTITY = 1

app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=-1, det_size=(640, 640))

def get_image_files(folder):
    return sorted([
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    ])

def detect_face_multi_scale(img):
    for det_size in [(640, 640), (512, 512), (384, 384), (256, 256)]:
        app.prepare(ctx_id=-1, det_size=det_size)
        faces = app.get(img)
        if faces:
            return max(
                faces,
                key=lambda face: (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1])
            )
    return None

def main():
    print("Project root:", project_root)
    print("Gallery path:", gallery_path)
    print("Max images per identity:", MAX_IMAGES_PER_IDENTITY)
    print("Datasets to scan:")
    for d in dataset_dirs:
        print(" -", d)
    print("-" * 60)

    gallery = {}
    total_identities = 0
    total_images_used = 0
    skipped_images = 0

    for dataset_dir in dataset_dirs:
        print(f"\nChecking dataset: {dataset_dir}")

        if not dataset_dir.exists():
            print("Dataset not found, skipping.")
            continue

        identity_folders = sorted([f for f in dataset_dir.iterdir() if f.is_dir()])
        print("Identity folders found:", len(identity_folders))

        for identity_folder in identity_folders:
            image_files = get_image_files(identity_folder)[:MAX_IMAGES_PER_IDENTITY]
            embeddings = []

            identity_name = f"{dataset_dir.name}_{identity_folder.name}"
            print(f"Processing: {identity_name}")

            for img_path in image_files:
                img = cv2.imread(str(img_path))

                if img is None:
                    print("Could not read:", img_path.name)
                    skipped_images += 1
                    continue

                face = detect_face_multi_scale(img)

                if face is None:
                    print("No face found:", img_path.name)
                    skipped_images += 1
                    continue

                embeddings.append(face.normed_embedding)
                total_images_used += 1

            if len(embeddings) == 0:
                print("No valid embeddings for:", identity_name)
                continue

            mean_embedding = np.mean(embeddings, axis=0)
            mean_embedding = mean_embedding / np.linalg.norm(mean_embedding)

            gallery[identity_name] = {
                "embedding": mean_embedding,
                "images_used": len(embeddings),
                "dataset": dataset_dir.name,
                "person_name": identity_folder.name
            }

            total_identities += 1
            print(f"Saved {identity_name} with {len(embeddings)} images")

    with open(gallery_path, "wb") as f:
        pickle.dump(gallery, f)

    print("\n" + "=" * 60)
    print("Combined gallery saved successfully.")
    print("Gallery file:", gallery_path)
    print("Total identities saved:", total_identities)
    print("Total images used:", total_images_used)
    print("Skipped images:", skipped_images)
    

if __name__ == "__main__":
    main()