import cv2
import pickle
import numpy as np
from pathlib import Path
from insightface.app import FaceAnalysis

project_root = Path(__file__).resolve().parent.parent
data_root = project_root / "data"
gallery_path = project_root / "face_gallery.pkl"

dataset_dirs = [
    data_root / "old_dataset",
    data_root / "new_dataset"
]

THRESHOLD = 0.60
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=-1, det_size=(640, 640))

def load_gallery():
    if not gallery_path.exists():
        print("Gallery file not found:", gallery_path)
        return None

    with open(gallery_path, "rb") as f:
        return pickle.load(f)

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

def predict_identity(img_path, gallery):
    img = cv2.imread(str(img_path))
    if img is None:
        return None, None

    face = detect_face_multi_scale(img)
    if face is None:
        return None, None

    query_embedding = face.normed_embedding

    best_match = None
    best_score = -1.0

    for identity, data in gallery.items():
        db_embedding = data["embedding"]
        score = float(np.dot(query_embedding, db_embedding))

        if score > best_score:
            best_score = score
            best_match = identity

    if best_score >= THRESHOLD:
        return best_match, best_score
    else:
        return "Unknown", best_score

def main():
    gallery = load_gallery()
    if gallery is None:
        print("Build gallery first.")
        return

    total = 0
    correct = 0
    undetected = 0

    for dataset_dir in dataset_dirs:
        print(f"\nChecking dataset: {dataset_dir}")

        if not dataset_dir.exists():
            print("Dataset not found, skipping.")
            continue

        identity_folders = sorted([f for f in dataset_dir.iterdir() if f.is_dir()])

        for identity_folder in identity_folders:
            true_label = f"{dataset_dir.name}_{identity_folder.name}"
            image_files = get_image_files(identity_folder)

            for img_path in image_files:
                predicted_label, score = predict_identity(img_path, gallery)
                total += 1

                if predicted_label is None:
                    undetected += 1
                    print(f"[NO FACE] {img_path.name}")
                    continue

                if predicted_label == true_label:
                    correct += 1
                    print(f"[CORRECT] {img_path.name} -> {predicted_label} ({score:.4f})")
                else:
                    print(f"[WRONG] {img_path.name} -> {predicted_label} ({score:.4f}) | True: {true_label}")

    accuracy = correct / total if total > 0 else 0

    print("\n" + "=" * 60)
    print("Evaluation complete")
    print("Total images:", total)
    print("Correct predictions:", correct)
    print("Undetected faces:", undetected)
    print(f"Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()