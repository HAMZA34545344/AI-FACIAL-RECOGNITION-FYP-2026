import cv2
import pickle
import numpy as np
from pathlib import Path
from insightface.app import FaceAnalysis


project_root = Path(__file__).resolve().parent.parent
gallery_path = project_root / "outputs" / "buffalo_gallery" / "gallery.pkl"
test_image_path = project_root / "test.jpg"
output_image_path = project_root / "recognized_output.jpg"

gallery_image_roots = [
    project_root / "old_dataset",
    project_root / "new_dataset",
]

MATCH_IF_ABOVE = 0.60
TOP_K = 4
THRESHOLDS_TO_CHECK = [0.20, 0.30, 0.40, 0.50, 0.60]
BLUR_THRESHOLD = 100.0


app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=-1, det_size=(640, 640))


def normalize_embedding(x):
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x)
    if norm == 0:
        return x
    return x / norm


def cosine_similarity(a, b):
    a = normalize_embedding(a)
    b = normalize_embedding(b)
    return float(np.dot(a, b))


def load_gallery(path):
    if not path.exists():
        return None

    with open(path, "rb") as f:
        return pickle.load(f)


def variance_of_laplacian(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def is_blurry(image, threshold=BLUR_THRESHOLD):
    score = variance_of_laplacian(image)
    return score < threshold, score


def preprocess_blurry_image(image):
    denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

    sharpen_kernel = np.array([
        [-1, -1, -1],
        [-1,  9, -1],
        [-1, -1, -1]
    ], dtype=np.float32)

    sharpened = cv2.filter2D(denoised, -1, sharpen_kernel)
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

    return sharpened


def detect_faces_with_fallback(image):
    detection_sizes = [
        (640, 640),
        (512, 512),
        (448, 448),
        (384, 384),
        (320, 320),
        (256, 256),
    ]

    for size in detection_sizes:
        try:
            app.det_model.input_size = size
            faces = app.get(image)
            print(f"Trying det_size={size} -> faces found: {len(faces)}")
            if len(faces) > 0:
                return faces
        except Exception as e:
            print(f"Detection failed at det_size={size}: {e}")

    return []


def extract_embedding(entry):
    if isinstance(entry, dict) and "embedding" in entry:
        return normalize_embedding(entry["embedding"])

    if isinstance(entry, np.ndarray):
        return normalize_embedding(entry)

    if isinstance(entry, list):
        return normalize_embedding(entry)

    return None


def rank_matches(test_embedding, gallery, top_k=4):
    matches = []

    for person_name, stored_entry in gallery.items():
        stored_embedding = extract_embedding(stored_entry)
        if stored_embedding is None:
            continue

        score = cosine_similarity(test_embedding, stored_embedding)
        matches.append((person_name, score))

    matches.sort(key=lambda x: x[1], reverse=True)
    return matches[:top_k]


def evaluate_thresholds(best_name, best_score, thresholds):
    result = {}
    for t in thresholds:
        result[f"{t:.2f}"] = best_name if best_score >= t else "Unknown"
    return result


def find_gallery_image(identity, gallery_entry):
    if not isinstance(gallery_entry, dict):
        return None

    used_images = gallery_entry.get("used_images", [])
    if not used_images:
        return None

    filename = used_images[0]

    for root in gallery_image_roots:
        candidate1 = root / identity / filename
        if candidate1.exists():
            return candidate1

        candidate2 = root / filename
        if candidate2.exists():
            return candidate2

        matches = list(root.rglob(filename))
        if matches:
            return matches[0]

    return None


def draw_results(image, results):
    annotated = image.copy()

    for result in results:
        if "bbox" not in result:
            continue

        x1, y1, x2, y2 = result["bbox"]
        name = result["best_name"]
        score = result["best_score"]
        blur_status = "Blurry" if result["was_blurry"] else "Clear"

        if name != "Unknown":
            color = (0, 255, 0)
            label = f"{name} | {score:.3f} | {blur_status}"
        else:
            color = (0, 0, 255)
            label = f"Unknown | {score:.3f} | {blur_status}"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        (text_w, text_h), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )

        text_y = max(y1 - 10, text_h + 5)
        cv2.rectangle(
            annotated,
            (x1, text_y - text_h - 6),
            (x1 + text_w + 6, text_y + 2),
            color,
            -1
        )

        cv2.putText(
            annotated,
            label,
            (x1 + 3, text_y - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    return annotated


def show_top_matches(query_image, top_matches, gallery):
    query_display = cv2.resize(query_image, (350, 350))
    cv2.imshow("Query Image", query_display)

    for idx, match in enumerate(top_matches, start=1):
        identity = match["name"]
        score = match["score"]
        gallery_entry = gallery.get(identity)

        image_path = find_gallery_image(identity, gallery_entry)
        if image_path is None:
            print(f"No display image found for {identity}")
            continue

        match_img = cv2.imread(str(image_path))
        if match_img is None:
            print(f"Could not load match image: {image_path}")
            continue

        match_img = cv2.resize(match_img, (250, 300))
        window_name = f"Top {idx}: {identity} | {score:.4f}"
        cv2.imshow(window_name, match_img)
        print(f"Top {idx}: {identity}, score={score:.4f}, image={image_path}")


def recognize_faces(image_path, gallery):
    image = cv2.imread(str(image_path))

    print("Script file:", __file__)
    print("Project root:", project_root)
    print("Gallery path:", gallery_path)
    print("Gallery exists?", gallery_path.exists())
    print("Test image path:", image_path)
    print("Test image exists?", image_path.exists())
    print("Image loaded:", image is not None)

    if image is None:
        return {"error": f"Could not read image: {image_path}"}, None

    print("Image shape:", image.shape)

    blurry, blur_score = is_blurry(image, BLUR_THRESHOLD)
    print(f"Blur score (Laplacian variance): {blur_score:.2f}")
    print(f"Is blurry? {blurry}")

    processed_image = image.copy()

    if blurry:
        print("Applying blur preprocessing...")
        processed_image = preprocess_blurry_image(image)

        blurry_after, blur_score_after = is_blurry(processed_image, BLUR_THRESHOLD)
        print(f"Blur score after preprocessing: {blur_score_after:.2f}")
        print(f"Still blurry after preprocessing? {blurry_after}")

    faces = detect_faces_with_fallback(processed_image)

    if not faces:
        return {"error": "No face detected in test image."}, processed_image

    results = []

    for i, face in enumerate(faces, start=1):
        bbox = face.bbox.astype(int).tolist()
        test_embedding = normalize_embedding(face.normed_embedding)

        top_matches_raw = rank_matches(test_embedding, gallery, TOP_K)

        if not top_matches_raw:
            results.append({
                "face_index": i,
                "bbox": bbox,
                "best_name": "Unknown",
                "best_score": -1.0,
                "decision_threshold": MATCH_IF_ABOVE,
                "blur_score": round(float(blur_score), 2),
                "was_blurry": bool(blurry),
                "threshold_check": {},
                "top_matches": []
            })
            continue

        best_name_raw, best_score = top_matches_raw[0]
        predicted_name = best_name_raw if best_score >= MATCH_IF_ABOVE else "Unknown"

        results.append({
            "face_index": i,
            "bbox": bbox,
            "best_name": predicted_name,
            "best_score": round(float(best_score), 4),
            "decision_threshold": MATCH_IF_ABOVE,
            "blur_score": round(float(blur_score), 2),
            "was_blurry": bool(blurry),
            "threshold_check": evaluate_thresholds(best_name_raw, best_score, THRESHOLDS_TO_CHECK),
            "top_matches": [
                {
                    "name": name,
                    "score": round(float(score), 4)
                }
                for name, score in top_matches_raw
            ]
        })

    return results, processed_image


def main():
    gallery = load_gallery(gallery_path)

    if gallery is None:
        print("Recognition Result:")
        print({"error": "Gallery not found. Run build_buffalo_gallery.py first."})
        return

    result, processed_image = recognize_faces(test_image_path, gallery)

    print("Recognition Result:")
    print(result)

    if isinstance(result, dict) and "error" in result:
        if processed_image is not None:
            cv2.imwrite(str(output_image_path), processed_image)
            print(f"Output image saved to: {output_image_path}")
        return

    annotated = draw_results(processed_image, result)
    cv2.imwrite(str(output_image_path), annotated)
    print(f"Annotated output image saved to: {output_image_path}")

    if len(result) > 0:
        show_top_matches(annotated, result[0]["top_matches"], gallery)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()