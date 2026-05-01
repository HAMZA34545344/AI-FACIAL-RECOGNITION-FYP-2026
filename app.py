import base64
import pickle
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, render_template, request, url_for
from insightface.app import FaceAnalysis
from torchvision import datasets, models, transforms
from torchvision.models import ResNet18_Weights

BASE_DIR = Path(r"C:/face-recognition-fyp") if Path(r"C:/face-recognition-fyp").exists() else Path.home() / "face-recognition-fyp"
PROJECT_ROOT = BASE_DIR
UPLOAD_DIR = Path(__file__).resolve().parent / "static" / "uploads"
RESULT_DIR = Path(__file__).resolve().parent / "static" / "results"
GALLERY_PATH = PROJECT_ROOT / "outputs" / "buffalo_gallery" / "gallery.pkl"

OLD_SPLIT_DATASET = PROJECT_ROOT / "data" / "split_dataset"
NEW_SPLIT_DATASET = PROJECT_ROOT / "data" / "subset_200_7images_split"

DATASET_ROOTS = [
    PROJECT_ROOT / "old_dataset",
    PROJECT_ROOT / "new_dataset",
]

TRAINED_MODELS = [
    {
        "id": "resnet_pretrained_stage2",
        "name": "ResNet Pretrained Stage 2",
        "arch": "resnet_pretrained",
        "path": PROJECT_ROOT / "checkpoints" / "best_resnet_pretrained_stage2.pth",
        "data_dir": NEW_SPLIT_DATASET,
    },
    {
        "id": "resnet_pretrained_stage1",
        "name": "ResNet Pretrained Stage 1",
        "arch": "resnet_pretrained",
        "path": PROJECT_ROOT / "checkpoints" / "best_resnet_pretrained_stage1.pth",
        "data_dir": NEW_SPLIT_DATASET,
    },
    {
        "id": "resnet_pretrained",
        "name": "ResNet Pretrained",
        "arch": "resnet_pretrained",
        "path": PROJECT_ROOT / "checkpoints" / "best_resnet_pretrained.pth",
        "data_dir": NEW_SPLIT_DATASET,
    },
    {
        "id": "resnet_scratch",
        "name": "ResNet Scratch",
        "arch": "resnet_scratch",
        "path": PROJECT_ROOT / "checkpoints" / "best_resnet_scratch.pth",
        "data_dir": NEW_SPLIT_DATASET,
    },
    {
        "id": "cnn_improved",
        "name": "Improved CNN",
        "arch": "cnn_improved",
        "path": PROJECT_ROOT / "outputs" / "checkpoints_improved" / "improved_best.pth",
        "data_dir": NEW_SPLIT_DATASET,
    },
    {
        "id": "cnn_baseline",
        "name": "CNN Baseline",
        "arch": "cnn_baseline",
        "path": PROJECT_ROOT / "outputs" / "checkpoints" / "baseline_best.pth",
        "data_dir": OLD_SPLIT_DATASET,
    },
]

MODEL_OPTIONS = [
    {"id": "auto_best", "name": "Auto Best Trained Model"},
    {"id": "resnet_pretrained_stage2", "name": "ResNet Pretrained Stage 2"},
    {"id": "resnet_pretrained_stage1", "name": "ResNet Pretrained Stage 1"},
    {"id": "resnet_pretrained", "name": "ResNet Pretrained"},
    {"id": "resnet_scratch", "name": "ResNet Scratch"},
    {"id": "cnn_improved", "name": "Improved CNN"},
    {"id": "cnn_baseline", "name": "CNN Baseline"},
    {"id": "buffalo_only", "name": "Buffalo Only"},
]

MATCH_IF_ABOVE = 0.60
TOP_K = 4
BLUR_THRESHOLD = 100.0
IMG_SIZE = 224
DEVICE = torch.device("cpu")

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=-1, det_size=(640, 640))


class CNNBaseline(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


class ImprovedCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class ResNetPretrained(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.layer4.parameters():
            param.requires_grad = True
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        return self.model(x)


class ResNetScratch(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.resnet18(weights=None)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        return self.model(x)


def load_gallery(path):
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


GALLERY = load_gallery(GALLERY_PATH)

TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

CLASS_NAMES_CACHE = {}
LOADED_MODEL_CACHE = {}


def get_class_names(data_dir):
    key = str(data_dir)
    if key in CLASS_NAMES_CACHE:
        return CLASS_NAMES_CACHE[key]

    train_dir = Path(data_dir) / "train"
    if not train_dir.exists():
        return None

    ds = datasets.ImageFolder(train_dir)
    CLASS_NAMES_CACHE[key] = ds.classes
    return CLASS_NAMES_CACHE[key]


def build_model(arch, num_classes):
    if arch == "cnn_baseline":
        return CNNBaseline(num_classes)
    if arch == "cnn_improved":
        return ImprovedCNN(num_classes)
    if arch == "resnet_scratch":
        return ResNetScratch(num_classes)
    return ResNetPretrained(num_classes)
def load_state_dict_safely(path):
    ckpt = torch.load(path, map_location=DEVICE)

    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
            state_dict = ckpt["model_state_dict"]
        elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    cleaned = {}
    for k, v in state_dict.items():
        new_key = k
        if new_key.startswith("module."):
            new_key = new_key[len("module."):]
        cleaned[new_key] = v

    return cleaned


def get_model_config_by_id(model_id):
    for cfg in TRAINED_MODELS:
        if cfg["id"] == model_id:
            return cfg
    return None

def load_model_from_config(cfg):
    cache_key = cfg["id"]
    if cache_key in LOADED_MODEL_CACHE:
        return LOADED_MODEL_CACHE[cache_key]

    path = cfg["path"]
    if not path.exists():
        print(f"[MODEL LOAD] Missing path: {path}")
        LOADED_MODEL_CACHE[cache_key] = None
        return None

    class_names = get_class_names(cfg["data_dir"])
    if not class_names:
        print(f"[MODEL LOAD] No class names found in: {cfg['data_dir']}")
        LOADED_MODEL_CACHE[cache_key] = None
        return None

    num_classes = len(class_names)
    model = build_model(cfg["arch"], num_classes)

    try:
        state_dict = load_state_dict_safely(path)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)

        print(f"[MODEL LOAD] Trying: {cfg['name']}")
        print(f"[MODEL LOAD] Missing keys: {len(missing)}")
        print(f"[MODEL LOAD] Unexpected keys: {len(unexpected)}")

        if len(missing) > 0:
            print("[MODEL LOAD] Sample missing:", missing[:10])
        if len(unexpected) > 0:
            print("[MODEL LOAD] Sample unexpected:", unexpected[:10])

        model.to(DEVICE)
        model.eval()

        result = {
            "model": model,
            "class_names": class_names,
            "name": cfg["name"],
            "arch": cfg["arch"],
            "path": str(path),
            "id": cfg["id"],
        }
        print(f"[MODEL LOAD] Loaded usable model: {cfg['name']}")
        LOADED_MODEL_CACHE[cache_key] = result
        return result

    except Exception as e:
        print(f"[MODEL LOAD ERROR] {cfg['name']}: {e}")
        LOADED_MODEL_CACHE[cache_key] = None
        return None
def get_selected_model(selected_model_id):
    if selected_model_id == "buffalo_only":
        return None

    if selected_model_id and selected_model_id != "auto_best":
        cfg = get_model_config_by_id(selected_model_id)
        if cfg:
            loaded = load_model_from_config(cfg)
            return loaded

    for cfg in TRAINED_MODELS:
        loaded = load_model_from_config(cfg)
        if loaded is not None:
            return loaded

    return None


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
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def detect_faces_with_fallback(image):
    detection_sizes = [(640, 640), (512, 512), (448, 448), (384, 384), (320, 320), (256, 256)]
    for size in detection_sizes:
        try:
            face_app.det_model.input_size = size
            faces = face_app.get(image)
            if len(faces) > 0:
                return faces, size
        except Exception:
            continue
    return [], None


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


def find_gallery_image(identity, gallery_entry, uploaded_filename=None):
    if not isinstance(gallery_entry, dict):
        return None

    used_images = gallery_entry.get("used_images", [])
    if not used_images:
        return None

    uploaded_name = Path(uploaded_filename).name.lower() if uploaded_filename else None
    fallback_match = None

    for filename in used_images:
        for root in DATASET_ROOTS:
            candidate1 = root / identity / filename
            candidate2 = root / filename

            for candidate in [candidate1, candidate2]:
                if candidate.exists():
                    if uploaded_name and candidate.name.lower() == uploaded_name:
                        if fallback_match is None:
                            fallback_match = candidate
                        continue
                    return candidate

        for root in DATASET_ROOTS:
            try:
                matches = list(root.rglob(Path(filename).name))
                for m in matches:
                    if uploaded_name and m.name.lower() == uploaded_name:
                        if fallback_match is None:
                            fallback_match = m
                        continue
                    return m
            except Exception:
                pass

    return fallback_match


def img_to_data_uri(path):
    data = Path(path).read_bytes()
    ext = Path(path).suffix.lower().replace(".", "")
    mime = "jpeg" if ext in ("jpg", "jpeg") else ext
    return f"data:image/{mime};base64," + base64.b64encode(data).decode("utf-8")


def draw_results(image, results):
    annotated = image.copy()
    for result in results:
        x1, y1, x2, y2 = result["bbox"]
        name = result["best_name"]
        blur_status = "Blurry" if result["was_blurry"] else "Clear"
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

        if result["confidence_score"] is not None:
            label = f"{name} | Conf: {result['confidence_score']:.1f}% | {blur_status}"
        elif result["similarity_score"] is not None:
            label = f"{name} | Sim: {result['similarity_score']:.3f} | {blur_status}"
        else:
            label = f"{name} | {blur_status}"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        text_y = max(y1 - 10, text_h + 5)
        cv2.rectangle(annotated, (x1, text_y - text_h - 6), (x1 + text_w + 6, text_y + 2), color, -1)
        cv2.putText(annotated, label, (x1 + 3, text_y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return annotated


def run_recognition(image_path, selected_model_id="auto_best"):
    image = cv2.imread(str(image_path))
    if image is None:
        return {"error": "Could not read uploaded image."}

    blurry, blur_score = is_blurry(image, BLUR_THRESHOLD)
    processed_image = image.copy()
    blur_score_after = None

    if blurry:
        processed_image = preprocess_blurry_image(image)
        _, blur_score_after = is_blurry(processed_image, BLUR_THRESHOLD)

    faces, used_det_size = detect_faces_with_fallback(processed_image)
    if not faces:
        return {
            "error": "No face detected in uploaded image.",
            "original_image": url_for("static", filename=f"uploads/{image_path.name}")
        }

    active_model = get_selected_model(selected_model_id)
    print(f"[DEBUG] selected_model_id = {selected_model_id}")
    print(f"[DEBUG] active_model = {active_model['name'] if active_model else 'None'}")

    results = []

    for i, face in enumerate(faces, start=1):
        bbox = face.bbox.astype(int).tolist()
        x1, y1, x2, y2 = bbox
        h, w = processed_image.shape[:2]

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        face_crop = processed_image[y1:y2, x1:x2]

        trained_result = None
        if selected_model_id != "buffalo_only" and active_model is not None and face_crop.size > 0:
            try:
                x = TRANSFORM(face_crop).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    outputs = active_model["model"](x)
                    probs = torch.softmax(outputs, dim=1)
                    conf, pred = torch.max(probs, 1)

                pred_idx = int(pred.item())
                class_names = active_model["class_names"]
                predicted_name = class_names[pred_idx] if 0 <= pred_idx < len(class_names) else str(pred_idx)
                confidence_score = round(float(conf.item()) * 100.0, 2)

                trained_result = {
                    "best_name": predicted_name,
                    "confidence_score": confidence_score,
                    "model_name": active_model["name"],
                    "model_path": active_model["path"],
                    "model_id": active_model["id"],
                }
            except Exception as e:
                print(f"[MODEL ERROR] {selected_model_id}: {e}")
                trained_result = None

        buffalo_similarity_score = None
        buffalo_top_matches = []
        buffalo_best_name = "Unknown"

        if GALLERY is not None:
            try:
                test_embedding = normalize_embedding(face.normed_embedding)
                top_matches_raw = rank_matches(test_embedding, GALLERY, TOP_K)

                if top_matches_raw:
                    best_name_raw, best_score = top_matches_raw[0]
                    buffalo_best_name = best_name_raw if best_score >= MATCH_IF_ABOVE else "Unknown"
                    buffalo_similarity_score = round(float(best_score), 4)

                    for name, score in top_matches_raw:
                        gallery_entry = GALLERY.get(name)
                        image_file = find_gallery_image(name, gallery_entry, uploaded_filename=image_path.name)
                        buffalo_top_matches.append({
                            "name": name,
                            "similarity_score": round(float(score), 4),
                            "similarity_type": "cosine",
                            "image_data": img_to_data_uri(image_file) if image_file and image_file.exists() else None,
                            "source_image": str(image_file) if image_file else None,
                        })
            except Exception as e:
                print(f"[BUFFALO ERROR] {e}")
                buffalo_similarity_score = None
                buffalo_top_matches = []

        if trained_result is not None:
            final_name = trained_result["best_name"]
            final_score_source = "trained_model_confidence"
            confidence_score = trained_result["confidence_score"]
            model_name = trained_result["model_name"]
            model_path = trained_result["model_path"]
            model_id = trained_result["model_id"]
        else:
            final_name = buffalo_best_name
            final_score_source = "buffalo_similarity"
            confidence_score = None
            model_name = "Buffalo"
            model_path = str(GALLERY_PATH) if GALLERY is not None else None
            model_id = "buffalo_only"

        results.append({
            "face_index": i,
            "bbox": [x1, y1, x2, y2],
            "best_name": final_name,
            "confidence_score": confidence_score,
            "similarity_score": buffalo_similarity_score,
            "similarity_type": "cosine" if buffalo_similarity_score is not None else None,
            "score_source": final_score_source,
            "model_name": model_name,
            "model_path": model_path,
            "model_id": model_id,
            "buffalo_best_name": buffalo_best_name,
            "blur_score": round(float(blur_score), 2),
            "blur_score_after": round(float(blur_score_after), 2) if blur_score_after is not None else None,
            "was_blurry": bool(blurry),
            "used_det_size": used_det_size,
            "top_matches": buffalo_top_matches,
        })

    annotated = draw_results(processed_image, results)
    result_file = RESULT_DIR / f"result_{image_path.stem}.jpg"
    cv2.imwrite(str(result_file), annotated)

    return {
        "original_image": url_for("static", filename=f"uploads/{image_path.name}"),
        "annotated_image": url_for("static", filename=f"results/{result_file.name}"),
        "results": results,
        "trained_model_loaded": active_model is not None,
        "trained_model_name": active_model["name"] if active_model is not None else "Buffalo Only",
        "trained_model_path": active_model["path"] if active_model is not None else str(GALLERY_PATH),
        "gallery_exists": GALLERY is not None,
        "selected_model": selected_model_id,
    }


@app.route("/", methods=["GET", "POST"])
def index():
    output = None
    selected_model = "auto_best"

    if request.method == "POST":
        selected_model = request.form.get("selected_model", "auto_best")
        file = request.files.get("image")
        if file and file.filename:
            filename = Path(file.filename).name
            save_path = UPLOAD_DIR / filename
            file.save(save_path)
            output = run_recognition(save_path, selected_model)
        else:
            output = {"error": "Please choose an image file."}

    return render_template(
        "index.html",
        output=output,
        gallery_exists=GALLERY is not None,
        gallery_path=str(GALLERY_PATH),
        model_options=MODEL_OPTIONS,
        selected_model=selected_model,
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)