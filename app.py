import os
import io
import base64
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
import tempfile

import torch
import torch.nn as nn
from torchvision import models, transforms
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 3 classes — order must match training (ImageFolder sorts alphabetically)
CLASSES  = ["non_recyclable", "recyclable", "unknown"]
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224

# ── Model ─────────────────────────────────────────────────────────────────
def build_model():
    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 3),   # 3 classes
    )
    return model

def load_model():
    model = build_model().to(DEVICE)
    checkpoint = torch.load("model/best_model.pth", map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

MODEL = load_model()

# ── Grad-CAM ──────────────────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model):
        self.model       = model
        self.gradients   = None
        self.activations = None
        target_layer = model.features[-1]
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        output[0, class_idx].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam).squeeze().cpu().numpy()
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam

GRADCAM = GradCAM(MODEL)

# ── Helpers ───────────────────────────────────────────────────────────────
TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

def make_overlay(img_np, cam):
    cam_resized = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (0.55 * img_np + 0.45 * heatmap).astype(np.uint8)
    return overlay

def img_to_b64(img_np):
    pil = Image.fromarray(img_np)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def analyse_pil(img_pil):
    img_resized = img_pil.resize((IMG_SIZE, IMG_SIZE))
    img_np  = np.array(img_resized)
    input_t = TRANSFORM(img_pil).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output   = MODEL(input_t)
        probs    = torch.softmax(output, dim=1)[0]
        pred_idx = probs.argmax().item()
        label    = CLASSES[pred_idx]
        confidence = probs[pred_idx].item()

    with torch.enable_grad():
        cam = GRADCAM.generate(input_t, pred_idx)

    overlay = make_overlay(img_np, cam)

    # Map to frontend-friendly fields
    is_recyclable = label == "recyclable"
    is_unknown    = label == "unknown"

    return {
        "label":       label,
        "confidence":  round(confidence * 100, 1),
        "recyclable":  is_recyclable,
        "unknown":     is_unknown,
        "original":    img_to_b64(img_np),
        "heatmap":     img_to_b64(overlay),
        "probs": {
            "recyclable":     round(probs[1].item() * 100, 1),
            "non_recyclable": round(probs[0].item() * 100, 1),
            "unknown":        round(probs[2].item() * 100, 1),
        }
    }

# ── Routes ────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    try:
        img_pil = Image.open(file.stream).convert("RGB")
        return jsonify(analyse_pil(img_pil))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict_video", methods=["POST"])
def predict_video():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    try:
        suffix = Path(file.filename).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        cap          = cv2.VideoCapture(tmp_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps          = cap.get(cv2.CAP_PROP_FPS) or 25
        duration_s   = total_frames / fps

        n_samples = min(8, max(1, total_frames))
        indices   = np.linspace(0, total_frames - 1, n_samples, dtype=int)

        results = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil   = Image.fromarray(frame_rgb)
            r = analyse_pil(img_pil)
            r["timestamp"] = round(idx / fps, 1)
            results.append(r)

        cap.release()
        os.unlink(tmp_path)

        n_recyclable = sum(1 for r in results if r["recyclable"])
        n_non        = sum(1 for r in results if not r["recyclable"] and not r["unknown"])
        n_unknown    = sum(1 for r in results if r["unknown"])
        known_frames = [r for r in results if not r["unknown"]]
        if len(known_frames) == 0:
            overall = "unknown"
        else:
            n_rec_known = sum(1 for r in known_frames if r["recyclable"])
            overall = "recyclable" if n_rec_known > len(known_frames) / 2 else "non_recyclable"

        return jsonify({
            "frames":       results,
            "overall":      overall,
            "duration":     round(duration_s, 1),
            "n_recyclable": n_recyclable,
            "n_total":      len(results),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5001)