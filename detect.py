"""
Real-Time Mask Wearing Detection
Pipeline: OpenCV (webcam) → YOLOv8 (detect person) → ResNet-50 (classify mask)
"""

import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os

# ── Config ──────────────────────────────────────────────────────────────────
RESNET_WEIGHTS = "models/resnet_mask.pth"
CLASSES        = ["Mask", "No Mask"]
CONF_THRESHOLD = 0.5          # YOLO person confidence threshold
IMG_SIZE       = 224           # ResNet input size

# Colors: green = mask, red = no mask
COLORS = {"Mask": (0, 200, 0), "No Mask": (0, 0, 220)}
WINDOW_NAME = "Mask Detection"

# ── Load Models ─────────────────────────────────────────────────────────────
def load_resnet(weights_path: str) -> nn.Module:
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    return model


def load_yolo() -> YOLO:
    # Downloads yolov8n.pt automatically on first run (~6 MB)
    return YOLO("yolov8n.pt")


# ── Image Transform ──────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


# ── Classify crop with ResNet ────────────────────────────────────────────────
def classify_crop(model: nn.Module, crop_bgr: np.ndarray) -> tuple[str, float]:
    """Return (label, confidence) for a cropped BGR image."""
    pil = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
    tensor = transform(pil).unsqueeze(0)          # (1, 3, 224, 224)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]
    idx   = probs.argmax().item()
    return CLASSES[idx], float(probs[idx])


# ── Draw overlay ─────────────────────────────────────────────────────────────
def draw_box(frame: np.ndarray, box, label: str, conf: float):
    x1, y1, x2, y2 = map(int, box)
    color = COLORS.get(label, (255, 255, 255))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    text = f"{label} {conf:.0%}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, text, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)


# ── Main loop ────────────────────────────────────────────────────────────────
def main():
    if not os.path.exists(RESNET_WEIGHTS):
        print(f"[ERROR] ResNet weights not found at '{RESNET_WEIGHTS}'.")
        print("        Please run  python train.py  first.")
        return

    print("[INFO] Loading models …")
    yolo   = load_yolo()
    resnet = load_resnet(RESNET_WEIGHTS)
    print("[INFO] Models loaded. Close the video window to quit.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. YOLO: detect persons
        results = yolo(frame, classes=[0], conf=CONF_THRESHOLD, verbose=False)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Guard against out-of-bounds crops
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 <= x1 or y2 <= y1:
                    continue

                crop = frame[y1:y2, x1:x2]

                # 2. ResNet: classify mask status
                label, conf = classify_crop(resnet, crop)

                # 3. Draw result
                draw_box(frame, (x1, y1, x2, y2), label, conf)

        cv2.imshow(WINDOW_NAME, frame)
        cv2.waitKey(20)
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
