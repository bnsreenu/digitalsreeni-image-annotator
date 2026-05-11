"""
Standalone SAM worker — runs in an isolated subprocess.

This script is intentionally free of PyQt5 imports so it can load
torch/ultralytics in a clean process where the parent GUI's loaded
DLLs do not interfere.

Communication:
  stdin  -> JSON request (image path + model + prompts)
  stdout -> JSON response (polygon + score or error)
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import traceback
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


MODELS = {
    "SAM 2 tiny": "sam2_t.pt",
    "SAM 2 small": "sam2_s.pt",
    "SAM 2 base": "sam2_b.pt",
    "SAM 2 large": "sam2_l.pt",
    "SAM 2.1 tiny": "sam2.1_t.pt",
    "SAM 2.1 small": "sam2.1_s.pt",
    "SAM 2.1 base": "sam2.1_b.pt",
    "SAM 2.1 large": "sam2.1_l.pt",
}


def mask_to_polygon(mask: np.ndarray) -> list | None:
    contours, _ = cv2.findContours(
        (mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    polygons = []
    for contour in contours:
        if cv2.contourArea(contour) > 10:
            polygon = contour.flatten().tolist()
            if len(polygon) >= 6:
                polygons.append(polygon)
    return polygons[0] if polygons else None


def load_image(image_path: str) -> np.ndarray:
    img = Image.open(image_path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.array(img)


def run_sam(
    image_path: str,
    model_name: str,
    bboxes: list | None = None,
    points: dict | None = None,
) -> dict:
    from ultralytics import SAM

    model_file = MODELS[model_name]
    sam_model = SAM(model_file)
    image_np = load_image(image_path)

    if points is not None:
        pos = points.get("positive", [])
        neg = points.get("negative", [])
        all_points = [pos + neg]
        all_labels = [([1] * len(pos)) + ([0] * len(neg))]
        if not all_points[0]:
            return {"error": "No points provided."}
        results = sam_model(image_np, points=all_points, labels=all_labels)
    elif bboxes is not None:
        results = sam_model(image_np, bboxes=[bboxes])
    else:
        return {"error": "No prompts provided."}

    mask = results[0].masks.data[0].cpu().numpy()
    if mask is None:
        return {"error": "No mask returned."}

    contour = mask_to_polygon(mask)
    if contour is None:
        return {"error": "No valid contours found in mask."}

    return {
        "segmentation": contour,
        "score": float(results[0].boxes.conf[0]),
    }


def main():
    raw = sys.stdin.read()
    if not raw.strip():
        return

    try:
        request = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(json.dumps({"error": f"Invalid JSON: {exc}"}))
        return

    image_path = request.get("image_path")
    model_name = request.get("model_name", "SAM 2 tiny")
    bboxes = request.get("bboxes")
    points = request.get("points")

    try:
        result = run_sam(image_path, model_name, bboxes=bboxes, points=points)
    except Exception:
        result = {"error": traceback.format_exc()}

    print(json.dumps(result))


if __name__ == "__main__":
    main()
