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

import io
import json
import os
import sys
import traceback

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


# ── helpers ──────────────────────────────────────────────────────────────────

def _log_device():
    try:
        import torch

        if torch.cuda.is_available():
            dev = torch.cuda.get_device_name(0)
            print(f"[SAM] Using CUDA: {torch.version.cuda} — {dev}")
        else:
            print("[SAM] No GPU available, running on CPU")
    except Exception:
        pass


def load_image(image_path: str) -> np.ndarray:
    img = Image.open(image_path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.array(img)


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
    if not polygons:
        return None
    # Return the polygon with the largest area (ignore tiny noise holes)
    biggest = max(polygons, key=lambda p: cv2.contourArea(np.array(p).reshape(-1, 2)))
    return biggest


def _bbox_of_contour(contour: list) -> tuple[float, float, float, float]:
    pts = np.array(contour).reshape(-1, 2)
    return float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max())


def _bbox_area(bbox: list) -> float:
    return float(max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1]))


def _check_points(contour: list, positive: list, negative: list) -> bool:
    """Return True iff all positive points are inside and all negative outside."""
    cnt = np.array(contour, dtype=np.int32).reshape(-1, 1, 2)
    for x, y in positive:
        if cv2.pointPolygonTest(cnt, (float(x), float(y)), False) < 0:
            return False
    for x, y in negative:
        if cv2.pointPolygonTest(cnt, (float(x), float(y)), False) >= 0:
            return False
    return True


def _predicted_bbox_area_ratio(pred_contour: list, user_bbox: list) -> float:
    """Ratio of predicted contour bbox area over user-drawn bbox area."""
    px1, py1, px2, py2 = _bbox_of_contour(pred_contour)
    user_area = _bbox_area(user_bbox)
    if user_area == 0:
        return 0.0
    pred_area = max(0, px2 - px1) * max(0, py2 - py1)
    return pred_area / user_area


# ── core SAM runner ──────────────────────────────────────────────────────────

def run_sam(
    image_path: str,
    model_name: str,
    bboxes: list | None = None,
    points: dict | None = None,
) -> dict:
    from ultralytics import SAM

    _log_device()

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
        user_bbox = None
        positive_pts = pos
        negative_pts = neg
    elif bboxes is not None:
        results = sam_model(image_np, bboxes=[bboxes])
        user_bbox = bboxes
        positive_pts = []
        negative_pts = []
    else:
        return {"error": "No prompts provided."}

    masks = results[0].masks.data.cpu().numpy()
    confidences = results[0].boxes.conf.cpu().numpy()

    best_result = None
    best_score = -1.0

    for i, mask in enumerate(masks):
        contour = mask_to_polygon(mask)
        if contour is None:
            continue

        score = float(confidences[i]) if i < len(confidences) else 0.0
        mask_pixels = int(mask.sum())

        # --- hard constraints ---
        if user_bbox is not None:
            ratio = _predicted_bbox_area_ratio(contour, user_bbox)
            if ratio < 0.20:          # bbox smaller than 20% of drawn box
                continue
            # Also require bbox dimensions within ±20% of user box
            ux, uy, ux2, uy2 = user_bbox
            uw, uh = ux2 - ux, uy2 - uy
            px, py, px2, py2 = _bbox_of_contour(contour)
            pw, ph = px2 - px, py2 - py
            if pw < 0.5 * uw or ph < 0.5 * uh:
                continue
            if pw > 1.5 * uw or ph > 1.5 * uh:
                continue

        if points is not None:
            if not _check_points(contour, positive_pts, negative_pts):
                continue

        # --- ranking: prefer larger masks among passing candidates ---
        if mask_pixels > best_score:
            best_score = mask_pixels
            best_result = {
                "segmentation": contour,
                "score": score,
                "mask_pixels": mask_pixels,
            }

    if best_result is None:
        hints = []
        if user_bbox:
            hints.append("Try drawing a tighter or larger box around the object.")
        if positive_pts or negative_pts:
            hints.append("Try repositioning positive/negative points.")
        hint = " " + " ".join(hints) if hints else ""
        return {"error": f"No SAM mask matches the given constraints.{hint}"}

    return {
        "segmentation": best_result["segmentation"],
        "score": best_result["score"],
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
