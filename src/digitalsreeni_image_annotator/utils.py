"""
Utility functions for the Image Annotator application.

This module contains helper functions used across the application.

@DigitalSreeni
Dr. Sreenivas Bhattiprolu
"""

import os

import numpy as np


def models_base_dir() -> str:
    """Return the absolute path of the `models/` directory used for ML weights.

    Resolution strategy (single source of truth used by sam_utils, dino_utils,
    and annotator_window so all three agree on where weights live):

    1. Editable / dev install: package source lives at
       ``<project_root>/src/digitalsreeni_image_annotator/utils.py``, so
       three ``dirname``s up is the project root and ``<root>/models`` is
       the canonical location.
    2. PyPI / site-packages install: the package lives in site-packages, where
       writing model weights would be wrong (system / venv dir, not
       user-visible). Fall back to ``<cwd>/models`` instead.
    """
    pkg_anchor = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    if "site-packages" not in pkg_anchor.replace(os.sep, "/"):
        return os.path.join(pkg_anchor, "models")
    return os.path.join(os.getcwd(), "models")


def calculate_area(annotation):
    if "segmentation" in annotation and annotation["segmentation"] is not None:
        # Polygon area
        x, y = annotation["segmentation"][0::2], annotation["segmentation"][1::2]
        return 0.5 * abs(sum(x[i] * y[i+1] - x[i+1] * y[i] for i in range(-1, len(x)-1)))
    elif "bbox" in annotation:
        # Rectangle area
        x, y, w, h = annotation["bbox"]
        return w * h
    return 0

def calculate_bbox(segmentation):
    x_coordinates, y_coordinates = segmentation[0::2], segmentation[1::2]
    x_min, y_min = min(x_coordinates), min(y_coordinates)
    x_max, y_max = max(x_coordinates), max(y_coordinates)
    width, height = x_max - x_min, y_max - y_min
    return [x_min, y_min, width, height]


def clamp_segmentation(segmentation, width, height):
    """Clamp every vertex of a flat ``[x1, y1, x2, y2, ...]`` polygon into the
    image rectangle ``[0, width] x [0, height]``.

    Per-coordinate clamping (not a shapely intersection) is deliberate for
    manual edits: it preserves the vertex count and ordering so a polygon being
    dragged never loses or splits points mid-edit. See ADR-024. (upstream #32)
    """
    clamped = list(segmentation)
    for i in range(0, len(clamped) - 1, 2):
        clamped[i] = min(max(clamped[i], 0), width)
        clamped[i + 1] = min(max(clamped[i + 1], 0), height)
    return clamped


def clamp_keypoints(keypoints, width, height):
    """Clamp every keypoint of a flat ``[x1, y1, v1, x2, y2, v2, ...]`` list into
    the image rectangle ``[0, width] x [0, height]``, leaving the visibility flag
    untouched. Per-coordinate and count-preserving (mirrors
    :func:`clamp_segmentation`) so a dragged point can't poison saved coords.
    See ADR-024 / ADR-029. (issue #35)"""
    clamped = list(keypoints)
    for i in range(0, len(clamped) - 2, 3):
        clamped[i] = min(max(clamped[i], 0), width)
        clamped[i + 1] = min(max(clamped[i + 1], 0), height)
    return clamped


def keypoint_instance_bbox(keypoints, width=None, height=None, margin=6):
    """Bounding box ``[x, y, w, h]`` around a pose instance's labelled (v>0)
    points, padded by ``margin`` and clamped to the image when its size is
    known. Falls back to a zero box if nothing is labelled. Shared by
    AnnotationController.finish_keypoint (placement) and COCO import
    (recovering a bbox for keypoint annotations whose source file omitted
    one). (issue #35 PR-2)"""
    xs = [keypoints[i] for i in range(0, len(keypoints), 3) if keypoints[i + 2] > 0]
    ys = [keypoints[i + 1] for i in range(0, len(keypoints), 3) if keypoints[i + 2] > 0]
    if not xs or not ys:
        return [0, 0, 0, 0]
    x0, y0 = min(xs) - margin, min(ys) - margin
    x1, y1 = max(xs) + margin, max(ys) + margin
    bbox = [x0, y0, x1 - x0, y1 - y0]
    if width is not None and height is not None:
        bbox = clamp_bbox(bbox, width, height)
    return bbox


def clamp_bbox(bbox, width, height):
    """Trim a ``[x, y, w, h]`` box to the image rectangle by clamping each
    corner independently, keeping it rectangular and at least 1x1. This is the
    **resize** clamp: the dragged edge snaps to the image border while the
    anchored (opposite) edge — already in bounds — stays put. For a **move**
    use :func:`fit_bbox_inside` instead, which preserves size. (upstream #40)"""
    x, y, w, h = bbox
    x0 = min(max(x, 0), width)
    y0 = min(max(y, 0), height)
    x1 = min(max(x + w, 0), width)
    y1 = min(max(y + h, 0), height)
    nx, ny = min(x0, x1), min(y0, y1)
    nw, nh = max(1, abs(x1 - x0)), max(1, abs(y1 - y0))
    if nx + nw > width:
        nx = max(0, width - nw)
    if ny + nh > height:
        ny = max(0, height - nh)
    return [nx, ny, nw, nh]


def fit_bbox_inside(bbox, width, height):
    """Translate a ``[x, y, w, h]`` box back inside the image **preserving its
    size** (shrinking only if it is larger than the image). For the move path:
    a box dragged past any edge should slide back in, not collapse at it the way
    independent-corner clamping (:func:`clamp_bbox`) would. (upstream #40)"""
    x, y, w, h = bbox
    w = min(w, width)
    h = min(h, height)
    x = min(max(x, 0), width - w)
    y = min(max(y, 0), height - h)
    return [x, y, w, h]


def clip_polygon_to_bounds(segmentation, width, height):
    """Clip a flat ``[x1, y1, ...]`` polygon to the image rectangle via a
    shapely intersection, returning the flat exterior of the largest resulting
    polygon, or ``None`` if nothing remains inside the image.

    Unlike :func:`clamp_segmentation`, this geometrically trims the shape (the
    correct choice for augmented data, where a rotated/zoomed polygon should be
    cut at the image edge rather than have stray vertices snapped onto it). The
    ``buffer(0)`` fixes self-intersections an affine augmentation can introduce.
    See ADR-024. (upstream #36)
    """
    from shapely.geometry import Polygon

    points = list(zip(segmentation[0::2], segmentation[1::2]))
    if len(points) < 3:
        return None
    polygon = Polygon(points)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    boundary = Polygon([(0, 0), (width, 0), (width, height), (0, height)])
    clipped = polygon.intersection(boundary)
    if clipped.is_empty:
        return None
    if isinstance(clipped, Polygon):
        chosen = clipped
    else:
        # MultiPolygon / GeometryCollection — keep the largest real polygon.
        polys = [g for g in getattr(clipped, "geoms", []) if isinstance(g, Polygon)]
        if not polys:
            return None
        chosen = max(polys, key=lambda p: p.area)
    if chosen.is_empty or chosen.area == 0:
        return None
    coords = list(chosen.exterior.coords)
    # shapely returns a closed ring (first vertex repeated last); drop it so the
    # output matches the app's unclosed flat-ring convention. shapely emits the
    # closing vertex as the identical tuple, so the `==` here is an exact match.
    if len(coords) > 1 and coords[0] == coords[-1]:
        coords = coords[:-1]
    return [c for point in coords for c in point]


def simplify_polygon(raw_segmentation, detail_pct):
    """Reduce the vertex count of a flat ``[x1, y1, ...]`` polygon (Douglas-
    Peucker via ``cv2.approxPolyDP``) so users can trade detail for smaller
    label files. ``detail_pct`` is 1..100 where **100 = the raw polygon
    unchanged** and lower keeps proportionally fewer vertices. Operates on the
    *raw* (dense) polygon so the result is always derived from full precision
    and is reversible. (upstream #24)

    Returns a flat list (>= 6 coords). Falls back to the raw polygon if it's
    already tiny or simplification can't keep a valid triangle.
    """
    import cv2

    raw = list(raw_segmentation)
    n = len(raw) // 2
    if detail_pct >= 100 or n <= 4:
        return raw

    target = max(3, round(n * detail_pct / 100.0))
    if target >= n:
        return raw

    contour = np.array(raw, dtype=np.float32).reshape(-1, 1, 2)
    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return raw

    # Binary-search the approxPolyDP epsilon for the *richest* polygon whose
    # vertex count is still <= target (i.e. the smallest epsilon meeting the
    # budget; >= 3 points). Higher epsilon -> fewer points.
    lo, hi = 0.0, perimeter
    best = None
    for _ in range(24):
        mid = (lo + hi) / 2.0
        approx = cv2.approxPolyDP(contour, mid, True)
        count = len(approx)
        if count <= target:
            if count >= 3:
                best = approx
            hi = mid  # try to keep more detail (smaller epsilon)
        else:
            lo = mid  # need more simplification
    if best is None:
        return raw
    flat = best.flatten().tolist()
    return flat if len(flat) >= 6 else raw


def normalize_image(image_array):
    """Normalize image array to 8-bit range."""
    if image_array.dtype != np.uint8:
        image_array = ((image_array - image_array.min()) / (image_array.max() - image_array.min()) * 255).astype(np.uint8)
    return image_array

