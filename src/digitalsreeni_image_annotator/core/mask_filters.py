"""Noise filters for unprompted mask proposals (issue #69).

An unprompted SAM pass over a busy image can return several hundred masks —
every cell, every shadow, every bit of background texture. Handed straight to
the canvas that is not a feature, it is an unusable review screen. These
filters ship *with* the feature rather than as a follow-up for exactly that
reason.

Qt-free and pure: filtering is arithmetic over polygons, and keeping it out of
the controller means the thresholds can be tested exhaustively without a model,
a canvas or a QApplication.
"""

from ..utils import calculate_area, calculate_bbox

# Tag carried by every unprompted proposal. Lives here rather than on the
# controller so the canvas and the review filter can recognise one without
# importing a controller (and creating a cycle).
SAM_EVERYTHING_SOURCE = "sam-everything"

# Defaults chosen to be permissive: the filters exist to stop the canvas
# drowning, not to second-guess what the user is annotating.
DEFAULT_MIN_AREA = 100.0        # px^2 -- below this it is speckle, not an object
DEFAULT_MAX_AREA_FRACTION = 0.5  # a mask over half the image is the background
# A pathological-output valve, NOT a routine filter. At 100 it was the latter:
# a dense field of cells legitimately produces 120+ masks, and the cap threw
# away perfectly good ones purely for being 101st by score. Everything that
# survives the area and overlap filters is a real object the user asked for,
# so the only job left here is to stop a degenerate run (fine texture, thousands
# of fragments) from freezing the canvas. There is no UI for it: the controller
# reads it once at construction, so it is a constant with a rationale rather
# than something the user can turn -- worth exposing if the value ever needs
# arguing about again.
DEFAULT_MAX_CANDIDATES = 500
DEFAULT_OVERLAP_IOU = 0.5        # already-annotated regions are not proposals


def _boxes_overlap(a, b):
    """Cheap AABB rejection before any polygon work."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


def polygon_iou(seg_a, seg_b):
    """Intersection-over-union of two flat ``[x1, y1, ...]`` polygons.

    Bounding boxes are compared first and a non-overlap short-circuits to 0.0:
    on a few hundred candidates against a few hundred existing annotations the
    pairwise polygon intersection is the expensive part, and most pairs are
    nowhere near each other.

    Invalid geometry is repaired with ``buffer(0)`` rather than raising — a
    self-intersecting proposal should still be comparable.
    """
    from shapely.geometry import Polygon

    if not seg_a or not seg_b or len(seg_a) < 6 or len(seg_b) < 6:
        return 0.0
    if not _boxes_overlap(calculate_bbox(seg_a), calculate_bbox(seg_b)):
        return 0.0

    poly_a = Polygon(list(zip(seg_a[0::2], seg_a[1::2])))
    poly_b = Polygon(list(zip(seg_b[0::2], seg_b[1::2])))
    if not poly_a.is_valid:
        poly_a = poly_a.buffer(0)
    if not poly_b.is_valid:
        poly_b = poly_b.buffer(0)
    if poly_a.is_empty or poly_b.is_empty:
        return 0.0

    intersection = poly_a.intersection(poly_b).area
    if intersection == 0:
        return 0.0
    union = poly_a.area + poly_b.area - intersection
    return intersection / union if union else 0.0


def filter_mask_proposals(
    proposals,
    image_width,
    image_height,
    existing_segmentations=(),
    min_area=DEFAULT_MIN_AREA,
    max_area_fraction=DEFAULT_MAX_AREA_FRACTION,
    max_candidates=DEFAULT_MAX_CANDIDATES,
    overlap_iou=DEFAULT_OVERLAP_IOU,
):
    """Reduce raw mask proposals to a reviewable set.

    Applied in this order, cheapest first:

    1. **Area bounds** — speckle below ``min_area`` and any mask covering more
       than ``max_area_fraction`` of the image (that one is the background,
       which SAM reliably proposes and nobody wants).
    2. **Overlap with what is already annotated** — a proposal matching an
       existing annotation above ``overlap_iou`` is re-proposing work already
       done.
    3. **Count cap**, applied *after* sorting by score descending, so the cap
       keeps the best candidates rather than whichever ones the model happened
       to emit first.

    Returns ``(kept, dropped_counts)`` where ``dropped_counts`` breaks down the
    rejections by reason. The caller reports that breakdown: a silent "120
    masks became 40" reads as a bug, whereas naming the reason turns the
    thresholds into something the user can actually tune.
    """
    image_area = float(image_width) * float(image_height)
    max_area = image_area * max_area_fraction if image_area > 0 else float("inf")

    dropped = {"too_small": 0, "too_large": 0, "overlapping": 0, "over_limit": 0}
    kept = []

    for proposal in proposals or []:
        segmentation = proposal.get("segmentation")
        if not segmentation or len(segmentation) < 6:
            dropped["too_small"] += 1
            continue
        area = calculate_area({"segmentation": segmentation})
        if area < min_area:
            dropped["too_small"] += 1
            continue
        if area > max_area:
            dropped["too_large"] += 1
            continue
        if any(
            polygon_iou(segmentation, existing) >= overlap_iou
            for existing in existing_segmentations
        ):
            dropped["overlapping"] += 1
            continue
        kept.append(proposal)

    kept.sort(key=lambda p: p.get("score", 0.0), reverse=True)
    if max_candidates is not None and len(kept) > max_candidates:
        dropped["over_limit"] = len(kept) - max_candidates
        kept = kept[:max_candidates]

    return kept, dropped


def describe_dropped(dropped):
    """Human-readable summary of a ``dropped_counts`` mapping, or ``""``."""
    labels = {
        "too_small": "below the minimum area",
        "too_large": "above the maximum area",
        "overlapping": "overlapping existing annotations",
        "over_limit": "beyond the candidate limit",
    }
    parts = [f"{count} {labels[key]}" for key, count in dropped.items() if count]
    return ", ".join(parts)
