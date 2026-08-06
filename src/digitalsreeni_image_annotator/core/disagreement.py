"""Model-vs-ground-truth disagreement and uncertainty scoring (issue #71).

Once a model is trained the app could run it and show predictions, but never
*compare* them with what was already labelled. Two questions an ML engineer
asks constantly went unanswered:

1. **Which of my labels are probably wrong?** A model disagreeing sharply with
   the ground truth is the strongest signal of a labelling mistake available
   without a second annotator.
2. **Which image should I annotate next?** With hundreds of unlabelled images,
   annotating in filename order spends effort on images the model already
   handles.

Qt-free for the same reason as :mod:`core.annotation_qc`: scoring is arithmetic
over two lists of dicts, it is fully testable without a GUI, and keeping it out
of the controller means the matching can be exercised on adversarial cases that
would be painful to construct through a model.

**A high score is a hint, not a verdict.** Nothing here decides an annotation
is wrong; it decides an annotation is worth a human look. Callers must word it
that way too.
"""

from .mask_filters import polygon_iou

# Below this, two shapes are not the same object and pairing them would inflate
# every score with meaningless near-misses.
DEFAULT_MATCH_IOU = 0.3


def _segmentation_of(annotation):
    """A comparable polygon for an annotation, or None.

    A bbox-only annotation is converted to its rectangle so box predictions and
    polygon ground truth still compare. A pose has neither and is excluded —
    see :func:`is_scorable`.
    """
    segmentation = annotation.get("segmentation")
    if segmentation and len(segmentation) >= 6:
        return list(segmentation)
    bbox = annotation.get("bbox")
    if bbox:
        x, y, w, h = bbox
        return [x, y, x + w, y, x + w, y + h, x, y + h]
    return None


def is_scorable(annotation) -> bool:
    """True when an annotation can take part in IoU matching.

    Pose instances cannot: mask IoU is meaningless for a set of keypoints, and
    the right metric there is OKS. Rather than compute a number that looks
    plausible and means nothing, poses are **excluded from v1 scoring** and the
    exclusion is reported to the caller so it can say so in the UI.
    """
    if "keypoints" in annotation and not annotation.get("segmentation"):
        return False
    return _segmentation_of(annotation) is not None


def strip_temp_prefix(class_name: str) -> str:
    """``Temp-cell`` -> ``cell``.

    Predictions arrive under ``Temp-<class>`` names (see
    ``process_yolo_results``). Without this mapping every prediction counts as
    unmatched and *every* image scores badly — the ranking would be pure noise
    while looking entirely reasonable.
    """
    return class_name[5:] if class_name.startswith("Temp-") else class_name


def match_pairs(ground_truth, predictions, iou_threshold=DEFAULT_MATCH_IOU):
    """Assign predictions to ground-truth annotations, maximising total IoU.

    Returns ``(pairs, unmatched_gt, unmatched_pred)`` where ``pairs`` is a list
    of ``(gt_index, pred_index, iou)``.

    **Greedy plus swap improvement, not Hungarian.** ``scipy`` is not a
    dependency and adding one for a single function — on a matrix that is
    almost always under 100x100 and extremely sparse — would be a poor trade.
    Plain greedy is not good enough on its own: it will happily give a
    prediction to the first ground truth it overlaps well and leave a strictly
    better global assignment on the table. The swap pass fixes exactly that by
    exchanging two pairs whenever the exchange raises the total, which recovers
    the optimum on the small conflicting neighbourhoods this data actually
    produces.

    Matching is per class; cross-class pairs are never formed, because a
    prediction of the wrong class *is* a disagreement, not a partial match.
    """
    iou = {}
    for i, gt in enumerate(ground_truth):
        gt_seg = _segmentation_of(gt)
        gt_class = gt.get("category_name")
        for j, pred in enumerate(predictions):
            if strip_temp_prefix(pred.get("category_name", "")) != gt_class:
                continue
            value = polygon_iou(gt_seg, _segmentation_of(pred))
            if value >= iou_threshold:
                iou[(i, j)] = value

    pairs = []
    used_gt, used_pred = set(), set()
    for (i, j), value in sorted(iou.items(), key=lambda kv: kv[1], reverse=True):
        if i in used_gt or j in used_pred:
            continue
        pairs.append([i, j, value])
        used_gt.add(i)
        used_pred.add(j)

    _improve_by_swapping(pairs, iou)

    matched_gt = {p[0] for p in pairs}
    matched_pred = {p[1] for p in pairs}
    unmatched_gt = [i for i in range(len(ground_truth)) if i not in matched_gt]
    unmatched_pred = [j for j in range(len(predictions)) if j not in matched_pred]
    return [tuple(p) for p in pairs], unmatched_gt, unmatched_pred


def _improve_by_swapping(pairs, iou):
    """Exchange partners between two pairs whenever it raises the total IoU.

    Repeated to a fixed point (bounded, since every pass strictly increases a
    quantity with a finite maximum). This is what lifts greedy to the optimum
    on the small conflicting groups real predictions produce.
    """
    improved = True
    while improved:
        improved = False
        for a in range(len(pairs)):
            for b in range(a + 1, len(pairs)):
                (i1, j1, v1), (i2, j2, v2) = pairs[a], pairs[b]
                swapped = iou.get((i1, j2), 0.0) + iou.get((i2, j1), 0.0)
                if swapped > v1 + v2 + 1e-12:
                    pairs[a] = [i1, j2, iou.get((i1, j2), 0.0)]
                    pairs[b] = [i2, j1, iou.get((i2, j1), 0.0)]
                    improved = True


def disagreement_score(
    ground_truth, predictions, iou_threshold=DEFAULT_MATCH_IOU
):
    """How much the model and the human disagree on one image.

    ``score = unmatched_ground_truth + unmatched_predictions + sum(1 - IoU)``
    over the matched pairs. Perfect agreement scores 0. Every term is in the
    same unit — "one object's worth of disagreement" — so a missing label, a
    spurious detection and a badly-fitting pair all contribute comparably.

    Returns ``(score, breakdown)``. The breakdown carries the counts and the
    number of pose instances skipped, so the UI can be honest about coverage
    rather than silently scoring a pose project as perfect.
    """
    gt_all = list(ground_truth or [])
    pred_all = list(predictions or [])
    gt = [a for a in gt_all if is_scorable(a)]
    pred = [a for a in pred_all if is_scorable(a)]
    skipped = (len(gt_all) - len(gt)) + (len(pred_all) - len(pred))

    pairs, unmatched_gt, unmatched_pred = match_pairs(gt, pred, iou_threshold)
    shape_error = sum(1.0 - value for _i, _j, value in pairs)
    score = len(unmatched_gt) + len(unmatched_pred) + shape_error

    return score, {
        "matched": len(pairs),
        "missed": len(unmatched_gt),
        "spurious": len(unmatched_pred),
        "shape_error": shape_error,
        "skipped_pose": skipped,
    }


def uncertainty_score(predictions, boundary=0.5):
    """How unsure the model is about an image it has never been shown labels for.

    Confidence near ``boundary`` is where the model is least decided, so each
    detection contributes ``1 - 2*|conf - boundary|`` — 1.0 at the boundary,
    0.0 at either extreme. Summed, not averaged: an image with ten borderline
    detections teaches more than one with a single borderline detection, and
    averaging would hide that.

    An image with no detections scores 0. That is deliberate: the model seeing
    nothing is not the same as the model being unsure, and conflating them
    floods the top of the ranking with empty images.

    Detections tagged ``source: "sam3-track"`` are excluded — SAM 3 tracking
    writes a constant per-frame confidence of 1.0 (ADR-040), so their
    "certainty" carries no information at all.
    """
    total = 0.0
    counted = 0
    for prediction in predictions or []:
        if prediction.get("source") == "sam3-track":
            continue
        confidence = prediction.get("score")
        if confidence is None:
            continue
        total += max(0.0, 1.0 - 2.0 * abs(float(confidence) - boundary))
        counted += 1
    return total, {"detections": counted}


def rank(scores, descending=True):
    """``[(name, score), ...]`` sorted by score, ties broken by name.

    The name tiebreak keeps the order stable across runs, which matters when
    the ranking is something a person works down over several sessions.
    """
    return sorted(
        (scores or {}).items(),
        key=lambda kv: (-kv[1] if descending else kv[1], kv[0]),
    )
