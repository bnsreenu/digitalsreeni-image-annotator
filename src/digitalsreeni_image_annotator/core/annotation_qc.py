"""Rule-based annotation quality audit (issue #70).

The app could describe a dataset but never tell you something was *wrong* with
it. A self-intersecting polygon, a duplicate annotation, a class named ``cell``
next to one named ``Cell``, a shape one pixel wide — all of them export
cleanly, train quietly, and degrade the model with no warning anywhere.

**This module must stay Qt-free.** That is not a style preference: the headless
CLI's ``validate`` command (issue #76) imports these rules to run label quality
as a CI gate, and it cannot require a display. Core raises, the UI boundary
catches and renders (ADR-031) — so there is no ``QMessageBox`` here for a bad
polygon and no ``QImage`` for a dimension. Image sizes arrive as plain tuples.

The engine returns plain data (:class:`Finding`), never widgets. What a finding
*means* is the caller's business; what makes it a finding is this module's.

A note on what is deliberately **not** auto-fixed: an area outlier might be a
genuinely large object, and a near-duplicate might be two real objects that
touch. Only unambiguous repairs — repairing self-intersection, recomputing a
bbox from its own polygon, clamping into bounds — are offered.
"""

from dataclasses import dataclass, field
from statistics import median

from ..utils import calculate_area, calculate_bbox
from .mask_filters import polygon_iou

# --- severities ------------------------------------------------------------

SEVERITY_ERROR = "error"      # will export wrong or train wrong
SEVERITY_WARNING = "warning"  # probably a mistake, worth a human look
SEVERITY_INFO = "info"        # observation about the dataset, not a defect

_SEVERITY_ORDER = {SEVERITY_ERROR: 0, SEVERITY_WARNING: 1, SEVERITY_INFO: 2}

# --- rule identifiers ------------------------------------------------------

RULE_SELF_INTERSECTING = "self_intersecting"
RULE_TOO_FEW_VERTICES = "too_few_vertices"
RULE_DEGENERATE_AREA = "degenerate_area"
RULE_OUT_OF_BOUNDS = "out_of_bounds"
RULE_BBOX_MISMATCH = "bbox_mismatch"
RULE_NEAR_DUPLICATE = "near_duplicate"
RULE_CROSS_CLASS_OVERLAP = "cross_class_overlap"
RULE_AREA_OUTLIER = "area_outlier"
RULE_CLASS_IMBALANCE = "class_imbalance"
RULE_EMPTY_IMAGE = "empty_image"
RULE_SIMILAR_CLASS_NAMES = "similar_class_names"
RULE_ORPHAN_TEMP_CLASS = "orphan_temp_class"
RULE_POSE_POINT_OUTSIDE_BBOX = "pose_point_outside_bbox"
RULE_POSE_COUNT_MISMATCH = "pose_count_mismatch"


@dataclass
class QCConfig:
    """Thresholds for the audit. Every one is a judgement call, so every one is
    configurable rather than hidden in a constant."""

    duplicate_iou: float = 0.9
    cross_class_iou: float = 0.7
    outlier_factor: float = 8.0     # x the class median area
    imbalance_ratio: float = 20.0   # largest class / smallest class, by count
    min_area: float = 1.0           # px^2 below which a shape has no substance
    bbox_tolerance: float = 2.0     # px of disagreement before it is a finding
    name_edit_distance: int = 1     # <= this many edits apart is suspicious
    check_empty_images: bool = True


@dataclass
class Finding:
    """One problem, in plain data. No widgets, no Qt types."""

    rule: str
    severity: str
    message: str
    image: str | None = None
    class_name: str | None = None
    annotation_number: int | None = None
    fixable: bool = False
    # Free-form rule-specific payload (the partner annotation of a duplicate,
    # the class counts behind an imbalance). Kept out of `message` so a caller
    # can act on it rather than parse prose.
    detail: dict = field(default_factory=dict)

    def sort_key(self):
        return (_SEVERITY_ORDER.get(self.severity, 9), self.rule, self.image or "")


# --- small pure helpers ----------------------------------------------------


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance, iterative two-row.

    Hand-rolled rather than pulled in as a dependency: this is the only place
    the codebase needs it, and it is fifteen lines.
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb))
            )
        previous = current
    return previous[-1]


def is_pose(annotation) -> bool:
    """True for a pose instance.

    Identified by the **absence** of a segmentation key, which is the
    discriminator the whole app routes on (ADR-029). Polygon rules must skip
    these entirely; they get their own group.
    """
    return "keypoints" in annotation and not annotation.get("segmentation")


def _polygon_is_valid(segmentation):
    """``(is_valid, reason)`` for a flat coordinate ring."""
    from shapely.geometry import Polygon

    points = list(zip(segmentation[0::2], segmentation[1::2]))
    if len(points) < 3:
        return False, "fewer than 3 vertices"
    polygon = Polygon(points)
    if not polygon.is_valid:
        return False, "self-intersecting outline"
    return True, ""


def _iter_annotations(all_annotations):
    """``(image, class_name, annotation)`` over the whole project.

    Iterates the annotations mapping directly, which is keyed by image *and*
    slice name — so slices of a multi-dimensional image and video frames are
    included for free. Iterating ``all_images`` instead would silently skip
    every slice, the same trap ``_collect_dino_batch_work_items`` exists to
    avoid.
    """
    for image, by_class in (all_annotations or {}).items():
        for class_name, annotations in (by_class or {}).items():
            for annotation in annotations or []:
                yield image, class_name, annotation


# --- individual rule groups ------------------------------------------------


def check_geometry(all_annotations, image_sizes, config):
    findings = []
    for image, class_name, annotation in _iter_annotations(all_annotations):
        number = annotation.get("number")
        if is_pose(annotation):
            continue

        segmentation = annotation.get("segmentation")
        if segmentation:
            vertex_count = len(segmentation) // 2
            if vertex_count < 3:
                findings.append(Finding(
                    RULE_TOO_FEW_VERTICES, SEVERITY_ERROR,
                    f"Polygon has only {vertex_count} vertices.",
                    image, class_name, number,
                ))
                continue
            valid, reason = _polygon_is_valid(segmentation)
            if not valid:
                findings.append(Finding(
                    RULE_SELF_INTERSECTING, SEVERITY_ERROR,
                    f"Invalid polygon: {reason}.",
                    image, class_name, number, fixable=True,
                ))

        area = calculate_area(annotation)
        if area < config.min_area:
            findings.append(Finding(
                RULE_DEGENERATE_AREA, SEVERITY_ERROR,
                f"Annotation has effectively no area ({area:.2f} px2).",
                image, class_name, number,
            ))

        size = image_sizes.get(image) if image_sizes else None
        if size:
            width, height = size
            if _out_of_bounds(annotation, width, height):
                findings.append(Finding(
                    RULE_OUT_OF_BOUNDS, SEVERITY_ERROR,
                    f"Coordinates fall outside the {width}x{height} image.",
                    image, class_name, number, fixable=True,
                ))

        if segmentation and annotation.get("bbox") is not None:
            derived = calculate_bbox(segmentation)
            stored = annotation["bbox"]
            if any(
                abs(a - b) > config.bbox_tolerance for a, b in zip(derived, stored)
            ):
                findings.append(Finding(
                    RULE_BBOX_MISMATCH, SEVERITY_WARNING,
                    "Stored bounding box does not match the outline.",
                    image, class_name, number, fixable=True,
                    detail={"stored": list(stored), "derived": derived},
                ))
    return findings


def _out_of_bounds(annotation, width, height):
    segmentation = annotation.get("segmentation")
    if segmentation:
        xs, ys = segmentation[0::2], segmentation[1::2]
        if xs and (min(xs) < 0 or max(xs) > width):
            return True
        if ys and (min(ys) < 0 or max(ys) > height):
            return True
    keypoints = annotation.get("keypoints")
    if keypoints:
        for i in range(0, len(keypoints) - 2, 3):
            if keypoints[i + 2] <= 0:
                continue  # v=0 points are padding pinned at (0, 0)
            if not (0 <= keypoints[i] <= width and 0 <= keypoints[i + 1] <= height):
                return True
    bbox = annotation.get("bbox")
    if bbox and not segmentation and not keypoints:
        x, y, w, h = bbox
        if x < 0 or y < 0 or x + w > width or y + h > height:
            return True
    return False


def check_pose(all_annotations, image_sizes, config):
    """Pose-specific rules. Mask IoU and polygon validity are meaningless for a
    keypoint instance, so these are their own group rather than a special case
    inside the geometry rules."""
    findings = []
    for image, class_name, annotation in _iter_annotations(all_annotations):
        if not is_pose(annotation):
            continue
        keypoints = annotation.get("keypoints") or []
        number = annotation.get("number")

        labelled = sum(1 for i in range(2, len(keypoints), 3) if keypoints[i] > 0)
        stated = annotation.get("num_keypoints")
        if stated is not None and stated != labelled:
            findings.append(Finding(
                RULE_POSE_COUNT_MISMATCH, SEVERITY_WARNING,
                f"num_keypoints says {stated} but {labelled} points are labelled.",
                image, class_name, number, fixable=True,
            ))

        bbox = annotation.get("bbox")
        if bbox:
            x, y, w, h = bbox
            outside = [
                i // 3
                for i in range(0, len(keypoints) - 2, 3)
                if keypoints[i + 2] > 0
                and not (x <= keypoints[i] <= x + w and y <= keypoints[i + 1] <= y + h)
            ]
            if outside:
                findings.append(Finding(
                    RULE_POSE_POINT_OUTSIDE_BBOX, SEVERITY_WARNING,
                    f"{len(outside)} keypoint(s) lie outside the instance box.",
                    image, class_name, number, fixable=True,
                    detail={"indices": outside},
                ))
    return findings


def check_redundancy(all_annotations, image_sizes, config):
    """Near-duplicates within a class and heavy cross-class overlap.

    Pairwise IoU is O(n²), so ``polygon_iou`` rejects on bounding boxes before
    doing any polygon work — on an image with thousands of annotations almost
    every pair is nowhere near its partner.
    """
    findings = []
    for image, by_class in (all_annotations or {}).items():
        flat = [
            (class_name, annotation)
            for class_name, annotations in (by_class or {}).items()
            for annotation in annotations or []
            if annotation.get("segmentation")
        ]
        for i in range(len(flat)):
            class_a, ann_a = flat[i]
            for j in range(i + 1, len(flat)):
                class_b, ann_b = flat[j]
                iou = polygon_iou(ann_a["segmentation"], ann_b["segmentation"])
                if class_a == class_b:
                    if iou >= config.duplicate_iou:
                        findings.append(Finding(
                            RULE_NEAR_DUPLICATE, SEVERITY_WARNING,
                            f"Almost identical to annotation "
                            f"#{ann_b.get('number')} (IoU {iou:.2f}).",
                            image, class_a, ann_a.get("number"),
                            detail={"other": ann_b.get("number"), "iou": iou},
                        ))
                elif iou >= config.cross_class_iou:
                    findings.append(Finding(
                        RULE_CROSS_CLASS_OVERLAP, SEVERITY_WARNING,
                        f"Overlaps '{class_b}' annotation "
                        f"#{ann_b.get('number')} heavily (IoU {iou:.2f}).",
                        image, class_a, ann_a.get("number"),
                        detail={"other_class": class_b, "iou": iou},
                    ))
    return findings


def check_statistics(all_annotations, image_sizes, config):
    """Area outliers, class imbalance and unannotated images.

    These are informational by design. An outlier might be a genuinely large
    object; an imbalance might be the real distribution. Reporting is useful,
    auto-fixing would be wrong.
    """
    findings = []

    areas_by_class = {}
    counts = {}
    for image, class_name, annotation in _iter_annotations(all_annotations):
        counts[class_name] = counts.get(class_name, 0) + 1
        if not is_pose(annotation):
            areas_by_class.setdefault(class_name, []).append(
                (image, annotation, calculate_area(annotation))
            )

    for class_name, entries in areas_by_class.items():
        areas = [area for _image, _ann, area in entries if area > 0]
        if len(areas) < 4:
            continue  # a median over three samples says nothing
        class_median = median(areas)
        if class_median <= 0:
            continue
        for image, annotation, area in entries:
            if area > class_median * config.outlier_factor:
                findings.append(Finding(
                    RULE_AREA_OUTLIER, SEVERITY_INFO,
                    f"Area {area:.0f} px2 is {area / class_median:.1f}x the "
                    f"median for '{class_name}'.",
                    image, class_name, annotation.get("number"),
                    detail={"area": area, "median": class_median},
                ))

    real_counts = {
        name: count
        for name, count in counts.items()
        if not name.startswith("Temp-")
    }
    if len(real_counts) > 1:
        largest = max(real_counts.items(), key=lambda kv: kv[1])
        smallest = min(real_counts.items(), key=lambda kv: kv[1])
        if smallest[1] > 0 and largest[1] / smallest[1] >= config.imbalance_ratio:
            findings.append(Finding(
                RULE_CLASS_IMBALANCE, SEVERITY_INFO,
                f"'{largest[0]}' has {largest[1]} annotations but "
                f"'{smallest[0]}' has {smallest[1]}.",
                detail={"counts": real_counts},
            ))

    if config.check_empty_images and image_sizes:
        for image in image_sizes:
            by_class = (all_annotations or {}).get(image) or {}
            if not any(annotations for annotations in by_class.values()):
                findings.append(Finding(
                    RULE_EMPTY_IMAGE, SEVERITY_INFO,
                    "No annotations on this image.",
                    image,
                ))
    return findings


def check_hygiene(all_annotations, class_names, config):
    """Class-name typos and orphaned review classes.

    ``cell`` next to ``Cell`` next to ``cells`` is three classes to a trainer
    and one class to a human, and nothing else in the app notices.
    """
    findings = []
    names = sorted(set(class_names or []))

    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if a.startswith("Temp-") or b.startswith("Temp-"):
                continue
            if a.lower() == b.lower():
                findings.append(Finding(
                    RULE_SIMILAR_CLASS_NAMES, SEVERITY_WARNING,
                    f"'{a}' and '{b}' differ only by case.",
                    detail={"names": [a, b]},
                ))
            elif edit_distance(a.lower(), b.lower()) <= config.name_edit_distance:
                findings.append(Finding(
                    RULE_SIMILAR_CLASS_NAMES, SEVERITY_WARNING,
                    f"'{a}' and '{b}' are nearly identical names.",
                    detail={"names": [a, b]},
                ))

    for name in names:
        if name.startswith("Temp-"):
            findings.append(Finding(
                RULE_ORPHAN_TEMP_CLASS, SEVERITY_ERROR,
                f"'{name}' is a leftover review class from an interrupted "
                "detection run.",
                class_name=name,
            ))
    return findings


# --- entry point -----------------------------------------------------------


def run_audit(all_annotations, image_sizes=None, class_names=None, config=None):
    """Run every rule over a project and return findings, most severe first.

    ``all_annotations`` is ``{image_name: {class_name: [annotation, ...]}}``.
    ``image_sizes`` is ``{image_name: (width, height)}``; images missing from
    it simply skip the bounds rules rather than failing — a caller that cannot
    resolve every size should still get the rest of the audit.
    ``class_names`` defaults to the classes that actually appear in the
    annotations, which is the right answer for a CLI run with no project UI.
    """
    config = config or QCConfig()
    image_sizes = image_sizes or {}
    if class_names is None:
        class_names = sorted(
            {name for _i, name, _a in _iter_annotations(all_annotations)}
        )

    findings = []
    findings += check_geometry(all_annotations, image_sizes, config)
    findings += check_pose(all_annotations, image_sizes, config)
    findings += check_redundancy(all_annotations, image_sizes, config)
    findings += check_statistics(all_annotations, image_sizes, config)
    findings += check_hygiene(all_annotations, class_names, config)
    findings.sort(key=Finding.sort_key)
    return findings


def summarise(findings):
    """``{severity: count}`` plus a ``total``. Used by the dialog header and by
    the CLI's exit-code decision."""
    summary = {SEVERITY_ERROR: 0, SEVERITY_WARNING: 0, SEVERITY_INFO: 0}
    for finding in findings:
        if finding.severity in summary:
            summary[finding.severity] += 1
    summary["total"] = len(findings)
    return summary


# --- repairs ---------------------------------------------------------------


def apply_fix(annotation, rule, width=None, height=None):
    """Repair one annotation in place for one rule. Returns True if it changed.

    Only unambiguous repairs are implemented — anything requiring a judgement
    call about what the user *meant* stays a report, never an edit.
    """
    if rule == RULE_SELF_INTERSECTING:
        return _fix_self_intersection(annotation)
    if rule == RULE_BBOX_MISMATCH:
        if annotation.get("segmentation"):
            annotation["bbox"] = calculate_bbox(annotation["segmentation"])
            return True
        return False
    if rule == RULE_OUT_OF_BOUNDS:
        return _fix_out_of_bounds(annotation, width, height)
    if rule == RULE_POSE_COUNT_MISMATCH:
        keypoints = annotation.get("keypoints") or []
        annotation["num_keypoints"] = sum(
            1 for i in range(2, len(keypoints), 3) if keypoints[i] > 0
        )
        return True
    if rule == RULE_POSE_POINT_OUTSIDE_BBOX:
        from ..utils import keypoint_instance_bbox

        keypoints = annotation.get("keypoints") or []
        if not keypoints:
            return False
        annotation["bbox"] = keypoint_instance_bbox(keypoints, width, height)
        return True
    return False


def _fix_self_intersection(annotation):
    """``buffer(0)`` is shapely's idiom for repairing an invalid ring. Keeps the
    largest resulting piece: a bow-tie repairs into two lobes, and silently
    turning one annotation into two would be a bigger surprise than losing the
    smaller lobe."""
    from shapely.geometry import MultiPolygon, Polygon

    segmentation = annotation.get("segmentation")
    if not segmentation or len(segmentation) < 6:
        return False
    polygon = Polygon(list(zip(segmentation[0::2], segmentation[1::2])))
    if polygon.is_valid:
        return False
    repaired = polygon.buffer(0)
    if repaired.is_empty:
        return False
    if isinstance(repaired, MultiPolygon):
        repaired = max(repaired.geoms, key=lambda p: p.area)
    coords = list(repaired.exterior.coords)
    if len(coords) > 1 and coords[0] == coords[-1]:
        coords = coords[:-1]  # the app's rings are unclosed
    annotation["segmentation"] = [c for point in coords for c in point]
    if annotation.get("bbox") is not None:
        annotation["bbox"] = calculate_bbox(annotation["segmentation"])
    return True


def _fix_out_of_bounds(annotation, width, height):
    from ..utils import clamp_bbox, clamp_keypoints, clamp_segmentation

    if width is None or height is None:
        return False
    changed = False
    if annotation.get("segmentation"):
        clamped = clamp_segmentation(annotation["segmentation"], width, height)
        changed = clamped != annotation["segmentation"]
        annotation["segmentation"] = clamped
    if annotation.get("keypoints"):
        clamped = clamp_keypoints(annotation["keypoints"], width, height)
        changed = changed or clamped != annotation["keypoints"]
        annotation["keypoints"] = clamped
    if annotation.get("bbox") is not None:
        if annotation.get("segmentation"):
            annotation["bbox"] = calculate_bbox(annotation["segmentation"])
        else:
            clamped = clamp_bbox(annotation["bbox"], width, height)
            changed = changed or clamped != annotation["bbox"]
            annotation["bbox"] = clamped
    return changed
