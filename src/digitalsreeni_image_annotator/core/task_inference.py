"""Infer the training task and summarise the dataset from annotations (#73).

What kind of model to train is not a decision the user should be asked to make:
it is entailed by what they annotated. Boxes only means detect, polygons mean
segment, keypoints mean pose. Asking is asking them to restate the data.

**One function, two consumers.** ``train_model`` already infers the task a
second time — from the prepared dataset YAML (``kpt_shape`` present means pose)
— and raises pre-flight if the loaded model's ``.task`` disagrees. A mismatch
between what the dialog says it is training and what the trainer decides to
train is a bug by construction, so both derive from the same rules here.

Qt-free: this is arithmetic over the annotations dict, and it is the sort of
thing that should be exhaustively unit-tested on hand-built inputs rather than
through a dialog.
"""

import os

TASK_DETECT = "detect"
TASK_SEGMENT = "segment"
TASK_POSE = "pose"


def _is_pose(annotation):
    """Pose instances are identified by the **absence** of a segmentation key,
    the discriminator the whole app routes on (ADR-029)."""
    return "keypoints" in annotation and not annotation.get("segmentation")


def infer_task(all_annotations):
    """``(task, reason)`` for a project's annotations.

    Precedence is pose > segment > detect, and that order is deliberate rather
    than arbitrary: a pose instance cannot be trained as anything else, and a
    polygon carries strictly more information than the box it implies. Where
    shapes are mixed the reason says so, because "segment" on a project that is
    90 % boxes is a surprise worth explaining up front.

    An empty project yields ``(None, reason)`` — there is nothing to train, and
    guessing a default would produce a confusing failure later.
    """
    counts = count_shapes(all_annotations)
    if counts["pose"]:
        if counts["polygon"] or counts["bbox"]:
            return TASK_POSE, (
                f"{counts['pose']} pose instance(s) present; "
                f"{counts['polygon'] + counts['bbox']} other shape(s) cannot be "
                "trained alongside them"
            )
        return TASK_POSE, f"{counts['pose']} pose instance(s)"
    if counts["polygon"]:
        if counts["bbox"]:
            return TASK_SEGMENT, (
                f"{counts['polygon']} polygon(s) and {counts['bbox']} box-only "
                "annotation(s); boxes train as their bounding rectangle"
            )
        return TASK_SEGMENT, f"{counts['polygon']} polygon(s)"
    if counts["bbox"]:
        return TASK_DETECT, f"{counts['bbox']} bounding box(es), no polygons"
    return None, "no annotations to train on"


def count_shapes(all_annotations):
    """``{"polygon", "bbox", "pose"}`` counts across the project.

    Iterates the annotations mapping, which is keyed by image *and* slice name,
    so slices count too.
    """
    counts = {"polygon": 0, "bbox": 0, "pose": 0}
    for by_class in (all_annotations or {}).values():
        for class_name, annotations in (by_class or {}).items():
            if class_name.startswith("Temp-"):
                continue  # pending review, not training data
            for annotation in annotations or []:
                if _is_pose(annotation):
                    counts["pose"] += 1
                elif annotation.get("segmentation"):
                    counts["polygon"] += 1
                elif annotation.get("bbox"):
                    counts["bbox"] += 1
    return counts


def summarise_dataset(all_annotations, image_names):
    """Live figures for the training dialog's Data row.

    ``unlabelled`` is the one that matters: a project where most images have no
    annotations trains badly, and the number is invisible until someone counts
    it. Surfacing it before the run is much cheaper than discovering it after.
    """
    names = list(image_names or [])
    annotated = 0
    for name in names:
        by_class = (all_annotations or {}).get(name) or {}
        if any(
            annotations
            for class_name, annotations in by_class.items()
            if not class_name.startswith("Temp-")
        ):
            annotated += 1

    counts = count_shapes(all_annotations)
    classes = sorted(
        {
            class_name
            for by_class in (all_annotations or {}).values()
            for class_name, annotations in (by_class or {}).items()
            if annotations and not class_name.startswith("Temp-")
        }
    )
    return {
        "images": len(names),
        "annotated_images": annotated,
        "unlabelled_images": len(names) - annotated,
        "annotations": sum(counts.values()),
        "classes": classes,
        "shape_counts": counts,
    }


def pose_training_blockers(all_annotations, keypoint_schemas):
    """Reasons a pose project cannot be exported for YOLO-pose training.

    YOLO-pose carries **one dataset-global** ``kpt_shape``, so a project mixing
    pose classes of different K — or pose alongside non-pose — cannot be
    expressed at all. Detected here so the dialog can refuse *before* the run,
    with the same actionable message ``_pose_export_check`` produces, rather
    than failing opaquely deep inside Ultralytics.

    Returns a list of human-readable strings; empty means clear to train.
    """
    blockers = []
    counts = count_shapes(all_annotations)
    if not counts["pose"]:
        return blockers

    if counts["polygon"] or counts["bbox"]:
        blockers.append(
            f"The project mixes {counts['pose']} pose instance(s) with "
            f"{counts['polygon'] + counts['bbox']} polygon/box annotation(s). "
            "YOLO-pose datasets cannot contain both."
        )

    pose_classes = {
        class_name
        for by_class in (all_annotations or {}).values()
        for class_name, annotations in (by_class or {}).items()
        if any(_is_pose(a) for a in annotations or [])
    }
    ks = {}
    for class_name in pose_classes:
        schema = (keypoint_schemas or {}).get(class_name)
        if schema and schema.get("names"):
            ks[class_name] = len(schema["names"])
    if len(set(ks.values())) > 1:
        detail = ", ".join(f"'{name}' K={k}" for name, k in sorted(ks.items()))
        blockers.append(
            "YOLO-pose needs one keypoint count for the whole dataset, but the "
            f"pose classes disagree: {detail}."
        )
    return blockers


def _base_name(file_name):
    return os.path.splitext(file_name or "")[0]


def trainable_image_names(all_images, slice_names_by_base=None):
    """The names a training export will actually write labels for.

    A stack or a video **is not itself an image**: it contributes its slices or
    frames, which is what carries the annotations and what the exporter writes.
    Counting the parent entries instead reported a video with 27 annotated
    frames as "368 annotation(s) across 0 of 1 image(s)" -- both halves wrong,
    and the second one alarming enough to look like data loss.
    """
    by_base = slice_names_by_base or {}
    names = []
    for info in all_images or []:
        file_name = info.get("file_name")
        if not file_name:
            continue
        if info.get("is_multi_slice") or info.get("is_video"):
            names.extend(by_base.get(_base_name(file_name), []))
        else:
            names.append(file_name)
    return names


def unresolvable_stack_blockers(
    all_images, loaded_stack_bases=(), annotated_names=()
):
    """Stacks or videos whose slices cannot be resolved to pixels.

    **Not** "stacks and videos are unsupported". They are supported: a video's
    ``image_slices[base]`` is an ordinary ``LazySliceList`` and the exporters
    resolve slice pixels through it (issues #45/#47 -- ``core.video_handler``
    says so in as many words). Annotate frames of a video and the YOLO export
    writes them like any other image.

    This function previously blocked every stack and video outright, on the
    strength of a note that predated slice-aware export, and so refused
    perfectly good datasets: 368 polygons across a video's frames, and training
    was unavailable with a message claiming videos cannot be used.

    What genuinely cannot be exported is a stack that **has annotations** but
    whose slices were never materialised -- the exporter would skip those
    annotations with nothing but a log line, silently training on less data
    than the user believes.

    An unannotated stack contributes no keys to the export and so cannot break
    anything: refusing to train because a 4 GB CZI is sitting unopened in the
    project would be a refusal with no failure behind it.
    """
    loaded = set(loaded_stack_bases or ())
    annotated = set(annotated_names or ())

    blocked = []
    for info in all_images or []:
        if not (info.get("is_multi_slice") or info.get("is_video")):
            continue
        base = _base_name(info.get("file_name"))
        if base in loaded:
            continue
        prefix = base + "_"
        if not any(key == base or key.startswith(prefix) for key in annotated):
            continue
        blocked.append(info.get("file_name"))

    if not blocked:
        return []
    listed = ", ".join(str(name) for name in blocked[:5])
    if len(blocked) > 5:
        listed += f", and {len(blocked) - 5} more"
    return [
        f"{len(blocked)} annotated stack(s)/video(s) have no loaded slices, so "
        f"their annotations cannot be exported: {listed}. Re-open the "
        "project, or check the file is still where it was and that its "
        "dimensions were confirmed."
    ]
