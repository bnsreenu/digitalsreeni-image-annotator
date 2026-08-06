"""``sreeni-cli`` command implementations (issue #76, ADR-041).

Each command is a plain function taking parsed args and returning an exit code.
Nothing here imports Qt, and only :func:`run_predict` imports torch — lazily, so
``validate`` stays fast enough to run on every commit.
"""

import json
import os
import sys

from .main import EXIT_ERROR, EXIT_FINDINGS, EXIT_OK, EXPORT_FORMATS, IMPORT_FORMATS

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def _stderr(message):
    """Progress narration. stderr so stdout stays pipeable."""
    print(message, file=sys.stderr)


def _load_project(path):
    from ..core.project_io import ProjectReadError, load_project

    try:
        return load_project(path), EXIT_OK
    except ProjectReadError as exc:
        _stderr(f"error: {exc}")
        return None, EXIT_ERROR


def _export_dispatch(label, project, out_dir, val_split):
    """Call the export function matching ``label``.

    Slices are passed as empty collections: the CLI supports what the project
    already materialised, and extracting new slices from a stack runs through
    the Qt-bound ``ImageController``. Documented limit, not a silent gap —
    :func:`run_export` reports the count it skipped.
    """
    from ..io import export_formats

    common = (
        project.all_annotations,
        project.class_mapping,
        project.image_paths,
        [],
        {},
        out_dir,
    )
    if label == "COCO JSON":
        return export_formats.export_coco_json(*common)
    if label == "YOLO (v4 and earlier)":
        return export_formats.export_yolo_v4(*common, val_split=val_split)
    if label == "YOLO (v5+)":
        return export_formats.export_yolo_v5plus(
            *common, val_split=val_split,
            keypoint_schemas=project.keypoint_schemas,
        )
    if label == "Pascal VOC (BBox)":
        return export_formats.export_pascal_voc_bbox(*common)
    if label == "Pascal VOC (BBox + Segmentation)":
        return export_formats.export_pascal_voc_both(*common)
    if label == "Labeled Images":
        return export_formats.export_labeled_images(*common)
    if label == "Semantic Labels":
        return export_formats.export_semantic_labels(*common)
    raise ValueError(f"Unsupported export format: {label}")


def run_export(args):
    """Export a project to an annotation format."""
    project, code = _load_project(args.project)
    if project is None:
        return code

    if project.missing_images:
        # A partial export that looks complete is worse than a refusal: the
        # dataset would train on fewer images than the user believes.
        _stderr(
            f"error: {len(project.missing_images)} image(s) referenced by the "
            "project could not be found on disk:"
        )
        for name in project.missing_images[:10]:
            _stderr(f"  {name}")
        if len(project.missing_images) > 10:
            _stderr(f"  ... and {len(project.missing_images) - 10} more")
        return EXIT_ERROR

    skipped = len(project.slice_names())
    if skipped:
        _stderr(
            f"note: {skipped} slice(s) of multi-dimensional images are not "
            "exported headlessly - slice extraction needs the GUI."
        )

    os.makedirs(args.out, exist_ok=True)
    label = EXPORT_FORMATS[args.format]

    # No split warning here, deliberately (ADR-044). The GUI raises one wherever
    # a percentage is chosen, but headlessly it would have nothing to say: slice
    # and frame pixels are unavailable, so `_is_exportable` drops those names
    # before the split ever sees them, and every name that survives is a file on
    # disk and therefore its own group. There is no leaky split to warn about
    # because there is no group larger than one image — which the `note:` above
    # about unexported slices already tells the user.
    _stderr(f"Exporting {len(project.image_paths)} image(s) as {label}...")
    try:
        _export_dispatch(label, project, args.out, args.val_split)
    except Exception as exc:
        _stderr(f"error: export failed: {exc}")
        return EXIT_ERROR

    print(args.out)
    return EXIT_OK


def run_convert(args):
    """Convert between annotation formats without a project."""
    from ..io.import_formats import process_import_format

    source_label = IMPORT_FORMATS[args.source_format]
    target_label = EXPORT_FORMATS[args.target_format]
    _stderr(f"Reading {source_label} from {args.source}...")
    try:
        annotations, image_info, _schemas = process_import_format(
            source_label, args.source, {}
        )
    except Exception as exc:
        _stderr(f"error: could not read the input: {exc}")
        return EXIT_ERROR

    images_dir = args.images or _guess_images_dir(args.source)
    image_paths = {}
    missing = []
    for info in image_info.values():
        file_name = info.get("file_name")
        if not file_name:
            continue
        # Exact key match first; a substring fallback would attach "bee.jpg"
        # annotations to "honeybee.jpg" (CLAUDE.md).
        candidate = os.path.join(images_dir, file_name)
        if os.path.exists(candidate):
            image_paths[file_name] = candidate
        else:
            missing.append(file_name)
    if missing:
        _stderr(
            f"warning: {len(missing)} image(s) not found under {images_dir}; "
            "formats that copy images will skip them."
        )

    class_mapping = {}
    for by_class in annotations.values():
        for class_name in by_class:
            class_mapping.setdefault(class_name, len(class_mapping) + 1)

    project = _StandaloneProject(annotations, class_mapping, image_paths)
    os.makedirs(args.out, exist_ok=True)
    _stderr(f"Writing {target_label} to {args.out}...")
    try:
        _export_dispatch(target_label, project, args.out, 0)
    except Exception as exc:
        _stderr(f"error: conversion failed: {exc}")
        return EXIT_ERROR

    print(args.out)
    return EXIT_OK


class _StandaloneProject:
    """The subset of :class:`LoadedProject` the export functions read.

    ``convert`` has no project file, so this stands in — same attribute names,
    so the dispatch above needs no special case.
    """

    def __init__(self, annotations, class_mapping, image_paths):
        self.all_annotations = annotations
        self.class_mapping = class_mapping
        self.image_paths = image_paths
        self.keypoint_schemas = {}

    def slice_names(self):
        return []


def _guess_images_dir(source):
    """Where the images probably live relative to an annotation input."""
    base = source if os.path.isdir(source) else os.path.dirname(source)
    for candidate in ("images", "JPEGImages"):
        path = os.path.join(base, candidate)
        if os.path.isdir(path):
            return path
    return base


def run_validate(args):
    """Run the QC rules and exit non-zero on findings.

    This is the command that turns label quality into a CI gate, which is why
    the exit code is the primary output and the JSON report is optional.
    """
    from ..core import annotation_qc

    project, code = _load_project(args.project)
    if project is None:
        return code

    findings = annotation_qc.run_audit(
        project.all_annotations,
        image_sizes=project.image_sizes(),
        class_names=project.class_names(),
    )
    summary = annotation_qc.summarise(findings)

    payload = {
        "project": os.path.abspath(args.project),
        "summary": summary,
        "findings": [
            {
                "rule": f.rule,
                "severity": f.severity,
                "message": f.message,
                "image": f.image,
                "class_name": f.class_name,
                "annotation_number": f.annotation_number,
                "fixable": f.fixable,
                "detail": f.detail,
            }
            for f in findings
        ],
    }

    if args.json_report:
        with open(args.json_report, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        _stderr(f"Report written to {args.json_report}")

    print(json.dumps(summary))
    for finding in findings:
        location = finding.image or "project"
        _stderr(f"[{finding.severity}] {location}: {finding.message}")

    return EXIT_FINDINGS if _should_fail(summary, args.fail_on) else EXIT_OK


def _should_fail(summary, fail_on):
    """Whether ``validate`` should exit non-zero.

    ``--fail-on warning`` includes errors, ``--fail-on info`` includes
    everything: severities are a scale, so the threshold is inclusive-upward.
    That is what makes the flag useful for tightening a gate over time.
    """
    from ..core import annotation_qc

    if fail_on == "never":
        return False
    thresholds = {
        "error": [annotation_qc.SEVERITY_ERROR],
        "warning": [annotation_qc.SEVERITY_ERROR, annotation_qc.SEVERITY_WARNING],
        "info": [
            annotation_qc.SEVERITY_ERROR,
            annotation_qc.SEVERITY_WARNING,
            annotation_qc.SEVERITY_INFO,
        ],
    }
    return any(summary.get(level, 0) for level in thresholds[fail_on])


def run_predict(args):
    """Run a model over a folder of images.

    The only command that needs torch and Ultralytics, and it imports them
    here so the other three never pay for it.
    """
    if not os.path.isdir(args.images):
        _stderr(f"error: not a directory: {args.images}")
        return EXIT_ERROR

    image_files = sorted(
        name for name in os.listdir(args.images)
        if name.lower().endswith(IMAGE_EXTENSIONS)
    )
    if not image_files:
        _stderr(f"error: no images found in {args.images}")
        return EXIT_ERROR

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        _stderr(f"error: ultralytics is not available: {exc}")
        return EXIT_ERROR

    try:
        model = YOLO(args.model)
    except Exception as exc:
        _stderr(f"error: could not load {args.model}: {exc}")
        return EXIT_ERROR

    annotations = {}
    image_paths = {}
    class_mapping = {}
    for index, file_name in enumerate(image_files, start=1):
        path = os.path.join(args.images, file_name)
        _stderr(f"[{index}/{len(image_files)}] {file_name}")
        try:
            results = model(path, conf=args.conf, save=False, verbose=False)
        except Exception as exc:
            _stderr(f"  warning: prediction failed: {exc}")
            continue
        image_paths[file_name] = path
        annotations[file_name] = _results_to_annotations(results, class_mapping)

    project = _StandaloneProject(annotations, class_mapping, image_paths)
    os.makedirs(args.out, exist_ok=True)
    label = EXPORT_FORMATS["coco" if args.format == "coco" else "yolov5"]
    try:
        _export_dispatch(label, project, args.out, 0)
    except Exception as exc:
        _stderr(f"error: could not write predictions: {exc}")
        return EXIT_ERROR

    print(args.out)
    return EXIT_OK


def run_doctor(args):
    """Report the Qt environment and diagnose a broken PyQt6 install (issue #92).

    Imports nothing from PyQt6 -- that is the entire point. The command has to work in
    the environment where the GUI cannot start, which is exactly the environment where
    importing Qt raises or kills the process.

    ``error`` and ``suspect`` findings fail the command -- anything that could explain
    a Qt that will not load. A ``warning`` is a forecast ("a second Qt is on the path
    but its version currently matches"), worth printing and not worth failing a build
    over.
    """
    from ..core.qt_diagnostics import (
        FAILING_SEVERITIES, diagnose, format_report, qt_environment,
    )

    # No `qt_failed`: this is a proactive preflight, so rules whose evidence only
    # means something after an actual failure stay quiet (ADR-046). The MSVC state is
    # still printed in the report -- it is just not a finding.
    env = qt_environment()
    findings = diagnose(env)
    print(format_report(env, findings))
    if any(finding.severity in FAILING_SEVERITIES for finding in findings):
        return EXIT_ERROR
    return EXIT_OK


def _results_to_annotations(results, class_mapping):
    """Ultralytics results -> the app's per-class annotation dict."""
    by_class = {}
    number = 0
    for result in results or []:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        names = getattr(result, "names", {}) or {}
        masks = getattr(result, "masks", None)
        for index, box in enumerate(boxes):
            class_id = int(box.cls)
            class_name = names.get(class_id, str(class_id))
            class_mapping.setdefault(class_name, len(class_mapping) + 1)
            number += 1
            annotation = {
                "category_id": class_mapping[class_name],
                "category_name": class_name,
                "score": float(box.conf),
                "number": number,
            }
            if masks is not None and index < len(masks.xy):
                annotation["segmentation"] = [
                    float(c) for point in masks.xy[index] for c in point
                ]
                annotation["type"] = "polygon"
            else:
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
                annotation["bbox"] = [x1, y1, x2 - x1, y2 - y1]
                annotation["type"] = "rectangle"
            by_class.setdefault(class_name, []).append(annotation)
    return by_class
