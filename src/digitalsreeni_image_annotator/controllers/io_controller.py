"""Import / export / save-slices orchestration extracted from `ImageAnnotator`.

The actual format readers and writers live in `io.import_formats` and
`io.export_formats` and are pure functions parameterised on annotation
state. The wrappers here are the UI glue: file dialogs, state mutation
on the main window, status message boxes, auto-save trigger.

Functions take the main window as the first argument so call sites
inside `annotator_window.py` delegate trivially.
"""

import os

from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QFileDialog, QInputDialog, QMessageBox

from ..core.constants import default_class_color
from ..core.keypoint_schema import sanitize_schema

from ..io.export_formats import (
    export_coco_json,
    export_labeled_images,
    export_pascal_voc_bbox,
    export_pascal_voc_both,
    export_semantic_labels,
    export_yolo_v4,
    export_yolo_v5plus,
)
from ..io.import_formats import import_coco_json, process_import_format

from ..core.logging_config import get_logger

logger = get_logger(__name__)


def prompt_validation_split(parent):
    """Ask what fraction of images to hold out for validation.

    Returns ``(val_split, ok)`` from a single shared QInputDialog so the YOLO
    menu export and the in-app YOLO trainer can't drift apart. ``0`` keeps the
    historical all-in-train layout.
    """
    return QInputDialog.getInt(
        parent,
        "Validation Split",
        "Percent of images for the validation set (0 = all in train):",
        20, 0, 100, 5,
    )


def _rebuild_imported_annotation(ann, category_name, number):
    """Rebuild one imported annotation into the app's internal dict shape,
    keeping only known keys. A keypoint instance gets a FULLY SEPARATE shape
    (no ``segmentation``/``type`` keys at all) rather than a shared base dict
    with those keys conditionally added — several existence-only checks
    elsewhere (``"segmentation" in annotation``, not a None-guard) in
    ``image_label.py::draw_annotations``/``start_polygon_edit`` and
    ``eraser_tool.py`` would misfire on a `None`-valued ``segmentation`` key,
    blanking a pose instance's rendering or crashing on the next double-click
    anywhere on the canvas. See ADR-029. (issue #35 PR-2)
    """
    if "keypoints" in ann:
        return {
            "keypoints": ann["keypoints"],
            "num_keypoints": ann.get(
                "num_keypoints",
                sum(1 for j in range(2, len(ann["keypoints"]), 3) if ann["keypoints"][j] > 0),
            ),
            "bbox": ann.get("bbox", [0, 0, 0, 0]),
            "category_id": ann["category_id"],
            "category_name": category_name,
            "number": number,
        }
    return {
        "segmentation": ann.get("segmentation"),
        "bbox": ann.get("bbox"),
        "category_id": ann["category_id"],
        "category_name": category_name,
        "number": number,
        "type": ann.get("type", "polygon"),
    }


def import_annotations(mw):
    if not mw.image_label.check_unsaved_changes():
        return
    logger.debug("Starting import_annotations")
    import_format = mw.import_format_selector.currentText()
    logger.debug(f"Import format: {import_format}")

    if import_format == "COCO JSON":
        file_name, _ = QFileDialog.getOpenFileName(
            mw, "Import COCO JSON Annotations", "", "JSON Files (*.json)"
        )
        if not file_name:
            logger.debug("No file selected, returning")
            return

        logger.debug(f"Selected file: {file_name}")
        json_dir = os.path.dirname(file_name)
        images_dir = os.path.join(json_dir, "images")
        try:
            imported_annotations, image_info, recovered_schemas = import_coco_json(file_name, mw.class_mapping)
        except ValueError as e:
            QMessageBox.warning(mw, "Import Error", str(e))
            return

    elif import_format in ["YOLO (v4 and earlier)", "YOLO (v5+)"]:
        yaml_file, _ = QFileDialog.getOpenFileName(
            mw, "Select YOLO Dataset YAML", "", "YAML Files (*.yaml *.yml)"
        )
        if not yaml_file:
            logger.debug("No YAML file selected, returning")
            return

        logger.debug(f"Selected YAML file: {yaml_file}")
        try:
            imported_annotations, image_info, recovered_schemas = process_import_format(
                import_format, yaml_file, mw.class_mapping
            )
            yaml_dir = os.path.dirname(yaml_file)
            if import_format == "YOLO (v4 and earlier)":
                images_dir = os.path.join(yaml_dir, "train", "images")
            else:
                images_dir = os.path.join(yaml_dir, "images", "train")
        except ValueError as e:
            QMessageBox.warning(mw, "Import Error", str(e))
            return

    else:
        QMessageBox.warning(
            mw,
            "Unsupported Format",
            f"The selected format '{import_format}' is not implemented for import.",
        )
        return

    logger.debug(
        f"JSON/YOLO directory: {json_dir if import_format == 'COCO JSON' else os.path.dirname(yaml_file)}"
    )
    logger.debug(f"Images directory: {images_dir}")
    logger.debug(f"Imported annotations count: {len(imported_annotations)}")
    logger.debug(f"Image info count: {len(image_info)}")

    images_loaded = 0
    images_not_found = []

    for info in image_info.values():
        logger.debug(f"Processing image: {info['file_name']}")
        image_path = os.path.join(images_dir, info["file_name"])

        if os.path.exists(image_path):
            logger.debug(f"Image found at: {image_path}")
            mw.image_paths[info["file_name"]] = image_path
            mw.all_images.append(
                {
                    "file_name": info["file_name"],
                    "height": info["height"],
                    "width": info["width"],
                    "id": info["id"],
                    "is_multi_slice": False,
                }
            )
            images_loaded += 1
        else:
            logger.debug(f"Image not found at: {image_path}")
            images_not_found.append(info["file_name"])

    logger.debug(f"Images loaded: {images_loaded}")
    logger.debug(f"Images not found: {len(images_not_found)}")

    if images_not_found:
        message = f"The following {len(images_not_found)} images were not found in the 'images' directory:\n\n"
        message += "\n".join(images_not_found[:10])
        if len(images_not_found) > 10:
            message += f"\n... and {len(images_not_found) - 10} more."
        message += "\n\nDo you want to proceed and ignore annotations for these missing images?"
        reply = QMessageBox.question(
            mw,
            "Missing Images",
            message,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.No:
            logger.debug("Import cancelled due to missing images")
            QMessageBox.information(
                mw,
                "Import Cancelled",
                "Import cancelled. Please ensure all images are in the 'images' directory and try again.",
            )
            return

    for image_name, annotations in imported_annotations.items():
        if image_name not in mw.image_paths:
            continue
        mw.all_annotations[image_name] = {}
        for category_name, category_annotations in annotations.items():
            mw.all_annotations[image_name][category_name] = []
            for i, ann in enumerate(category_annotations, start=1):
                new_ann = _rebuild_imported_annotation(ann, category_name, i)
                mw.all_annotations[image_name][category_name].append(new_ann)

    for annotations in mw.all_annotations.values():
        for category_name in annotations.keys():
            if category_name not in mw.class_mapping:
                new_id = len(mw.class_mapping) + 1
                mw.class_mapping[category_name] = new_id
                mw.image_label.class_colors[category_name] = QColor(
                    default_class_color(new_id - 1)
                )

    # Register any keypoint schemas recovered from the import (COCO
    # categories, or YOLO-pose's data.yaml kpt_shape/flip_idx). Defensive
    # re-sanitize mirrors project_controller.py's load-time pattern.
    # (issue #35 PR-2)
    for class_name, schema in recovered_schemas.items():
        sanitized = sanitize_schema(schema)
        if sanitized is not None:
            mw.keypoint_schemas[class_name] = sanitized
        else:
            logger.warning(f"Skipped malformed keypoint schema recovered for class '{class_name}' during import.")

    logger.debug("Updating UI")
    mw.update_class_list()
    mw.update_image_list()
    mw.update_annotation_list()

    if mw.image_list.count() > 0:
        mw.image_list.setCurrentRow(0)
        mw.switch_image(mw.image_list.item(0))

    if mw.class_list.count() > 0:
        mw.class_list.setCurrentRow(0)
        mw.on_class_selected()

    mw.image_label.update()

    message = (
        f"Annotations have been imported successfully from "
        f"{file_name if import_format == 'COCO JSON' else yaml_file}.\n"
    )
    message += f"{images_loaded} images were loaded from the 'images' directory.\n"
    if images_not_found:
        message += f"Annotations for {len(images_not_found)} missing images were ignored."

    logger.debug("Import complete, showing message")
    QMessageBox.information(mw, "Import Complete", message)
    mw.auto_save()


def export_annotations(mw):
    if not mw.image_label.check_unsaved_changes():
        return
    export_format = mw.export_format_selector.currentText()

    supported_formats = [
        "COCO JSON",
        "YOLO (v4 and earlier)",
        "YOLO (v5+)",
        "Labeled Images",
        "Semantic Labels",
        "Pascal VOC (BBox)",
        "Pascal VOC (BBox + Segmentation)",
    ]

    if export_format not in supported_formats:
        QMessageBox.warning(
            mw,
            "Unsupported Format",
            f"The selected format '{export_format}' is not implemented.",
        )
        return

    if export_format == "COCO JSON":
        file_name, _ = QFileDialog.getSaveFileName(
            mw, "Export COCO JSON Annotations", "", "JSON Files (*.json)"
        )
    else:
        file_name = QFileDialog.getExistingDirectory(
            mw, f"Select Output Directory for {export_format} Export"
        )

    if not file_name:
        return

    # YOLO training needs a non-empty validation set; let the user choose how
    # much of the data to hold out (0 keeps the historical all-in-train layout).
    val_split = 0
    if export_format in ("YOLO (v4 and earlier)", "YOLO (v5+)"):
        val_split, ok = prompt_validation_split(mw)
        if not ok:
            return

    mw.save_current_annotations()

    if export_format == "COCO JSON":
        output_dir = os.path.dirname(file_name)
        json_filename = os.path.basename(file_name)
        json_file, images_dir = export_coco_json(
            mw.all_annotations,
            mw.class_mapping,
            mw.image_paths,
            mw.slices,
            mw.image_slices,
            output_dir,
            json_filename,
            keypoint_schemas=mw.keypoint_schemas,
        )
        message = "Annotations have been exported successfully in COCO JSON format.\n"
        message += f"JSON file: {json_file}\nImages directory: {images_dir}"

    elif export_format == "YOLO (v4 and earlier)":
        labels_dir, yaml_path = export_yolo_v4(
            mw.all_annotations,
            mw.class_mapping,
            mw.image_paths,
            mw.slices,
            mw.image_slices,
            file_name,
            val_split,
        )
        message = "Annotations have been exported successfully in YOLO (v4 and earlier) format.\n"
        message += f"Labels: {labels_dir}\nYAML: {yaml_path}\nValidation split: {val_split}%"

    elif export_format == "YOLO (v5+)":
        try:
            output_dir, yaml_path = export_yolo_v5plus(
                mw.all_annotations,
                mw.class_mapping,
                mw.image_paths,
                mw.slices,
                mw.image_slices,
                file_name,
                val_split,
                keypoint_schemas=mw.keypoint_schemas,
            )
        except ValueError as e:
            QMessageBox.warning(mw, "Export Error", str(e))
            return
        message = "Annotations have been exported successfully in YOLO (v5+) format.\n"
        message += f"Output directory: {output_dir}\nYAML: {yaml_path}\nValidation split: {val_split}%"

    elif export_format == "Labeled Images":
        labeled_images_dir = export_labeled_images(
            mw.all_annotations,
            mw.class_mapping,
            mw.image_paths,
            mw.slices,
            mw.image_slices,
            file_name,
        )
        message = (
            f"Labeled images have been exported successfully.\n"
            f"Labeled Images: {labeled_images_dir}\n"
        )
        message += (
            f"A class summary has been saved in: "
            f"{os.path.join(labeled_images_dir, 'class_summary.txt')}"
        )

    elif export_format == "Semantic Labels":
        semantic_labels_dir = export_semantic_labels(
            mw.all_annotations,
            mw.class_mapping,
            mw.image_paths,
            mw.slices,
            mw.image_slices,
            file_name,
        )
        message = (
            f"Semantic labels have been exported successfully.\n"
            f"Semantic Labels: {semantic_labels_dir}\n"
        )
        message += (
            f"A class-pixel mapping has been saved in: "
            f"{os.path.join(semantic_labels_dir, 'class_pixel_mapping.txt')}"
        )

    elif export_format == "Pascal VOC (BBox)":
        voc_dir = export_pascal_voc_bbox(
            mw.all_annotations,
            mw.class_mapping,
            mw.image_paths,
            mw.slices,
            mw.image_slices,
            file_name,
        )
        message = "Annotations have been exported successfully in Pascal VOC format (BBox only).\n"
        message += f"Pascal VOC Annotations: {voc_dir}"

    elif export_format == "Pascal VOC (BBox + Segmentation)":
        voc_dir = export_pascal_voc_both(
            mw.all_annotations,
            mw.class_mapping,
            mw.image_paths,
            mw.slices,
            mw.image_slices,
            file_name,
        )
        message = "Annotations have been exported successfully in Pascal VOC format (BBox + Segmentation).\n"
        message += f"Pascal VOC Annotations: {voc_dir}"

    QMessageBox.information(mw, "Export Complete", message)


def export_annotated_frames(mw):
    """Save each annotated frame of the current video as a PNG (issue #48).

    Requires a video to be the active image. Decodes ONE frame at a time via
    the ``VideoHandler`` (the issue-#47 lazy fetch — never bulk pre-extracts)
    and writes each under its frame key, e.g. ``clip_F00003.png``.
    """
    from ..core.video_handler import frame_key

    video = mw.image_controller.current_video()
    if video is None:
        QMessageBox.information(
            mw,
            "Export Annotated Frames",
            "Open a video and select it first, then export its annotated frames.",
        )
        return
    base_name, handler, _info = video

    directory = QFileDialog.getExistingDirectory(
        mw, "Select Output Directory for Annotated Frames"
    )
    if not directory:
        return

    # Flush the current frame's in-progress annotations so it's counted.
    mw.save_current_annotations()

    indices = sorted(mw.image_controller.annotated_frame_indices(base_name))
    written = 0
    failed = 0
    for idx in indices:
        qimage = handler.get_frame(idx)  # one frame at a time (lazy, #47)
        # Count a frame as failed if it can't be decoded OR the PNG write
        # fails (qimage.save returns False), so the summary can't claim
        # success for a frame that silently vanished.
        if qimage is not None and qimage.save(
            os.path.join(directory, f"{frame_key(base_name, idx)}.png"), "PNG"
        ):
            written += 1
        else:
            failed += 1

    message = (
        f"{written} annotated frame{'' if written == 1 else 's'} "
        f"written to:\n{directory}"
    )
    if failed:
        message += (
            f"\n\n{failed} frame{'' if failed == 1 else 's'} could not be "
            "decoded or written."
        )
    QMessageBox.information(mw, "Export Complete", message)


def save_slices(mw, directory):
    # Decode ONLY the annotated slices (issue #45): iterate names first (no
    # pixel work) and materialise a slice's QImage only when it has
    # annotations, instead of decoding every slice of every stack.
    from ..core.slice_cache import slice_names

    slices_saved = False
    for image_file, image_slices in mw.image_slices.items():
        getter = getattr(image_slices, "get", None)
        name_to_qimage = None  # lazily built only for plain-list collections
        for slice_name in slice_names(image_slices):
            if not mw.all_annotations.get(slice_name):
                continue
            if getter is not None:
                qimage = getter(slice_name)
            else:
                if name_to_qimage is None:
                    name_to_qimage = {n: q for n, q in image_slices}
                qimage = name_to_qimage.get(slice_name)
            if qimage is not None:
                qimage.save(os.path.join(directory, f"{slice_name}.png"), "PNG")
                slices_saved = True
    return slices_saved
