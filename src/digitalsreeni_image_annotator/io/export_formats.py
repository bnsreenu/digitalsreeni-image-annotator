import json
from PyQt6.QtGui import QImage
from ..core.keypoint_schema import schema_k
from ..utils import calculate_area, calculate_bbox
import yaml
import hashlib
import os
import shutil
import tempfile
import xml.etree.ElementTree as ET
from xml.dom import minidom
from datetime import datetime

import numpy as np
import skimage.draw
from PIL import Image


# Utility function to handle the COCO conversion for all export formats
def convert_to_coco(all_annotations, class_mapping, image_paths, slices, image_slices, keypoint_schemas=None):
    with tempfile.TemporaryDirectory() as temp_dir:
        json_file_path, images_dir = export_coco_json(all_annotations, class_mapping, image_paths, slices, image_slices, temp_dir, keypoint_schemas=keypoint_schemas)
        
        with open(json_file_path, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)
        
    return coco_data, images_dir



def _coco_category(name, cat_id, keypoint_schemas):
    """A plain {id, name} category, plus COCO-keypoints fields for a pose
    class. ``skeleton`` is 1-based per the COCO ``person_keypoints`` spec;
    ``flip_idx`` has no COCO precedent and is kept 0-based (its only
    consumers — this app's own importer and the PR-3 trainer — are
    0-based, so converting it would just add a pointless round-trip).
    (issue #35 PR-2)"""
    cat = {"id": cat_id, "name": name}
    schema = (keypoint_schemas or {}).get(name)
    if schema:
        cat["keypoints"] = list(schema["names"])
        cat["skeleton"] = [[a + 1, b + 1] for a, b in schema["skeleton"]]
        cat["flip_idx"] = list(schema["flip_idx"])
    return cat


def export_coco_json(all_annotations, class_mapping, image_paths, slices, image_slices, output_dir, json_filename=None, keypoint_schemas=None):
    coco_format = {
        "images": [],
        "categories": [_coco_category(name, id, keypoint_schemas) for name, id in class_mapping.items()],
        "annotations": []
    }
    
    # Create images directory
    images_dir = os.path.join(output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    
    annotation_id = 1
    image_id = 1
    # Create a mapping of slice names to their QImage objects
    slice_map = {slice_name: qimage for slice_name, qimage in slices}
    
    # Handle all images and slices
    for image_name, annotations in all_annotations.items():
        # Skip if there are no annotations for this image/slice
        if not annotations:
            continue

        # Check if it's a slice (either in slice_map or has underscores and no file extension)
        is_slice = image_name in slice_map or ('_' in image_name and '.' not in image_name)
        
        if is_slice:
            qimage = slice_map.get(image_name)
            if qimage is None:
                # If the slice is not in slice_map, it might be a CZI slice or a TIFF slice
                # Find the corresponding QImage in slices or image_slices
                matching_slices = [s for s in slices if s[0] == image_name]
                if matching_slices:
                    qimage = matching_slices[0][1]
                else:
                    # Check in image_slices
                    for stack_slices in image_slices.values():
                        matching_slices = [s for s in stack_slices if s[0] == image_name]
                        if matching_slices:
                            qimage = matching_slices[0][1]
                            break
                if qimage is None:
                    print(f"No image data found for slice {image_name}, skipping")
                    continue
            file_name_img = f"{image_name}.png"
            # Save the QImage as a file
            save_path = os.path.join(images_dir, file_name_img)
            if not os.path.exists(save_path):
                qimage.save(save_path)
            else:
                print(f"Image {file_name_img} already exists in the target directory. Skipping save.")
        else:
            # Check if the image_name exists in image_paths
            image_path = next((path for name, path in image_paths.items() if image_name in name), None)
            if not image_path:
                print(f"No image path found for {image_name}, skipping")
                continue
            if image_path.lower().endswith(('.tif', '.tiff', '.czi')):
                print(f"Skipping main tiff/czi file: {image_name}")
                continue
            file_name_img = image_name
            # Copy the image file
            dst_path = os.path.join(images_dir, file_name_img)
            if not os.path.exists(dst_path):
                shutil.copy2(image_path, dst_path)
            else:
                print(f"Image {file_name_img} already exists in the target directory. Skipping copy.")

        image_info = {
            "file_name": file_name_img,
            "height": qimage.height() if is_slice else QImage(image_path).height(),
            "width": qimage.width() if is_slice else QImage(image_path).width(),
            "id": image_id
        }
        coco_format["images"].append(image_info)
        
        for class_name, class_annotations in annotations.items():
            for ann in class_annotations:
                coco_ann = create_coco_annotation(ann, image_id, annotation_id, class_name, class_mapping)
                coco_format["annotations"].append(coco_ann)
                annotation_id += 1
        
        image_id += 1

    # Generate JSON filename if not provided
    if json_filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_filename = f"annotations_{timestamp}.json"
    elif not json_filename.lower().endswith('.json'):
        json_filename += '.json'

    # Save COCO JSON file
    json_file_path = os.path.join(output_dir, json_filename)
    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(coco_format, f, indent=2)

    return json_file_path, images_dir


def create_coco_annotation(ann, image_id, annotation_id, class_name, class_mapping):
    coco_ann = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": class_mapping[class_name],
        "area": calculate_area(ann),
        "iscrowd": 0
    }
    
    if "keypoints" in ann:
        # Checked before segmentation/bbox — a keypoint instance also
        # carries a bbox, so bbox must not be checked first. No
        # "segmentation" key: the app has no mask for a pose instance
        # (ADR-029). (issue #35 PR-2)
        flat = list(ann["keypoints"])
        for i in range(2, len(flat), 3):
            flat[i] = int(flat[i])  # pycocotools expects an int visibility flag
        coco_ann["keypoints"] = flat
        coco_ann["num_keypoints"] = int(ann.get(
            "num_keypoints",
            sum(1 for i in range(2, len(flat), 3) if flat[i] > 0),
        ))
        coco_ann["bbox"] = ann.get("bbox", [0, 0, 0, 0])
    elif "segmentation" in ann:
        coco_ann["segmentation"] = [ann["segmentation"]]
        coco_ann["bbox"] = calculate_bbox(ann["segmentation"])
    elif "bbox" in ann:
        coco_ann["bbox"] = ann["bbox"]

    return coco_ann



def assign_train_val(image_names, val_pct):
    """Deterministically partition image names into (train_set, val_set).

    val_pct in [0, 100]; 0 -> everything in train (the original behaviour).
    Ordering uses a stable filename hash so the split is reproducible across
    runs and machines (unlike the built-in hash() which is salted per process).
    The val count is the nearest integer to the requested fraction, clamped so
    the val set is never accidentally empty: whenever val_pct > 0 and there are
    >= 2 annotated images, at least one image lands in val and at least one
    stays in train.
    """
    names = list(image_names)
    if val_pct <= 0 or len(names) < 2:
        return set(names), set()
    ordered = sorted(names, key=lambda n: hashlib.md5(n.encode("utf-8")).hexdigest())
    n = len(ordered)
    # round() is half-to-even, which is fine here; the clamp keeps both sides
    # non-empty regardless of how the nearest-integer falls.
    val_count = max(1, min(n - 1, round(n * val_pct / 100)))
    val = set(ordered[:val_count])
    return set(names) - val, val


def export_yolo_v4(all_annotations, class_mapping, image_paths, slices, image_slices, output_dir, val_split=0):
    # Create output directories
    train_dir = os.path.join(output_dir, 'train')
    valid_dir = os.path.join(output_dir, 'valid')
    for dir_path in [train_dir, valid_dir]:
        os.makedirs(os.path.join(dir_path, 'images'), exist_ok=True)
        os.makedirs(os.path.join(dir_path, 'labels'), exist_ok=True)

    # Create a mapping of class names to YOLO indices
    class_to_index = {name: i for i, name in enumerate(class_mapping.keys())}

    # Create a mapping of slice names to their QImage objects
    slice_map = {slice_name: qimage for slice_name, qimage in slices}

    # Deterministically split the annotated images into train/val.
    annotated = [name for name, ann in all_annotations.items() if ann]
    _, val_names = assign_train_val(annotated, val_split)

    for image_name, annotations in all_annotations.items():
        # Skip if there are no annotations for this image/slice
        if not annotations:
            continue

        # Route this image into the train or val directory.
        split_dir = valid_dir if image_name in val_names else train_dir
        images_dir = os.path.join(split_dir, 'images')
        labels_dir = os.path.join(split_dir, 'labels')

        # Handle image saving (similar to before, but adjusted for new directory structure)
        if image_name in slice_map or ('_' in image_name and '.' not in image_name):
            # Handle slice images
            qimage = slice_map.get(image_name) or next((s[1] for s in slices if s[0] == image_name), None)
            if qimage is None:
                for stack_slices in image_slices.values():
                    qimage = next((s[1] for s in stack_slices if s[0] == image_name), None)
                    if qimage:
                        break
            if qimage is None:
                print(f"No image data found for slice {image_name}, skipping")
                continue
            file_name_img = f"{image_name}.png"
            save_path = os.path.join(images_dir, file_name_img)
            if not os.path.exists(save_path):
                qimage.save(save_path)
            img_width, img_height = qimage.width(), qimage.height()
        else:
            # Handle regular images. Exact key match first; substring
            # fallback (the original behaviour) is fragile when one image
            # name is a prefix of another.
            image_path = image_paths.get(image_name)
            if image_path is None:
                image_path = next(
                    (path for name, path in image_paths.items() if image_name in name),
                    None,
                )
            if not image_path or image_path.lower().endswith(('.tif', '.tiff', '.czi')):
                print(f"[YOLO v4] skipping {image_name!r}: no image path / TIFF source")
                continue
            file_name_img = image_name
            dst_path = os.path.join(images_dir, file_name_img)
            if not os.path.exists(dst_path):
                shutil.copy2(image_path, dst_path)
            img = QImage(image_path)
            img_width, img_height = img.width(), img.height()

        # Write YOLO format annotation
        label_file = os.path.splitext(file_name_img)[0] + '.txt'
        with open(os.path.join(labels_dir, label_file), 'w', encoding='utf-8') as f:
            for class_name, class_annotations in annotations.items():
                if class_name not in class_to_index:
                    print(f"[YOLO v4] warning: class {class_name!r} not in class_mapping, skipped")
                    continue
                class_index = class_to_index[class_name]
                for ann in class_annotations:
                    if 'segmentation' in ann and ann['segmentation']:
                        polygon = ann['segmentation']
                        normalized_polygon = [coord / img_width if i % 2 == 0 else coord / img_height for i, coord in enumerate(polygon)]
                        f.write(f"{class_index} " + " ".join(map(lambda x: f"{x:.6f}", normalized_polygon)) + "\n")
                    elif 'bbox' in ann and ann['bbox']:
                        x, y, w, h = ann['bbox']
                        x_center = (x + w/2) / img_width
                        y_center = (y + h/2) / img_height
                        w = w / img_width
                        h = h / img_height
                        f.write(f"{class_index} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

    # Create YAML file. Point val at the populated valid/ dir only when images
    # were actually routed there; otherwise fall back to the train images so
    # the path stays non-empty (single-image projects, or val_split == 0).
    names = list(class_mapping.keys())
    val_images_dir = valid_dir if val_names else train_dir
    yaml_data = {
        'train': os.path.abspath(os.path.join(train_dir, 'images')),
        'val': os.path.abspath(os.path.join(val_images_dir, 'images')),
        'test': '../test/images',  # Placeholder
        'nc': len(names),
        'names': names
    }

    # Save YAML file in the output directory
    yaml_path = os.path.join(output_dir, 'data.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_data, f, default_flow_style=False)

    return train_dir, yaml_path



def _pose_export_check(all_annotations, class_mapping, keypoint_schemas):
    """None for an ordinary (non-pose) export. Otherwise ``(K, flip_idx)`` —
    the single schema every exported class must share. Raises ``ValueError``
    if the exported data mixes more than one distinct K, or mixes pose and
    non-pose classes: a YOLO-pose dataset's data.yaml has ONE global
    kpt_shape/flip_idx, not one per class.

    Detection is based on the actual ``keypoints`` key on each annotation
    being exported, not solely on ``keypoint_schemas`` — so a caller that
    omits ``keypoint_schemas`` (e.g. the PR-3 training dataset-prep call
    site) still gets a correct K (flip_idx degrades to identity) instead of
    writing pose-shaped label lines with no matching data.yaml key.
    (issue #35 PR-2)
    """
    per_class_k = {}
    non_pose_classes = set()
    any_keypoints = False
    for image_annotations in all_annotations.values():
        for class_name, anns in image_annotations.items():
            if class_name not in class_mapping or not anns:
                continue
            has_plain = False
            for ann in anns:
                if ann.get('keypoints'):
                    any_keypoints = True
                    per_class_k.setdefault(class_name, set()).add(len(ann['keypoints']) // 3)
                else:
                    has_plain = True
            if has_plain:
                non_pose_classes.add(class_name)

    if not any_keypoints:
        return None

    inconsistent = {c: ks for c, ks in per_class_k.items() if len(ks) > 1}
    distinct_k = {next(iter(ks)) for ks in per_class_k.values()}
    if inconsistent or len(distinct_k) > 1 or non_pose_classes:
        lines = [
            f"  - {name}: K={next(iter(ks))}" + (" (also has non-keypoint instances)" if name in non_pose_classes else "")
            for name, ks in per_class_k.items()
        ]
        msg = (
            "YOLO-pose export requires every exported class to share exactly one "
            "keypoint schema (K) — a dataset's data.yaml has a single global "
            "kpt_shape, not one per class.\n\nPose classes found:\n" + "\n".join(lines)
        )
        purely_non_pose = non_pose_classes - set(per_class_k)
        if purely_non_pose:
            msg += "\n\nNon-pose classes with annotations (no keypoints): " + ", ".join(sorted(purely_non_pose))
        msg += "\n\nExport only the pose class(es) that share one schema, or split the export."
        raise ValueError(msg)

    k = next(iter(distinct_k))
    flip_idx = None
    for class_name in per_class_k:
        schema = (keypoint_schemas or {}).get(class_name)
        if schema and schema_k(schema) == k:
            flip_idx = list(schema["flip_idx"])
            break
    return k, (flip_idx or list(range(k)))


def export_yolo_v5plus(all_annotations, class_mapping, image_paths, slices, image_slices, output_dir, val_split=0, keypoint_schemas=None):
    """
    Export annotations in YOLO v5+ format.
    Directory structure:
    output_dir/
        ├── data.yaml
        ├── images/
        │   ├── train/
        │   └── val/
        └── labels/
            ├── train/
            └── val/
    """
    # Validate before writing anything to disk — a rejected export must
    # leave zero output (issue #35 PR-2).
    pose_info = _pose_export_check(all_annotations, class_mapping, keypoint_schemas)

    # Create output directories with new structure
    images_train_dir = os.path.join(output_dir, 'images', 'train')
    images_val_dir = os.path.join(output_dir, 'images', 'val')
    labels_train_dir = os.path.join(output_dir, 'labels', 'train')
    labels_val_dir = os.path.join(output_dir, 'labels', 'val')

    for dir_path in [images_train_dir, images_val_dir, labels_train_dir, labels_val_dir]:
        os.makedirs(dir_path, exist_ok=True)

    # Create a mapping of class names to YOLO indices
    class_to_index = {name: i for i, name in enumerate(class_mapping.keys())}

    # Create a mapping of slice names to their QImage objects
    slice_map = {slice_name: qimage for slice_name, qimage in slices}

    # Deterministically split the annotated images into train/val.
    annotated = [name for name, ann in all_annotations.items() if ann]
    _, val_names = assign_train_val(annotated, val_split)

    print(f"[YOLO v5+] export: {len(all_annotations)} image entries, "
          f"{len(image_paths)} known image paths, "
          f"{len(class_to_index)} class(es) -> {list(class_to_index.keys())}; "
          f"val_split={val_split}% -> {len(val_names)} val / "
          f"{len(annotated) - len(val_names)} train")

    label_files_written = 0
    for image_name, annotations in all_annotations.items():
        print(f"[YOLO v5+]   image={image_name!r} annotation-classes={list(annotations.keys()) if annotations else '(none)'}")
        # Skip if there are no annotations for this image/slice
        if not annotations:
            print("[YOLO v5+]     skipping: no annotations")
            continue

        # Route this image into the train or val directory.
        if image_name in val_names:
            images_dir, labels_dir = images_val_dir, labels_val_dir
        else:
            images_dir, labels_dir = images_train_dir, labels_train_dir

        # Handle image saving (similar logic to the v4 version)
        if image_name in slice_map or ('_' in image_name and '.' not in image_name):
            # Handle slice images
            qimage = slice_map.get(image_name)
            if qimage is None:
                for stack_slices in image_slices.values():
                    qimage = next((s[1] for s in stack_slices if s[0] == image_name), None)
                    if qimage:
                        break
            if qimage is None:
                print(f"[YOLO v5+]     skipping: no image data for slice {image_name}")
                continue
            file_name_img = f"{image_name}.png"
            save_path = os.path.join(images_dir, file_name_img)
            if not os.path.exists(save_path):
                qimage.save(save_path)
            img_width, img_height = qimage.width(), qimage.height()
        else:
            # Handle regular images. Use exact-key match first; only fall
            # back to substring match if no exact key is found (substring
            # match was the original behaviour but it produces wrong hits
            # when one image name is a prefix of another).
            image_path = image_paths.get(image_name)
            if image_path is None:
                image_path = next(
                    (path for name, path in image_paths.items() if image_name in name),
                    None,
                )
            if not image_path:
                print(f"[YOLO v5+]     skipping: no image_paths entry for {image_name!r}")
                continue
            if image_path.lower().endswith(('.tif', '.tiff', '.czi')):
                print(f"[YOLO v5+]     skipping: TIFF/CZI source {image_name!r} (use slice export)")
                continue
            file_name_img = image_name
            dst_path = os.path.join(images_dir, file_name_img)
            if not os.path.exists(dst_path):
                shutil.copy2(image_path, dst_path)
                print(f"[YOLO v5+]     copied image -> {dst_path}")
            img = QImage(image_path)
            img_width, img_height = img.width(), img.height()

        # Write YOLO format annotation
        label_file = os.path.splitext(file_name_img)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_file)
        ann_lines = 0
        with open(label_path, 'w', encoding='utf-8') as f:
            for class_name, class_annotations in annotations.items():
                if class_name not in class_to_index:
                    print(f"[YOLO v5+]     warning: class {class_name!r} not in class_mapping, skipped")
                    continue
                class_index = class_to_index[class_name]
                for ann in class_annotations:
                    if 'keypoints' in ann and ann['keypoints']:
                        # Checked first — a pose instance also carries a bbox
                        # (issue #35 PR-2), matching the COCO ordering.
                        flat = ann['keypoints']
                        x, y, w, h = ann.get('bbox') or [0, 0, 0, 0]
                        x_center = (x + w/2) / img_width
                        y_center = (y + h/2) / img_height
                        w_n = w / img_width
                        h_n = h / img_height
                        tokens = [f"{x_center:.6f}", f"{y_center:.6f}", f"{w_n:.6f}", f"{h_n:.6f}"]
                        for i in range(0, len(flat), 3):
                            tokens.append(f"{flat[i] / img_width:.6f}")
                            tokens.append(f"{flat[i + 1] / img_height:.6f}")
                            tokens.append(str(int(flat[i + 2])))
                        f.write(f"{class_index} " + " ".join(tokens) + "\n")
                        ann_lines += 1
                    elif 'segmentation' in ann and ann['segmentation']:
                        polygon = ann['segmentation']
                        normalized_polygon = [coord / img_width if i % 2 == 0 else coord / img_height
                                           for i, coord in enumerate(polygon)]
                        f.write(f"{class_index} " + " ".join(map(lambda x: f"{x:.6f}", normalized_polygon)) + "\n")
                        ann_lines += 1
                    elif 'bbox' in ann and ann['bbox']:
                        x, y, w, h = ann['bbox']
                        x_center = (x + w/2) / img_width
                        y_center = (y + h/2) / img_height
                        w = w / img_width
                        h = h / img_height
                        f.write(f"{class_index} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
                        ann_lines += 1
        print(f"[YOLO v5+]     wrote {ann_lines} annotation line(s) -> {label_path}")
        label_files_written += 1

    print(f"[YOLO v5+] export complete: {label_files_written} label file(s) written")

    # Create YAML file. Point val at the val split only when images were
    # actually routed there; otherwise fall back to train so `yolo train`
    # never reads an empty val dir (single-image projects, or val_split == 0).
    names = list(class_mapping.keys())
    val_rel = os.path.join('images', 'val' if val_names else 'train')
    yaml_data = {
        'path': os.path.abspath(output_dir),  # Root directory
        'train': os.path.join('images', 'train'),  # Relative to path
        'val': val_rel,  # Relative to path
        'nc': len(names),
        'names': names
    }
    if pose_info is not None:
        k, flip_idx = pose_info
        yaml_data['kpt_shape'] = [k, 3]
        yaml_data['flip_idx'] = flip_idx

    # Save YAML file in the output directory
    yaml_path = os.path.join(output_dir, 'data.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_data, f, default_flow_style=False)

    return output_dir, yaml_path



def export_sam_dataset(all_annotations, class_mapping, image_paths, slices, image_slices, output_dir):
    """Export a SAM fine-tuning dataset: ``images/`` + ``manifest.json``.

    The manifest is the authoritative training source — per-instance ``bbox``/
    ``segmentation`` specs are rasterised to masks deterministically at train
    time (see ``training.sam_dataset``), so no separate mask PNGs are written.
    Image resolution mirrors ``export_yolo_v5plus`` (slices via ``slices`` /
    ``image_slices``; regular images via ``image_paths``; TIFF/CZI skipped).

    Returns ``(output_dir, manifest_path)``.
    """
    images_dir = os.path.join(output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    slice_map = {slice_name: qimage for slice_name, qimage in slices}

    manifest = {"classes": list(class_mapping.keys()), "images": []}
    for image_name, annotations in all_annotations.items():
        if not annotations:
            continue

        # Resolve + save the image (same branching as export_yolo_v5plus).
        if image_name in slice_map or ('_' in image_name and '.' not in image_name):
            qimage = slice_map.get(image_name)
            if qimage is None:
                for stack_slices in image_slices.values():
                    qimage = next((s[1] for s in stack_slices if s[0] == image_name), None)
                    if qimage is not None:
                        break
            if qimage is None:
                continue
            # basename guards against a separator in an image/slice key
            # escaping images/ during write.
            file_name_img = f"{os.path.basename(image_name)}.png"
            save_path = os.path.join(images_dir, file_name_img)
            if not os.path.exists(save_path):
                qimage.save(save_path)
        else:
            image_path = image_paths.get(image_name)
            if image_path is None:
                image_path = next(
                    (path for name, path in image_paths.items() if image_name in name),
                    None,
                )
            if not image_path:
                continue
            if image_path.lower().endswith(('.tif', '.tiff', '.czi')):
                continue
            file_name_img = os.path.basename(image_name)
            dst_path = os.path.join(images_dir, file_name_img)
            if not os.path.exists(dst_path):
                shutil.copy2(image_path, dst_path)

        instances = []
        for class_name, class_annotations in annotations.items():
            for ann in class_annotations:
                if ann.get('segmentation'):
                    instances.append({"class": class_name, "segmentation": ann['segmentation']})
                elif ann.get('bbox'):
                    instances.append({"class": class_name, "bbox": ann['bbox']})
        if instances:
            manifest["images"].append({
                "image": os.path.join('images', file_name_img),
                "instances": instances,
            })

    manifest_path = os.path.join(output_dir, 'manifest.json')
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
    print(f"[SAM dataset] wrote {len(manifest['images'])} image entries -> {manifest_path}")
    return output_dir, manifest_path


def export_labeled_images(all_annotations, class_mapping, image_paths, slices, image_slices, output_dir):
    # Create output directories
    images_dir = os.path.join(output_dir, 'images')
    labeled_images_dir = os.path.join(output_dir, 'labeled_images')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labeled_images_dir, exist_ok=True)

    # Create a dictionary to store class information for the summary
    class_summary = {class_name: [] for class_name in class_mapping.keys()}

    # Create directories for each class inside labeled_images_dir
    for class_name in class_mapping.keys():
        os.makedirs(os.path.join(labeled_images_dir, class_name), exist_ok=True)

    # Create a mapping of slice names to their QImage objects
    slice_map = {slice_name: qimage for slice_name, qimage in slices}

    for image_name, annotations in all_annotations.items():
        # Skip if there are no annotations for this image/slice
        if not annotations:
            continue

        # Check if it's a slice (either in slice_map or has underscores and no file extension)
        is_slice = image_name in slice_map or ('_' in image_name and '.' not in image_name)
        
        if is_slice:
            qimage = slice_map.get(image_name)
            if qimage is None:
                # If the slice is not in slice_map, it might be a CZI slice or a TIFF slice
                matching_slices = [s for s in slices if s[0] == image_name]
                if matching_slices:
                    qimage = matching_slices[0][1]
                else:
                    # Check in image_slices
                    for stack_slices in image_slices.values():
                        matching_slices = [s for s in stack_slices if s[0] == image_name]
                        if matching_slices:
                            qimage = matching_slices[0][1]
                            break
                if qimage is None:
                    print(f"No image data found for slice {image_name}, skipping")
                    continue
            file_name_img = f"{image_name}.png"
            # Save the QImage as a file
            save_path = os.path.join(images_dir, file_name_img)
            if not os.path.exists(save_path):
                qimage.save(save_path)
            else:
                print(f"Image {file_name_img} already exists in the target directory. Skipping copy.")
            img_width, img_height = qimage.width(), qimage.height()
        else:
            # Check if the image_name exists in image_paths
            image_path = next((path for name, path in image_paths.items() if image_name in name), None)
            if not image_path:
                print(f"No image path found for {image_name}, skipping")
                continue
            if image_path.lower().endswith(('.tif', '.tiff', '.czi')):
                print(f"Skipping main tiff/czi file: {image_name}")
                continue
            file_name_img = image_name
            # Copy the image file
            dst_path = os.path.join(images_dir, file_name_img)
            if not os.path.exists(dst_path):
                shutil.copy2(image_path, dst_path)
            else:
                print(f"Image {file_name_img} already exists in the target directory. Skipping copy.")


            img = Image.open(image_path)
            img_width, img_height = img.size

        # Create a dictionary to store masks for each class
        class_masks = {class_name: np.zeros((img_height, img_width), dtype=np.uint16) for class_name in class_mapping.keys()}

        for class_name, class_annotations in annotations.items():
            mask = class_masks[class_name]
            for ann in class_annotations:
                object_number = np.max(mask) + 1  # Increment object number for this class
                
                if 'segmentation' in ann:
                    polygon = np.array(ann['segmentation']).reshape(-1, 2)
                    rr, cc = skimage.draw.polygon(polygon[:, 1], polygon[:, 0], (img_height, img_width))
                    mask[rr, cc] = object_number
                elif 'bbox' in ann:
                    x, y, w, h = map(int, ann['bbox'])
                    mask[y:y+h, x:x+w] = object_number

            class_summary[class_name].append(file_name_img)

        # Save masks for each class
        for class_name, mask in class_masks.items():
            if np.any(mask):  # Only save if the mask is not empty
                mask_filename = f"{os.path.splitext(file_name_img)[0]}_{class_name}_mask.png"
                mask_path = os.path.join(labeled_images_dir, class_name, mask_filename)
                Image.fromarray(mask.astype(np.uint16)).save(mask_path)

    # Create summary text file
    summary_path = os.path.join(labeled_images_dir, 'class_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("Classes (folder names):\n")
        for class_name, files in class_summary.items():
            if files:  # Only include classes that have annotations
                f.write(f"- {class_name}\n")
                f.write(f"  Images: {', '.join(sorted(set(files)))}\n\n")

    return output_dir



def export_semantic_labels(all_annotations, class_mapping, image_paths, slices, image_slices, output_dir):
    # Create output directories
    images_dir = os.path.join(output_dir, 'images')
    segmented_images_dir = os.path.join(output_dir, 'segmented_images')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(segmented_images_dir, exist_ok=True)

    # Create a mapping of class names to unique pixel values
    class_to_pixel = {name: i+1 for i, name in enumerate(sorted(class_mapping.keys()))}

    # Create a mapping of slice names to their QImage objects
    slice_map = {slice_name: qimage for slice_name, qimage in slices}

    for image_name, annotations in all_annotations.items():
        # Skip if there are no annotations for this image/slice
        if not annotations:
            continue

        # Check if it's a slice (either in slice_map or has underscores and no file extension)
        is_slice = image_name in slice_map or ('_' in image_name and '.' not in image_name)
        
        if is_slice:
            qimage = slice_map.get(image_name)
            if qimage is None:
                # If the slice is not in slice_map, it might be a CZI slice or a TIFF slice
                matching_slices = [s for s in slices if s[0] == image_name]
                if matching_slices:
                    qimage = matching_slices[0][1]
                else:
                    # Check in image_slices
                    for stack_slices in image_slices.values():
                        matching_slices = [s for s in stack_slices if s[0] == image_name]
                        if matching_slices:
                            qimage = matching_slices[0][1]
                            break
                if qimage is None:
                    print(f"No image data found for slice {image_name}, skipping")
                    continue
            file_name_img = f"{image_name}.png"
            # Save the QImage as a file
            save_path = os.path.join(images_dir, file_name_img)
            if not os.path.exists(save_path):
                qimage.save(save_path)
            else:
                print(f"Image {file_name_img} already exists in the target directory. Skipping copy.")
            img_width, img_height = qimage.width(), qimage.height()
        else:
            # Check if the image_name exists in image_paths
            image_path = next((path for name, path in image_paths.items() if image_name in name), None)
            if not image_path:
                print(f"No image path found for {image_name}, skipping")
                continue
            if image_path.lower().endswith(('.tif', '.tiff', '.czi')):
                print(f"Skipping main tiff/czi file: {image_name}")
                continue
            file_name_img = image_name
            # Copy the image file
            dst_path = os.path.join(images_dir, file_name_img)
            if not os.path.exists(dst_path):
                shutil.copy2(image_path, dst_path)
            else:
                print(f"Image {file_name_img} already exists in the target directory. Skipping copy.")

            img = Image.open(image_path)
            img_width, img_height = img.size

        # Create a single mask for all classes
        semantic_mask = np.zeros((img_height, img_width), dtype=np.uint8)

        for class_name, class_annotations in annotations.items():
            pixel_value = class_to_pixel[class_name]
            for ann in class_annotations:
                if 'segmentation' in ann:
                    polygon = np.array(ann['segmentation']).reshape(-1, 2)
                    rr, cc = skimage.draw.polygon(polygon[:, 1], polygon[:, 0], (img_height, img_width))
                    semantic_mask[rr, cc] = pixel_value
                elif 'bbox' in ann:
                    x, y, w, h = map(int, ann['bbox'])
                    semantic_mask[y:y+h, x:x+w] = pixel_value

        # Save semantic mask
        mask_filename = f"{os.path.splitext(file_name_img)[0]}_semantic_mask.png"
        mask_path = os.path.join(segmented_images_dir, mask_filename)
        Image.fromarray(semantic_mask).save(mask_path)

    # Create class mapping text file
    mapping_path = os.path.join(segmented_images_dir, 'class_pixel_mapping.txt')
    with open(mapping_path, 'w', encoding='utf-8') as f:
        f.write("Pixel Value : Class Name\n")
        for class_name, pixel_value in class_to_pixel.items():
            f.write(f"{pixel_value} : {class_name}\n")

    return output_dir



def export_pascal_voc_bbox(all_annotations, class_mapping, image_paths, slices, image_slices, output_dir):
    # Create output directories
    images_dir = os.path.join(output_dir, 'images')
    annotations_dir = os.path.join(output_dir, 'Annotations')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(annotations_dir, exist_ok=True)

    # Create a mapping of slice names to their QImage objects
    slice_map = {slice_name: qimage for slice_name, qimage in slices}

    for image_name, annotations in all_annotations.items():
        # Skip if there are no annotations for this image/slice
        if not annotations:
            continue

        # Check if it's a slice (either in slice_map or has underscores and no file extension)
        is_slice = image_name in slice_map or ('_' in image_name and '.' not in image_name)
        
        if is_slice:
            qimage = slice_map.get(image_name)
            if qimage is None:
                # If the slice is not in slice_map, it might be a CZI slice or a TIFF slice
                matching_slices = [s for s in slices if s[0] == image_name]
                if matching_slices:
                    qimage = matching_slices[0][1]
                else:
                    # Check in image_slices
                    for stack_slices in image_slices.values():
                        matching_slices = [s for s in stack_slices if s[0] == image_name]
                        if matching_slices:
                            qimage = matching_slices[0][1]
                            break
                if qimage is None:
                    print(f"No image data found for slice {image_name}, skipping")
                    continue
            file_name_img = f"{image_name}.png"
            # Save the QImage as a file
            save_path = os.path.join(images_dir, file_name_img)
            if not os.path.exists(save_path):
                qimage.save(save_path)
            else:
                print(f"Image {file_name_img} already exists in the target directory. Skipping copy.")
            img_width, img_height = qimage.width(), qimage.height()
        else:
            # Check if the image_name exists in image_paths
            image_path = next((path for name, path in image_paths.items() if image_name in name), None)
            if not image_path:
                print(f"No image path found for {image_name}, skipping")
                continue
            if image_path.lower().endswith(('.tif', '.tiff', '.czi')):
                print(f"Skipping main tiff/czi file: {image_name}")
                continue
            file_name_img = image_name
            # Copy the image file
            dst_path = os.path.join(images_dir, file_name_img)
            if not os.path.exists(dst_path):
                shutil.copy2(image_path, dst_path)
            else:
                print(f"Image {file_name_img} already exists in the target directory. Skipping copy.")

            img = QImage(image_path)
            img_width, img_height = img.width(), img.height()

        # Create the XML structure
        root = ET.Element('annotation')
        ET.SubElement(root, 'folder').text = 'images'
        ET.SubElement(root, 'filename').text = file_name_img
        ET.SubElement(root, 'path').text = os.path.join('images', file_name_img)

        size = ET.SubElement(root, 'size')
        ET.SubElement(size, 'width').text = str(img_width)
        ET.SubElement(size, 'height').text = str(img_height)
        ET.SubElement(size, 'depth').text = '3'  # Assuming RGB images

        ET.SubElement(root, 'segmented').text = '0'

        # Add object annotations
        for class_name, class_annotations in annotations.items():
            for ann in class_annotations:
                obj = ET.SubElement(root, 'object')
                ET.SubElement(obj, 'name').text = class_name
                ET.SubElement(obj, 'pose').text = 'Unspecified'
                ET.SubElement(obj, 'truncated').text = '0'
                ET.SubElement(obj, 'difficult').text = '0'

                if 'bbox' in ann:
                    x, y, w, h = ann['bbox']
                    bndbox = ET.SubElement(obj, 'bndbox')
                    ET.SubElement(bndbox, 'xmin').text = str(int(x))
                    ET.SubElement(bndbox, 'ymin').text = str(int(y))
                    ET.SubElement(bndbox, 'xmax').text = str(int(x + w))
                    ET.SubElement(bndbox, 'ymax').text = str(int(y + h))
    
        # Save the XML file
        xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="    ")
        xml_filename = os.path.splitext(file_name_img)[0] + '.xml'
        with open(os.path.join(annotations_dir, xml_filename), 'w', encoding='utf-8') as f:
            f.write(xml_str)
    
    return output_dir         



def export_pascal_voc_both(all_annotations, class_mapping, image_paths, slices, image_slices, output_dir):
    # Create output directories
    images_dir = os.path.join(output_dir, 'images')
    annotations_dir = os.path.join(output_dir, 'Annotations')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(annotations_dir, exist_ok=True)

    # Create a mapping of slice names to their QImage objects
    slice_map = {slice_name: qimage for slice_name, qimage in slices}

    for image_name, annotations in all_annotations.items():
        # Skip if there are no annotations for this image/slice
        if not annotations:
            continue

        # Check if it's a slice (either in slice_map or has underscores and no file extension)
        is_slice = image_name in slice_map or ('_' in image_name and '.' not in image_name)
        
        if is_slice:
            qimage = slice_map.get(image_name)
            if qimage is None:
                # If the slice is not in slice_map, it might be a CZI slice or a TIFF slice
                matching_slices = [s for s in slices if s[0] == image_name]
                if matching_slices:
                    qimage = matching_slices[0][1]
                else:
                    # Check in image_slices
                    for stack_slices in image_slices.values():
                        matching_slices = [s for s in stack_slices if s[0] == image_name]
                        if matching_slices:
                            qimage = matching_slices[0][1]
                            break
                if qimage is None:
                    print(f"No image data found for slice {image_name}, skipping")
                    continue
            file_name_img = f"{image_name}.png"
            # Save the QImage as a file
            save_path = os.path.join(images_dir, file_name_img)
            if not os.path.exists(save_path):
                qimage.save(save_path)
            else:
                print(f"Image {file_name_img} already exists in the target directory. Skipping copy.")
            img_width, img_height = qimage.width(), qimage.height()
        else:
            # Check if the image_name exists in image_paths
            image_path = next((path for name, path in image_paths.items() if image_name in name), None)
            if not image_path:
                print(f"No image path found for {image_name}, skipping")
                continue
            if image_path.lower().endswith(('.tif', '.tiff', '.czi')):
                print(f"Skipping main tiff/czi file: {image_name}")
                continue
            file_name_img = image_name
            # Copy the image file
            dst_path = os.path.join(images_dir, file_name_img)
            if not os.path.exists(dst_path):
                shutil.copy2(image_path, dst_path)
            else:
                print(f"Image {file_name_img} already exists in the target directory. Skipping copy.")

            img = QImage(image_path)
            img_width, img_height = img.width(), img.height()

        # Create the XML structure
        root = ET.Element('annotation')
        ET.SubElement(root, 'folder').text = 'images'
        ET.SubElement(root, 'filename').text = file_name_img
        ET.SubElement(root, 'path').text = os.path.join('images', file_name_img)

        size = ET.SubElement(root, 'size')
        ET.SubElement(size, 'width').text = str(img_width)
        ET.SubElement(size, 'height').text = str(img_height)
        ET.SubElement(size, 'depth').text = '3'  # Assuming RGB images

        ET.SubElement(root, 'segmented').text = '1'  # Set to 1 if segmentation is included

        # Add object annotations
        for class_name, class_annotations in annotations.items():
            for ann in class_annotations:
                obj = ET.SubElement(root, 'object')
                ET.SubElement(obj, 'name').text = class_name
                ET.SubElement(obj, 'pose').text = 'Unspecified'
                ET.SubElement(obj, 'truncated').text = '0'
                ET.SubElement(obj, 'difficult').text = '0'

                if 'bbox' in ann:
                    x, y, w, h = ann['bbox']
                    bndbox = ET.SubElement(obj, 'bndbox')
                    ET.SubElement(bndbox, 'xmin').text = str(int(x))
                    ET.SubElement(bndbox, 'ymin').text = str(int(y))
                    ET.SubElement(bndbox, 'xmax').text = str(int(x + w))
                    ET.SubElement(bndbox, 'ymax').text = str(int(y + h))

                if 'segmentation' in ann:
                    segmentation = ET.SubElement(obj, 'segmentation')
                    ET.SubElement(segmentation, 'area').text = str(ann.get('area', 0))
                    
                    # Convert polygon to a list of (x,y) tuples
                    polygon = ann['segmentation']
                    points = [(polygon[i], polygon[i+1]) for i in range(0, len(polygon), 2)]
                    
                    # Create the polygon element
                    polygon_elem = ET.SubElement(segmentation, 'polygon')
                    for i, (x, y) in enumerate(points):
                        point = ET.SubElement(polygon_elem, f'pt{i+1}')
                        ET.SubElement(point, 'x').text = str(int(x))
                        ET.SubElement(point, 'y').text = str(int(y))

        # Save the XML file
        xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="    ")
        xml_filename = os.path.splitext(file_name_img)[0] + '.xml'
        with open(os.path.join(annotations_dir, xml_filename), 'w', encoding='utf-8') as f:
            f.write(xml_str)

    return output_dir