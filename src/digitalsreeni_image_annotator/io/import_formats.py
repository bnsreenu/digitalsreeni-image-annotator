
import copy
import json
import os
import yaml
from PIL import Image

from ..core.keypoint_schema import sanitize_schema
from ..utils import (
    calculate_bbox,
    clamp_bbox,
    clamp_segmentation,
    keypoint_instance_bbox,
)

from ..core.logging_config import get_logger

logger = get_logger(__name__)


def import_coco_json(file_path, class_mapping):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)

        # Validate required fields
        required_fields = ['images', 'annotations', 'categories']
        for field in required_fields:
            if field not in coco_data:
                raise ValueError(f"Missing required field '{field}' in JSON file")

        imported_annotations = {}
        image_info = {}

        # Create reverse mapping of category IDs to names
        category_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}

        # Recover per-class keypoint schemas from categories carrying a COCO
        # "keypoints" (names) field. "skeleton" is 1-based per spec, converted
        # back to the app's 0-based indices; "flip_idx" is our own export
        # extension (no COCO precedent), already 0-based. (issue #35 PR-2)
        keypoint_schemas = {}
        for cat in coco_data['categories']:
            names = cat.get('keypoints')
            if not names:
                continue
            skeleton_0based = []
            for edge in (cat.get('skeleton') or []):
                if isinstance(edge, (list, tuple)) and len(edge) == 2:
                    try:
                        skeleton_0based.append([int(edge[0]) - 1, int(edge[1]) - 1])
                    except (TypeError, ValueError):
                        continue
            schema = sanitize_schema({
                "names": names,
                "skeleton": skeleton_0based,
                "flip_idx": cat.get('flip_idx'),
            })
            if schema is not None:
                keypoint_schemas[cat['name']] = schema
            else:
                logger.warning(f"Skipped malformed keypoint schema for COCO category '{cat.get('name')}'")

        # Determine the image directory
        json_dir = os.path.dirname(file_path)
        images_dir = os.path.join(json_dir, 'images')
        
        if not os.path.exists(images_dir):
            logger.warning(f"'images' subdirectory not found at {images_dir}")

        # Process images
        for image in coco_data['images']:
            try:
                file_name = image['file_name']
                image_path = os.path.join(images_dir, file_name)
                
                image_info[image['id']] = {
                    'file_name': file_name,
                    'width': int(image['width']),  # Ensure integers
                    'height': int(image['height']),
                    'path': image_path,
                    'id': int(image['id'])
                }
            except KeyError:
                logger.exception("Missing required field in image data")
                continue

        # Process annotations
        masks_dropped_for_keypoints = 0
        for ann in coco_data['annotations']:
            try:
                image_id = int(ann['image_id'])
                if image_id not in image_info:
                    logger.warning(f"Annotation refers to non-existent image ID: {image_id}")
                    continue

                if ann['category_id'] not in category_id_to_name:
                    logger.warning(f"Invalid category ID: {ann['category_id']}")
                    continue

                file_name = image_info[image_id]['file_name']
                category_name = category_id_to_name[ann['category_id']]

                if file_name not in imported_annotations:
                    imported_annotations[file_name] = {}

                if category_name not in imported_annotations[file_name]:
                    imported_annotations[file_name][category_name] = []

                annotation = {
                    'category_id': int(ann['category_id']),
                    'category_name': category_name
                }

                # Keypoint / pose instance (issue #35 PR-2) — checked before
                # segmentation/bbox handling, and skips the bbox->polygon
                # synthesis below entirely (a pose instance has no mask).
                raw_kps = ann.get('keypoints')
                if raw_kps:
                    flat = [float(v) for v in raw_kps]
                    if flat and len(flat) % 3 == 0:
                        if ann.get('segmentation'):
                            # The app's pose instance model has no mask (ADR-029)
                            # -- a source annotation carrying both is not an
                            # error, but the mask is a silent data reduction
                            # worth surfacing (e.g. real person_keypoints_*.json
                            # files often carry both).
                            masks_dropped_for_keypoints += 1
                        annotation['keypoints'] = flat
                        annotation['num_keypoints'] = int(ann.get(
                            'num_keypoints',
                            sum(1 for i in range(2, len(flat), 3) if flat[i] > 0),
                        ))
                        raw_bbox = ann.get('bbox')
                        if raw_bbox and len(raw_bbox) == 4:
                            annotation['bbox'] = [float(v) for v in raw_bbox]
                        else:
                            width = image_info[image_id]['width']
                            height = image_info[image_id]['height']
                            annotation['bbox'] = keypoint_instance_bbox(flat, width, height)
                        imported_annotations[file_name][category_name].append(annotation)
                        continue

                # Handle segmentation data
                has_valid_segmentation = False
                if 'segmentation' in ann and ann['segmentation']:  # Check if segmentation exists and is not empty
                    seg_data = ann['segmentation']
                    if isinstance(seg_data, list):
                        if seg_data and isinstance(seg_data[0], list):
                            # Take the first polygon if multiple are present
                            annotation['segmentation'] = [float(x) for x in seg_data[0]]
                            has_valid_segmentation = True
                        elif seg_data:  # Single polygon
                            annotation['segmentation'] = [float(x) for x in seg_data]
                            has_valid_segmentation = True

                # If no valid segmentation but bbox exists, create segmentation from bbox
                if not has_valid_segmentation and 'bbox' in ann:
                    x, y, w, h = [float(x) for x in ann['bbox']]
                    # Create rectangle polygon from bbox [x,y, x+w,y, x+w,y+h, x,y+h]
                    annotation['segmentation'] = [x, y, x + w, y, x + w, y + h, x, y + h]
                    annotation['type'] = 'polygon'
                    # Also store bbox data
                    annotation['bbox'] = [x, y, w, h]
                elif has_valid_segmentation:
                    annotation['type'] = 'polygon'
                elif 'bbox' in ann:  # Fallback to pure bbox if no segmentation could be created
                    annotation['bbox'] = [float(x) for x in ann['bbox']]
                    annotation['type'] = 'rectangle'

                imported_annotations[file_name][category_name].append(annotation)
                
            except (KeyError, ValueError, TypeError):
                logger.exception("Error processing annotation")
                continue

        if masks_dropped_for_keypoints:
            logger.info(
                f"{masks_dropped_for_keypoints} annotation(s) carried both "
                f"'keypoints' and a 'segmentation' -- imported as keypoints-only, "
                f"source mask(s) dropped (issue #35 PR-2)."
            )

        return imported_annotations, image_info, keypoint_schemas

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON file: {e}")
    except Exception as e:
        raise ValueError(f"Error importing COCO JSON: {e}")


def import_yolo_v4(yaml_file_path, class_mapping, confirm=None):
    """Import a legacy YOLO (v4 and earlier) dataset.

    ``confirm`` is called with a message when images and labels do not line up,
    and must return True to continue. It exists so this module can stay
    Qt-free (issue #76): the prompt used to be a ``QMessageBox`` raised from
    inside the importer, which is a UI concern in a core module (ADR-031) and
    was enough to make a headless import require a display.

    Default when no callback is given: **proceed**. A non-interactive caller
    that has already chosen to import a partially-matched dataset should get
    the data, not a refusal it cannot answer.
    """
    if not os.path.exists(yaml_file_path):
        raise ValueError("The selected YAML file does not exist.")
    
    directory_path = os.path.dirname(yaml_file_path)
    
    with open(yaml_file_path, 'r', encoding='utf-8') as f:
        yaml_data = yaml.safe_load(f)
    
    class_names = yaml_data.get('names', [])
    if not class_names:
        raise ValueError("No class names found in the YAML file.")
    
    train_dir = os.path.join(directory_path, 'train')
    if not os.path.exists(train_dir):
        raise ValueError("No 'train' subdirectory found in the YAML file's directory.")
    
    imported_annotations = {}
    image_info = {}
    
    images_dir = os.path.join(train_dir, 'images')
    labels_dir = os.path.join(train_dir, 'labels')
    
    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        raise ValueError("The 'train' directory must contain both 'images' and 'labels' subdirectories.")
    
    missing_images = []
    missing_labels = []
    
    for label_file in os.listdir(labels_dir):
        if label_file.lower().endswith('.txt'):
            base_name = os.path.splitext(label_file)[0]
            img_file = None
            img_path = None
            
            # Check for various image formats
            for ext in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.gif']:
                potential_img_file = base_name + ext
                potential_img_path = os.path.join(images_dir, potential_img_file)
                if os.path.exists(potential_img_path):
                    img_file = potential_img_file
                    img_path = potential_img_path
                    break
            
            if img_path is None:
                missing_images.append(base_name)
                continue
            
            with Image.open(img_path) as img:
                img_width, img_height = img.size
            
            image_id = len(image_info) + 1
            image_info[image_id] = {
                'file_name': img_file,
                'width': img_width,
                'height': img_height,
                'id': image_id,
                'path': img_path
            }
            
            imported_annotations[img_file] = {}
            
            label_path = os.path.join(labels_dir, label_file)
            with open(label_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    if class_id >= len(class_names):
                        logger.warning(f"Class ID {class_id} in {label_file} is out of range. Skipping this annotation.")
                        continue
                    class_name = class_names[class_id]
                    
                    if class_name not in imported_annotations[img_file]:
                        imported_annotations[img_file][class_name] = []
                    
                    if len(parts) == 5:  # bounding box format
                        x_center, y_center, width, height = map(float, parts[1:5])
                        x1 = (x_center - width/2) * img_width
                        y1 = (y_center - height/2) * img_height
                        x2 = (x_center + width/2) * img_width
                        y2 = (y_center + height/2) * img_height
                        
                        annotation = {
                            'category_id': class_id,
                            'category_name': class_name,
                            'type': 'rectangle',
                            'bbox': [x1, y1, x2-x1, y2-y1]
                        }
                    else:  # polygon format
                        polygon = [float(coord) * (img_width if i % 2 == 0 else img_height) for i, coord in enumerate(parts[1:])]
                        
                        annotation = {
                            'category_id': class_id,
                            'category_name': class_name,
                            'type': 'polygon',
                            'segmentation': polygon
                        }
                    
                    imported_annotations[img_file][class_name].append(annotation)
    
    # Check for images without labels
    for img_file in os.listdir(images_dir):
        base_name, ext = os.path.splitext(img_file)
        if ext.lower() in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.gif']:
            label_file = base_name + '.txt'
            if not os.path.exists(os.path.join(labels_dir, label_file)):
                missing_labels.append(img_file)
    
    if missing_images or missing_labels:
        message = "The following issues were found:\n\n"
        if missing_images:
            message += f"Labels without corresponding images: {', '.join(missing_images)}\n\n"
        if missing_labels:
            message += f"Images without corresponding labels: {', '.join(missing_labels)}\n\n"
        message += "Do you want to continue importing the remaining data?"
        
        if confirm is not None and not confirm(message):
            raise ValueError("Import cancelled due to missing files.")

    # Legacy format stays detection-only (issue #35 PR-2) — no keypoint
    # schemas to recover, but the 3-tuple contract must stay uniform across
    # every import_* entry point.
    return imported_annotations, image_info, {}


def import_yolo_v5plus(yaml_file_path, class_mapping):
    """
    Import annotations from YOLO v5+ format.
    Expected directory structure:
    root_dir/
        ├── data.yaml
        ├── images/
        │   ├── train/
        │   └── val/
        └── labels/
            ├── train/
            └── val/
    """
    if not os.path.exists(yaml_file_path):
        raise ValueError("The selected YAML file does not exist.")
    
    root_dir = os.path.dirname(yaml_file_path)
    
    with open(yaml_file_path, 'r', encoding='utf-8') as f:
        yaml_data = yaml.safe_load(f)
    
    class_names = yaml_data.get('names', [])
    if not class_names:
        raise ValueError("No class names found in the YAML file.")

    # YOLO-pose declares one dataset-global kpt_shape/flip_idx (issue #35
    # PR-2) — not one per class — so every class in `names` is treated as a
    # pose class with this K, even one with zero instances in this label set.
    kpt_shape = yaml_data.get('kpt_shape')
    pose_k = None
    if isinstance(kpt_shape, (list, tuple)) and len(kpt_shape) >= 1:
        try:
            pose_k = int(kpt_shape[0]) or None
        except (TypeError, ValueError):
            pose_k = None

    imported_annotations = {}
    image_info = {}

    # Process both train and val directories
    for split in ['train', 'val']:
        images_dir = os.path.join(root_dir, 'images', split)
        labels_dir = os.path.join(root_dir, 'labels', split)
        
        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            logger.warning(f"{split} directory not found, skipping")
            continue
        
        for label_file in os.listdir(labels_dir):
            if label_file.lower().endswith('.txt'):
                base_name = os.path.splitext(label_file)[0]
                img_file = None
                img_path = None
                
                # Check for various image formats
                for ext in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.gif']:
                    potential_img_file = base_name + ext
                    potential_img_path = os.path.join(images_dir, potential_img_file)
                    if os.path.exists(potential_img_path):
                        img_file = potential_img_file
                        img_path = potential_img_path
                        break
                
                if img_path is None:
                    logger.warning(f"No image found for label {label_file}")
                    continue
                
                with Image.open(img_path) as img:
                    img_width, img_height = img.size
                
                image_id = len(image_info) + 1
                image_info[image_id] = {
                    'file_name': img_file,
                    'width': img_width,
                    'height': img_height,
                    'id': image_id,
                    'path': img_path
                }
                
                imported_annotations[img_file] = {}
                
                label_path = os.path.join(labels_dir, label_file)
                with open(label_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        if class_id >= len(class_names):
                            logger.warning(f"Class ID {class_id} in {label_file} is out of range")
                            continue
                        class_name = class_names[class_id]
                        
                        if class_name not in imported_annotations[img_file]:
                            imported_annotations[img_file][class_name] = []

                        # Disambiguated purely by token count: kpt_shape in
                        # data.yaml declares this WHOLE dataset pose-only (issue
                        # #35 PR-2), so a line with 5+3*pose_k tokens is always
                        # a pose instance, never a same-length segmentation
                        # polygon -- YOLO-pose datasets don't mix in polygons.
                        if pose_k and len(parts) == 5 + 3 * pose_k:  # YOLO-pose format
                            x_center, y_center, width, height = map(float, parts[1:5])
                            x1 = (x_center - width/2) * img_width
                            y1 = (y_center - height/2) * img_height
                            w = width * img_width
                            h = height * img_height

                            flat = []
                            for i in range(5, len(parts), 3):
                                flat.extend([
                                    float(parts[i]) * img_width,
                                    float(parts[i + 1]) * img_height,
                                    float(parts[i + 2]),
                                ])

                            annotation = {
                                'category_id': class_id,
                                'category_name': class_name,
                                'keypoints': flat,
                                'num_keypoints': sum(1 for i in range(2, len(flat), 3) if flat[i] > 0),
                                'bbox': [x1, y1, w, h],
                            }
                        elif len(parts) == 5:  # bounding box format
                            x_center, y_center, width, height = map(float, parts[1:5])
                            x1 = (x_center - width/2) * img_width
                            y1 = (y_center - height/2) * img_height
                            w = width * img_width
                            h = height * img_height

                            annotation = {
                                'category_id': class_id,
                                'category_name': class_name,
                                'type': 'rectangle',
                                'bbox': [x1, y1, w, h]
                            }
                        else:  # polygon format
                            polygon = []
                            for i in range(1, len(parts), 2):
                                x = float(parts[i]) * img_width
                                y = float(parts[i+1]) * img_height
                                polygon.extend([x, y])
                            
                            annotation = {
                                'category_id': class_id,
                                'category_name': class_name,
                                'type': 'polygon',
                                'segmentation': polygon
                            }
                        
                        imported_annotations[img_file][class_name].append(annotation)

    # Applied uniformly to every declared class (see kpt_shape comment above),
    # not just classes observed with pose-shaped lines. Generic kp0..kp{K-1}
    # names — YOLO-pose carries no point names. copy.deepcopy per class so no
    # two class entries alias the same schema dict. (issue #35 PR-2)
    keypoint_schemas = {}
    if pose_k:
        schema = sanitize_schema({
            "names": [f"kp{i}" for i in range(pose_k)],
            "skeleton": [],
            "flip_idx": yaml_data.get('flip_idx'),
        })
        if schema is not None:
            for name in class_names:
                keypoint_schemas[name] = copy.deepcopy(schema)

    return imported_annotations, image_info, keypoint_schemas



def _voc_object_polygon(obj):
    """Flat polygon from a VOC ``<object>``'s inline segmentation, or ``None``.

    ``export_pascal_voc_both`` writes the outline **inline** in the XML:

    ```xml
    <segmentation>
      <area>1234</area>
      <polygon><pt1><x>10</x><y>20</y></pt1> …</polygon>
    </segmentation>
    ```

    Reading this is what makes the export/import round-trip real. The earlier
    version of this importer looked for ``SegmentationClass`` mask PNGs
    instead — a layout the app has never written — so importing the app's own
    VOC-with-segmentation export silently degraded every polygon to its
    bounding box.

    Mask-PNG reconstruction was deliberately **not** kept as a fallback: in a
    foreign VOC dataset the mask palette index is that producer's class id,
    which has no defined relationship to a class name in this project, so any
    colour-to-class mapping would be a guess that attributes regions to the
    wrong classes while looking like it worked.
    """
    segmentation = obj.find("segmentation")
    if segmentation is None:
        return None
    polygon = segmentation.find("polygon")
    if polygon is None:
        return None

    flat = []
    # Points are named pt1, pt2, ... in document order; iterate the children
    # rather than parsing the tag numbers, so a producer starting at pt0 or
    # padding to pt007 still reads correctly.
    for point in polygon:
        x = point.findtext("x")
        y = point.findtext("y")
        if x is None or y is None:
            continue
        try:
            flat.extend([float(x), float(y)])
        except ValueError:
            return None
    return flat if len(flat) >= 6 else None


def _voc_object_bbox(obj, xml_name):
    """``[x, y, w, h]`` from a VOC ``<bndbox>``, or ``None`` if absent/unreadable.

    VOC stores corners (``xmin, ymin, xmax, ymax``); the app stores origin plus
    size. Convert, never copy.
    """
    box = obj.find("bndbox")
    if box is None:
        return None
    corners = []
    for tag in ("xmin", "ymin", "xmax", "ymax"):
        text = box.findtext(tag)
        if text is None:
            # A missing corner must yield None, not a default of 0. Returning
            # [0, 0, 0, 0] is truthy, so it would suppress the polygon-derived
            # fallback and import a bogus zero box next to a perfectly good
            # outline.
            logger.warning("bndbox missing <%s> in %s", tag, xml_name)
            return None
        try:
            corners.append(float(text))
        except ValueError:
            logger.warning("unreadable bndbox <%s> in %s", tag, xml_name)
            return None
    xmin, ymin, xmax, ymax = corners
    return [xmin, ymin, max(0.0, xmax - xmin), max(0.0, ymax - ymin)]


def import_pascal_voc(directory_path, class_mapping):
    """Import a directory of Pascal VOC XML annotations (issue #75).

    Closes a plain asymmetry in ``io/``: the app has exported VOC since before
    this change but could never read its own output back, let alone the large
    amount of VOC-format data in the wild.

    ``directory_path`` may be the dataset root (containing ``Annotations/`` and
    ``images/``, the layout ``export_pascal_voc_bbox`` writes) or the
    ``Annotations`` directory itself — both are what a user actually picks.

    Where an object carries an inline ``<segmentation><polygon>`` (what
    ``export_pascal_voc_both`` writes), the outline is read too, so the
    export/import round-trip preserves masks and not just boxes.

    Returns the uniform ``(annotations, image_info, keypoint_schemas)`` triple
    every entry point in this module returns. The schema dict is always empty:
    VOC has no keypoint concept. Breaking that shape would break the caller.
    """
    import xml.etree.ElementTree as ET

    if not os.path.isdir(directory_path):
        raise ValueError("The selected Pascal VOC path is not a directory.")

    annotations_dir = directory_path
    if os.path.isdir(os.path.join(directory_path, "Annotations")):
        annotations_dir = os.path.join(directory_path, "Annotations")

    xml_files = sorted(
        name for name in os.listdir(annotations_dir) if name.lower().endswith(".xml")
    )
    if not xml_files:
        raise ValueError(
            f"No .xml annotation files found in {annotations_dir}."
        )

    imported_annotations = {}
    image_info = {}
    next_class_id = max(class_mapping.values(), default=0) + 1
    local_mapping = dict(class_mapping)

    for xml_name in xml_files:
        xml_path = os.path.join(annotations_dir, xml_name)
        try:
            tree = ET.parse(xml_path)
        except ET.ParseError as exc:
            # Abort rather than import half a dataset: a partially-imported
            # project is harder to recover from than a refused import.
            raise ValueError(f"Malformed Pascal VOC XML in {xml_name}: {exc}")
        root = tree.getroot()

        file_name = (root.findtext("filename") or "").strip()
        if not file_name:
            file_name = os.path.splitext(xml_name)[0] + ".png"

        size = root.find("size")
        img_width = int(float(size.findtext("width", "0"))) if size is not None else 0
        img_height = int(float(size.findtext("height", "0"))) if size is not None else 0

        image_info[file_name] = {
            "file_name": file_name,
            "width": img_width,
            "height": img_height,
            "id": len(image_info) + 1,
        }
        imported_annotations.setdefault(file_name, {})

        for obj in root.findall("object"):
            class_name = (obj.findtext("name") or "").strip()
            if not class_name:
                continue
            if class_name not in local_mapping:
                local_mapping[class_name] = next_class_id
                next_class_id += 1
            class_id = local_mapping[class_name]

            # The outline is read FIRST and a missing <bndbox> is not fatal.
            # This is not defensiveness for its own sake: shapes drawn in this
            # app carry no `bbox` key at all (see edit_gestures.sync_bbox_key),
            # so `export_pascal_voc_both` emits no <bndbox> for them — and an
            # importer that required one silently discarded every polygon the
            # app itself had exported, while reporting success.
            polygon = _voc_object_polygon(obj)
            if polygon and img_width > 0 and img_height > 0:
                polygon = clamp_segmentation(polygon, img_width, img_height)

            bbox = _voc_object_bbox(obj, xml_name)
            if bbox is None and polygon:
                # Derive it from the outline the object does have.
                bbox = calculate_bbox(polygon)
            if bbox is None:
                continue  # neither a box nor an outline: nothing to import
            # Producers disagree on whether VOC coordinates are 0- or 1-based,
            # so clamp into the image rather than trusting the file (ADR-024).
            if img_width > 0 and img_height > 0:
                bbox = clamp_bbox(bbox, img_width, img_height)

            annotation = {
                "category_id": class_id,
                "category_name": class_name,
                "type": "polygon" if polygon else "rectangle",
                "bbox": bbox,
            }
            if polygon:
                # The outline belongs to THIS object, so it needs no pairing
                # heuristic — the other reason inline beats masks here.
                annotation["segmentation"] = polygon

            # `difficult` and `truncated` have no home in the data model.
            # Ignored deliberately rather than invented into new fields.

            imported_annotations[file_name].setdefault(class_name, []).append(annotation)

    # VOC has no keypoint concept, but the triple's shape is the contract.
    return imported_annotations, image_info, {}


def process_import_format(import_format, file_path, class_mapping, confirm=None):
    if import_format == "COCO JSON":
        return import_coco_json(file_path, class_mapping)
    elif import_format == "YOLO (v4 and earlier)":
        # `confirm` lets the GUI supply a prompt without this module importing
        # Qt (issue #76).
        return import_yolo_v4(file_path, class_mapping, confirm)
    elif import_format == "YOLO (v5+)":
        return import_yolo_v5plus(file_path, class_mapping)  # New format handling
    elif import_format == "Pascal VOC":
        return import_pascal_voc(file_path, class_mapping)  # issue #75
    else:
        raise ValueError(f"Unsupported import format: {import_format}")


