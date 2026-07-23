
import copy
import json
import os
import yaml
from PIL import Image

from PyQt6.QtWidgets import QMessageBox

from ..core.keypoint_schema import sanitize_schema
from ..utils import keypoint_instance_bbox


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
                print(f"Warning: Skipped malformed keypoint schema for COCO category '{cat.get('name')}'")

        # Determine the image directory
        json_dir = os.path.dirname(file_path)
        images_dir = os.path.join(json_dir, 'images')
        
        if not os.path.exists(images_dir):
            print(f"Warning: 'images' subdirectory not found at {images_dir}")

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
            except KeyError as e:
                print(f"Warning: Missing required field in image data: {e}")
                continue

        # Process annotations
        masks_dropped_for_keypoints = 0
        for ann in coco_data['annotations']:
            try:
                image_id = int(ann['image_id'])
                if image_id not in image_info:
                    print(f"Warning: Annotation refers to non-existent image ID: {image_id}")
                    continue

                if ann['category_id'] not in category_id_to_name:
                    print(f"Warning: Invalid category ID: {ann['category_id']}")
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
                
            except (KeyError, ValueError, TypeError) as e:
                print(f"Warning: Error processing annotation: {e}")
                continue

        if masks_dropped_for_keypoints:
            print(
                f"Note: {masks_dropped_for_keypoints} annotation(s) carried both "
                f"'keypoints' and a 'segmentation' -- imported as keypoints-only, "
                f"source mask(s) dropped (issue #35 PR-2)."
            )

        return imported_annotations, image_info, keypoint_schemas

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON file: {e}")
    except Exception as e:
        raise ValueError(f"Error importing COCO JSON: {e}")


def import_yolo_v4(yaml_file_path, class_mapping):
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
                        print(f"Warning: Class ID {class_id} in {label_file} is out of range. Skipping this annotation.")
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
        
        reply = QMessageBox.question(None, "Import Issues", message, 
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        
        if reply == QMessageBox.StandardButton.No:
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
            print(f"Warning: {split} directory not found, skipping")
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
                    print(f"Warning: No image found for label {label_file}")
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
                            print(f"Warning: Class ID {class_id} in {label_file} is out of range")
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



def process_import_format(import_format, file_path, class_mapping):
    if import_format == "COCO JSON":
        return import_coco_json(file_path, class_mapping)
    elif import_format == "YOLO (v4 and earlier)":
        return import_yolo_v4(file_path, class_mapping)  # Still using same function, just updated format name
    elif import_format == "YOLO (v5+)":
        return import_yolo_v5plus(file_path, class_mapping)  # New format handling
    else:
        raise ValueError(f"Unsupported import format: {import_format}")


