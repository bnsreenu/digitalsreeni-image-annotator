
import json
import os
import yaml
from PIL import Image

from PyQt5.QtCore import QRectF
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QMessageBox, QFileDialog

import os
import json
from PyQt5.QtWidgets import QMessageBox

def import_coco_json(file_path, class_mapping):
    with open(file_path, 'r') as f:
        coco_data = json.load(f)

    imported_annotations = {}
    image_info = {}

    # Create reverse mapping of category IDs to names
    category_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}

    # Determine the image directory
    json_dir = os.path.dirname(file_path)
    images_dir = os.path.join(json_dir, 'images')
    
    if not os.path.exists(images_dir):
        print(f"Warning: 'images' subdirectory not found at {images_dir}")

    for image in coco_data['images']:
        file_name = image['file_name']
        image_path = os.path.join(images_dir, file_name)
        
        image_info[image['id']] = {
            'file_name': file_name,
            'width': image['width'],
            'height': image['height'],
            'path': image_path,
            'id': image['id']
        }
        
        if not os.path.isfile(image_path):
            print(f"Image not found: {file_name}")

    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in image_info:
            print(f"Warning: Annotation refers to non-existent image ID: {image_id}")
            continue

        file_name = image_info[image_id]['file_name']
        category_name = category_id_to_name[ann['category_id']]

        if file_name not in imported_annotations:
            imported_annotations[file_name] = {}

        if category_name not in imported_annotations[file_name]:
            imported_annotations[file_name][category_name] = []

        annotation = {
            'category_id': ann['category_id'],
            'category_name': category_name
        }

        if 'segmentation' in ann:
            annotation['segmentation'] = ann['segmentation'][0]
            annotation['type'] = 'polygon'
        elif 'bbox' in ann:
            annotation['bbox'] = ann['bbox']
            annotation['type'] = 'rectangle'

        imported_annotations[file_name][category_name].append(annotation)

    return imported_annotations, image_info


def import_yolo_v8(directory_path, class_mapping):
    images_dir = os.path.join(directory_path, 'images')
    labels_dir = os.path.join(directory_path, 'labels')
    
    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        raise ValueError("The selected directory must contain 'images' and 'labels' subdirectories.")
    
    yaml_file = next((f for f in os.listdir(directory_path) if f.endswith('.yaml')), None)
    if not yaml_file:
        raise ValueError("No YAML file found in the selected directory. Please add a YAML file and try again.")
    
    with open(os.path.join(directory_path, yaml_file), 'r') as f:
        yaml_data = yaml.safe_load(f)
    
    class_names = yaml_data.get('names', [])
    if not class_names:
        raise ValueError("No class names found in the YAML file.")
    
    imported_annotations = {}
    image_info = {}
    
    # First, process all label files
    for label_file in os.listdir(labels_dir):
        if label_file.lower().endswith('.txt'):
            img_file = os.path.splitext(label_file)[0] + '.jpg'  # Assume .jpg, but we'll check for other formats
            img_path = os.path.join(images_dir, img_file)
            
            # Check for other image formats if .jpg doesn't exist
            if not os.path.exists(img_path):
                for ext in ['.png', '.jpeg', '.tiff', '.bmp', '.gif']:
                    alt_img_file = os.path.splitext(label_file)[0] + ext
                    alt_img_path = os.path.join(images_dir, alt_img_file)
                    if os.path.exists(alt_img_path):
                        img_file = alt_img_file
                        img_path = alt_img_path
                        break
            
            # Get image dimensions if the image exists
            if os.path.exists(img_path):
                with Image.open(img_path) as img:
                    img_width, img_height = img.size
            else:
                print(f"Warning: Image not found for label {label_file}")
                img_width, img_height = 0, 0  # Use placeholder values
            
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
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
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
    
    return imported_annotations, image_info



def process_import_format(import_format, file_path, class_mapping):
    if import_format == "COCO JSON":
        return import_coco_json(file_path, class_mapping)
    elif import_format == "YOLO v8":
        return import_yolo_v8(file_path, class_mapping)
    else:
        raise ValueError(f"Unsupported import format: {import_format}")