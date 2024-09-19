# import_formats.py

import json
from PyQt5.QtCore import QRectF
from PyQt5.QtGui import QColor

def import_coco_json(file_path, class_mapping):
    with open(file_path, 'r') as f:
        coco_data = json.load(f)

    imported_annotations = {}
    image_info = {}

    # Create reverse mapping of category IDs to names
    category_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}

    for image in coco_data['images']:
        image_info[image['id']] = {
            'file_name': image['file_name'],
            'width': image['width'],
            'height': image['height']
        }

    for ann in coco_data['annotations']:
        image_id = ann['image_id']
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