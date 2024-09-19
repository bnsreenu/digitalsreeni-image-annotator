import json
from PyQt5.QtGui import QImage
from .utils import calculate_area, calculate_bbox
import yaml
import os
import tempfile
import xml.etree.ElementTree as ET
from xml.dom import minidom

import numpy as np
import skimage.draw
from PIL import Image


def export_coco_json(all_annotations, class_mapping, image_paths, slices, image_slices):
    coco_format = {
        "images": [],
        "categories": [{"id": id, "name": name} for name, id in class_mapping.items()],
        "annotations": []
    }
    
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
                # We need to find the corresponding QImage in slices or image_slices
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
        else:
            # Check if the image_name exists in image_paths
            image_path = next((path for name, path in image_paths.items() if image_name in name), None)
            if not image_path:
                print(f"No image path found for {image_name}, skipping")
                continue
            if image_path.lower().endswith(('.tif', '.tiff', '.czi')):
                print(f"Skipping main tiff/czi file: {image_name}")
                continue
            qimage = QImage(image_path)
            file_name_img = image_name

        image_info = {
            "file_name": file_name_img,
            "height": qimage.height(),
            "width": qimage.width(),
            "id": image_id
        }
        coco_format["images"].append(image_info)
        
        for class_name, class_annotations in annotations.items():
            for ann in class_annotations:
                coco_ann = create_coco_annotation(ann, image_id, annotation_id, class_name, class_mapping)
                coco_format["annotations"].append(coco_ann)
                annotation_id += 1
        
        image_id += 1

    return coco_format

def create_coco_annotation(ann, image_id, annotation_id, class_name, class_mapping):
    coco_ann = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": class_mapping[class_name],  # Use the correct category ID from class_mapping
        "area": calculate_area(ann),
        "iscrowd": 0
    }
    
    if "segmentation" in ann:
        coco_ann["segmentation"] = [ann["segmentation"]]
        coco_ann["bbox"] = calculate_bbox(ann["segmentation"])
    elif "bbox" in ann:
        coco_ann["bbox"] = ann["bbox"]
    
    return coco_ann


def export_yolo_v8(all_annotations, class_mapping, image_paths, slices, image_slices, output_dir):
    # First, export to COCO JSON format
    coco_data = export_coco_json(all_annotations, class_mapping, image_paths, slices, image_slices)
    
    # Create output directory for labels
    labels_dir = os.path.join(output_dir, 'labels')
    os.makedirs(labels_dir, exist_ok=True)

    # Convert COCO JSON to YOLO format
    cat_id_to_index = {cat['id']: i for i, cat in enumerate(coco_data['categories'])}

    for img in coco_data['images']:
        filename = img['file_name']
        img_id = img['id']
        img_w = img['width']
        img_h = img['height']
        
        img_anns = [ann for ann in coco_data['annotations'] if ann['image_id'] == img_id]

        if img_anns:
            with open(os.path.join(labels_dir, f"{os.path.splitext(filename)[0]}.txt"), "w") as f:
                for ann in img_anns:
                    category_index = cat_id_to_index[ann['category_id']]
                    if 'segmentation' in ann:
                        polygon = ann['segmentation'][0]
                        normalized_polygon = [format(coord / img_w if i % 2 == 0 else coord / img_h, '.6f') for i, coord in enumerate(polygon)]
                        f.write(f"{category_index} " + " ".join(normalized_polygon) + "\n")
                    elif 'bbox' in ann:
                        x, y, w, h = ann['bbox']
                        x_center = (x + w/2) / img_w
                        y_center = (y + h/2) / img_h
                        w = w / img_w
                        h = h / img_h
                        f.write(f"{category_index} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

    # Create YAML file
    names = [category['name'] for category in coco_data['categories']]
    yaml_data = {
        'path': os.path.dirname(output_dir),  # Set path to parent directory of labels
        'train': '../images',  # Relative path to images from labels directory
        'val': '../images',    # Relative path to images from labels directory
        'names': names,
        'nc': len(names)
    }

    # Save YAML file in the labels directory
    yaml_path = os.path.join(labels_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False)

    return labels_dir, yaml_path



def export_labeled_images(all_annotations, class_mapping, image_paths, slices, image_slices, output_dir):
    # First, export to COCO JSON format
    coco_data = export_coco_json(all_annotations, class_mapping, image_paths, slices, image_slices)
    
    # Create output directory for labeled images
    labeled_images_dir = os.path.join(output_dir, 'labeled_images')
    os.makedirs(labeled_images_dir, exist_ok=True)

    # Create a dictionary to store class information for the summary
    class_summary = {class_name: [] for class_name in class_mapping.keys()}

    # Create directories for each class
    for class_name in class_mapping.keys():
        os.makedirs(os.path.join(labeled_images_dir, class_name), exist_ok=True)

    for img in coco_data['images']:
        filename = img['file_name']
        img_id = img['id']
        img_w = img['width']
        img_h = img['height']
        
        # Create a dictionary to store masks for each class
        class_masks = {class_name: np.zeros((img_h, img_w), dtype=np.uint16) for class_name in class_mapping.keys()}
        
        img_anns = [ann for ann in coco_data['annotations'] if ann['image_id'] == img_id]

        for ann in img_anns:
            class_name = next(name for name, id in class_mapping.items() if id == ann['category_id'])
            mask = class_masks[class_name]
            
            object_number = np.max(mask) + 1  # Increment object number for this class
            
            if 'segmentation' in ann:
                for seg in ann['segmentation']:
                    # Convert polygons to a binary mask and add it to the class mask
                    polygon = np.array(seg).reshape(-1, 2)
                    rr, cc = skimage.draw.polygon(polygon[:, 1], polygon[:, 0], (img_h, img_w))
                    mask[rr, cc] = object_number
            elif 'bbox' in ann:
                x, y, w, h = map(int, ann['bbox'])
                mask[y:y+h, x:x+w] = object_number

            class_summary[class_name].append(filename)

        # Save masks for each class
        for class_name, mask in class_masks.items():
            mask_filename = f"{os.path.splitext(filename)[0]}_mask.png"
            mask_path = os.path.join(labeled_images_dir, class_name, mask_filename)
            Image.fromarray(mask.astype(np.uint16)).save(mask_path)

    # Create summary text file
    
    # Create a set to store unique class names
    unique_classes = set(class_mapping.keys())
    
    # Create summary text file
    summary_path = os.path.join(labeled_images_dir, 'class_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("Classes (folder names):\n")
        for class_name in sorted(unique_classes):
            f.write(f"- {class_name}\n")
    
    

    return labeled_images_dir


def export_semantic_labels(all_annotations, class_mapping, image_paths, slices, image_slices, output_dir):
    # First, export to COCO JSON format
    coco_data = export_coco_json(all_annotations, class_mapping, image_paths, slices, image_slices)
    
    # Create output directory for semantic labels
    semantic_labels_dir = os.path.join(output_dir, 'semantic_labels')
    os.makedirs(semantic_labels_dir, exist_ok=True)

    # Create a mapping of class names to unique pixel values
    class_to_pixel = {name: i+1 for i, name in enumerate(sorted(class_mapping.keys()))}

    for img in coco_data['images']:
        filename = img['file_name']
        img_id = img['id']
        img_w = img['width']
        img_h = img['height']
        
        # Create a single mask for all classes
        semantic_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        
        img_anns = [ann for ann in coco_data['annotations'] if ann['image_id'] == img_id]

        for ann in img_anns:
            class_name = next(name for name, id in class_mapping.items() if id == ann['category_id'])
            pixel_value = class_to_pixel[class_name]
            
            if 'segmentation' in ann:
                for seg in ann['segmentation']:
                    # Convert polygons to a binary mask and add it to the semantic mask
                    polygon = np.array(seg).reshape(-1, 2)
                    rr, cc = skimage.draw.polygon(polygon[:, 1], polygon[:, 0], (img_h, img_w))
                    semantic_mask[rr, cc] = pixel_value
            elif 'bbox' in ann:
                x, y, w, h = map(int, ann['bbox'])
                semantic_mask[y:y+h, x:x+w] = pixel_value

        # Save semantic mask
        mask_filename = f"{os.path.splitext(filename)[0]}_semantic_mask.png"
        mask_path = os.path.join(semantic_labels_dir, mask_filename)
        Image.fromarray(semantic_mask).save(mask_path)

    # Create class mapping text file
    mapping_path = os.path.join(semantic_labels_dir, 'class_pixel_mapping.txt')
    with open(mapping_path, 'w') as f:
        f.write("Pixel Value : Class Name\n")
        for class_name, pixel_value in class_to_pixel.items():
            f.write(f"{pixel_value} : {class_name}\n")

    return semantic_labels_dir



def export_pascal_voc_bbox(all_annotations, class_mapping, image_paths, slices, image_slices, output_dir):
    # Create output directory for Pascal VOC annotations
    voc_dir = os.path.join(output_dir, 'pascal_voc')
    annotations_dir = os.path.join(voc_dir, 'Annotations')
    os.makedirs(annotations_dir, exist_ok=True)

    # First, export to COCO JSON format
    coco_data = export_coco_json(all_annotations, class_mapping, image_paths, slices, image_slices)

    for img in coco_data['images']:
        filename = img['file_name']
        img_id = img['id']
        img_w = img['width']
        img_h = img['height']

        # Create the XML structure
        root = ET.Element('annotation')
        ET.SubElement(root, 'folder').text = 'images'
        ET.SubElement(root, 'filename').text = filename
        ET.SubElement(root, 'path').text = os.path.join('images', filename)

        size = ET.SubElement(root, 'size')
        ET.SubElement(size, 'width').text = str(img_w)
        ET.SubElement(size, 'height').text = str(img_h)
        ET.SubElement(size, 'depth').text = '3'  # Assuming RGB images

        ET.SubElement(root, 'segmented').text = '0'

        # Add object annotations
        img_anns = [ann for ann in coco_data['annotations'] if ann['image_id'] == img_id]
        for ann in img_anns:
            obj = ET.SubElement(root, 'object')
            class_name = next(name for name, id in class_mapping.items() if id == ann['category_id'])
            ET.SubElement(obj, 'name').text = class_name
            ET.SubElement(obj, 'pose').text = 'Unspecified'
            ET.SubElement(obj, 'truncated').text = '0'
            ET.SubElement(obj, 'difficult').text = '0'

            bbox = ann.get('bbox')
            if bbox:
                x, y, w, h = bbox
                bndbox = ET.SubElement(obj, 'bndbox')
                ET.SubElement(bndbox, 'xmin').text = str(int(x))
                ET.SubElement(bndbox, 'ymin').text = str(int(y))
                ET.SubElement(bndbox, 'xmax').text = str(int(x + w))
                ET.SubElement(bndbox, 'ymax').text = str(int(y + h))

        # Save the XML file
        xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="    ")
        xml_filename = os.path.splitext(filename)[0] + '.xml'
        with open(os.path.join(annotations_dir, xml_filename), 'w') as f:
            f.write(xml_str)

    return voc_dir



def export_pascal_voc_both(all_annotations, class_mapping, image_paths, slices, image_slices, output_dir):
    voc_dir = os.path.join(output_dir, 'pascal_voc')
    annotations_dir = os.path.join(voc_dir, 'Annotations')
    os.makedirs(annotations_dir, exist_ok=True)

    coco_data = export_coco_json(all_annotations, class_mapping, image_paths, slices, image_slices)

    for img in coco_data['images']:
        filename = img['file_name']
        img_id = img['id']
        img_w = img['width']
        img_h = img['height']

        root = ET.Element('annotation')
        ET.SubElement(root, 'folder').text = 'images'
        ET.SubElement(root, 'filename').text = filename
        ET.SubElement(root, 'path').text = os.path.join('images', filename)

        size = ET.SubElement(root, 'size')
        ET.SubElement(size, 'width').text = str(img_w)
        ET.SubElement(size, 'height').text = str(img_h)
        ET.SubElement(size, 'depth').text = '3'  # Assuming RGB images

        ET.SubElement(root, 'segmented').text = '1'  # Set to 1 if segmentation is included

        img_anns = [ann for ann in coco_data['annotations'] if ann['image_id'] == img_id]
        for ann in img_anns:
            obj = ET.SubElement(root, 'object')
            class_name = next(name for name, id in class_mapping.items() if id == ann['category_id'])
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
                polygon = ann['segmentation'][0]  # Assume the first polygon if multiple exist
                segmentation = ET.SubElement(obj, 'segmentation')
                ET.SubElement(segmentation, 'area').text = str(ann.get('area', 0))
                
                # Convert polygon to a list of (x,y) tuples
                points = [(polygon[i], polygon[i+1]) for i in range(0, len(polygon), 2)]
                
                # Create the polygon element
                polygon_elem = ET.SubElement(segmentation, 'polygon')
                for i, (x, y) in enumerate(points):
                    point = ET.SubElement(polygon_elem, f'pt{i+1}')
                    ET.SubElement(point, 'x').text = str(int(x))
                    ET.SubElement(point, 'y').text = str(int(y))

        xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="    ")
        xml_filename = os.path.splitext(filename)[0] + '.xml'
        with open(os.path.join(annotations_dir, xml_filename), 'w') as f:
            f.write(xml_str)

    return voc_dir