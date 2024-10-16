import json
from PyQt5.QtGui import QImage
from .utils import calculate_area, calculate_bbox
import yaml
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
def convert_to_coco(all_annotations, class_mapping, image_paths, slices, image_slices):
    with tempfile.TemporaryDirectory() as temp_dir:
        json_file_path, images_dir = export_coco_json(all_annotations, class_mapping, image_paths, slices, image_slices, temp_dir)
        
        with open(json_file_path, 'r') as f:
            coco_data = json.load(f)
        
    return coco_data, images_dir



def export_coco_json(all_annotations, class_mapping, image_paths, slices, image_slices, output_dir, json_filename=None):
    coco_format = {
        "images": [],
        "categories": [{"id": id, "name": name} for name, id in class_mapping.items()],
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
    with open(json_file_path, 'w') as f:
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
    
    if "segmentation" in ann:
        coco_ann["segmentation"] = [ann["segmentation"]]
        coco_ann["bbox"] = calculate_bbox(ann["segmentation"])
    elif "bbox" in ann:
        coco_ann["bbox"] = ann["bbox"]
    
    return coco_ann



def export_yolo_v8(all_annotations, class_mapping, image_paths, slices, image_slices, output_dir):
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

    for image_name, annotations in all_annotations.items():
        # Skip if there are no annotations for this image/slice
        if not annotations:
            continue

        # For simplicity, we'll put all data in the train directory
        images_dir = os.path.join(train_dir, 'images')
        labels_dir = os.path.join(train_dir, 'labels')

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
            # Handle regular images
            image_path = next((path for name, path in image_paths.items() if image_name in name), None)
            if not image_path or image_path.lower().endswith(('.tif', '.tiff', '.czi')):
                print(f"Skipping file: {image_name}")
                continue
            file_name_img = image_name
            dst_path = os.path.join(images_dir, file_name_img)
            if not os.path.exists(dst_path):
                shutil.copy2(image_path, dst_path)
            img = QImage(image_path)
            img_width, img_height = img.width(), img.height()

        # Write YOLO format annotation
        label_file = os.path.splitext(file_name_img)[0] + '.txt'
        with open(os.path.join(labels_dir, label_file), 'w') as f:
            for class_name, class_annotations in annotations.items():
                class_index = class_to_index[class_name]
                for ann in class_annotations:
                    if 'segmentation' in ann:
                        polygon = ann['segmentation']
                        normalized_polygon = [coord / img_width if i % 2 == 0 else coord / img_height for i, coord in enumerate(polygon)]
                        f.write(f"{class_index} " + " ".join(map(lambda x: f"{x:.6f}", normalized_polygon)) + "\n")
                    elif 'bbox' in ann:
                        x, y, w, h = ann['bbox']
                        x_center = (x + w/2) / img_width
                        y_center = (y + h/2) / img_height
                        w = w / img_width
                        h = h / img_height
                        f.write(f"{class_index} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

    # Create YAML file
    names = list(class_mapping.keys())
    yaml_data = {
        'train': os.path.abspath(os.path.join(train_dir, 'images')),
        'val': os.path.abspath(os.path.join(train_dir, 'images')),  # Using train as val
        'test': '../test/images',  # Placeholder
        'nc': len(names),
        'names': names
    }

    # Save YAML file in the output directory
    yaml_path = os.path.join(output_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False)

    return train_dir, yaml_path



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
    with open(summary_path, 'w') as f:
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
    with open(mapping_path, 'w') as f:
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
        with open(os.path.join(annotations_dir, xml_filename), 'w') as f:
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
        with open(os.path.join(annotations_dir, xml_filename), 'w') as f:
            f.write(xml_str)

    return output_dir