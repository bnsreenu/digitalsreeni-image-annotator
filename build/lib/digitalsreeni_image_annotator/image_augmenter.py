import os
import random
import cv2
import numpy as np
import json

from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QFileDialog, QLabel, QMessageBox, QSpinBox, 
                             QCheckBox, QDoubleSpinBox, QProgressBar, QApplication)
from PyQt5.QtCore import Qt

class ImageAugmenterDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Image Augmenter")
        self.setGeometry(100, 100, 400, 600)
        self.setWindowFlags(self.windowFlags() | Qt.Window)
        self.setWindowModality(Qt.ApplicationModal)
        self.input_dir = ""
        self.output_dir = ""
        self.coco_file = ""
        self.coco_data = None
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # Input directory selection
        input_layout = QHBoxLayout()
        self.input_label = QLabel("Input Directory: Not selected")
        input_button = QPushButton("Select Input Directory")
        input_button.clicked.connect(self.select_input_directory)
        input_layout.addWidget(self.input_label)
        input_layout.addWidget(input_button)
        layout.addLayout(input_layout)

        # Output directory selection
        output_layout = QHBoxLayout()
        self.output_label = QLabel("Output Directory: Not selected")
        output_button = QPushButton("Select Output Directory")
        output_button.clicked.connect(self.select_output_directory)
        output_layout.addWidget(self.output_label)
        output_layout.addWidget(output_button)
        layout.addLayout(output_layout)

        # Number of augmentations per image
        aug_count_layout = QHBoxLayout()
        aug_count_layout.addWidget(QLabel("Augmentations per image:"))
        self.aug_count_spin = QSpinBox()
        self.aug_count_spin.setRange(1, 100)
        self.aug_count_spin.setValue(5)
        aug_count_layout.addWidget(self.aug_count_spin)
        layout.addLayout(aug_count_layout)
        
        
        # Add COCO JSON annotation augmentation checkbox and file selection
        self.coco_check = QCheckBox("Augment COCO JSON annotations")
        self.coco_check.stateChanged.connect(self.toggle_elastic_deformation)
        layout.addWidget(self.coco_check)
        
        coco_layout = QHBoxLayout()
        self.coco_label = QLabel("COCO JSON File: Not selected")
        coco_button = QPushButton("Select COCO JSON")
        coco_button.clicked.connect(self.select_coco_json)
        coco_layout.addWidget(self.coco_label)
        coco_layout.addWidget(coco_button)
        layout.addLayout(coco_layout)

        # Transformations
        layout.addWidget(QLabel("Transformations:"))

        self.rotate_check = QCheckBox("Rotate")
        self.rotate_spin = QSpinBox()
        self.rotate_spin.setRange(-180, 180)
        self.rotate_spin.setValue(30)
        rotate_layout = QHBoxLayout()
        rotate_layout.addWidget(self.rotate_check)
        rotate_layout.addWidget(QLabel("Max degrees:"))
        rotate_layout.addWidget(self.rotate_spin)
        layout.addLayout(rotate_layout)

        self.zoom_check = QCheckBox("Zoom")
        self.zoom_spin = QDoubleSpinBox()
        self.zoom_spin.setRange(0.1, 2.0)
        self.zoom_spin.setValue(0.2)
        self.zoom_spin.setSingleStep(0.1)
        zoom_layout = QHBoxLayout()
        zoom_layout.addWidget(self.zoom_check)
        zoom_layout.addWidget(QLabel("Scale factor:"))
        zoom_layout.addWidget(self.zoom_spin)
        layout.addLayout(zoom_layout)

        self.blur_check = QCheckBox("Gaussian Blur")
        layout.addWidget(self.blur_check)

        self.brightness_contrast_check = QCheckBox("Random Brightness and Contrast")
        layout.addWidget(self.brightness_contrast_check)

        self.sharpen_check = QCheckBox("Sharpen")
        layout.addWidget(self.sharpen_check)
        
        # Flip transformation
        flip_layout = QHBoxLayout()
        self.flip_check = QCheckBox("Flip")
        flip_layout.addWidget(self.flip_check)
        self.flip_horizontal_check = QCheckBox("Horizontal")
        self.flip_vertical_check = QCheckBox("Vertical")
        flip_layout.addWidget(self.flip_horizontal_check)
        flip_layout.addWidget(self.flip_vertical_check)
        self.flip_horizontal_check.stateChanged.connect(self.update_flip_check)
        self.flip_vertical_check.stateChanged.connect(self.update_flip_check)
        layout.addLayout(flip_layout)


        # Elastic Deformation
        self.elastic_check = QCheckBox("Elastic Deformation")
        layout.addWidget(self.elastic_check)
        elastic_layout = QHBoxLayout()
        elastic_layout.addWidget(self.elastic_check)
        elastic_layout.addWidget(QLabel("Alpha:"))
        self.elastic_alpha_spin = QSpinBox()
        self.elastic_alpha_spin.setRange(1, 1000)
        self.elastic_alpha_spin.setValue(500)
        elastic_layout.addWidget(self.elastic_alpha_spin)
        elastic_layout.addWidget(QLabel("Sigma:"))
        self.elastic_sigma_spin = QSpinBox()
        self.elastic_sigma_spin.setRange(1, 100)
        self.elastic_sigma_spin.setValue(20)
        elastic_layout.addWidget(self.elastic_sigma_spin)
        layout.addLayout(elastic_layout)

        # Grayscale Conversion
        self.grayscale_check = QCheckBox("Convert to Grayscale")
        layout.addWidget(self.grayscale_check)

        # Histogram Equalization
        self.hist_equalize_check = QCheckBox("Histogram Equalization")
        layout.addWidget(self.hist_equalize_check)


        # Augment button
        self.augment_button = QPushButton("Start Augmentation")
        self.augment_button.clicked.connect(self.start_augmentation)
        layout.addWidget(self.augment_button)

        # Progress bar
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        self.setLayout(layout)

    def select_input_directory(self):
        self.input_dir = QFileDialog.getExistingDirectory(self, "Select Input Directory")
        if self.input_dir:
            self.input_label.setText(f"Input Directory: {os.path.basename(self.input_dir)}")

    def select_output_directory(self):
        self.output_dir = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if self.output_dir:
            self.output_label.setText(f"Output Directory: {os.path.basename(self.output_dir)}")

    def update_flip_check(self, state):
        if self.flip_horizontal_check.isChecked() or self.flip_vertical_check.isChecked():
            self.flip_check.setChecked(True)
        else:
            self.flip_check.setChecked(False)
    
    def select_coco_json(self):
        self.coco_file, _ = QFileDialog.getOpenFileName(self, "Select COCO JSON File", "", "JSON Files (*.json)")
        if self.coco_file:
            self.coco_label.setText(f"COCO JSON File: {os.path.basename(self.coco_file)}")
            with open(self.coco_file, 'r') as f:
                self.coco_data = json.load(f)
            self.coco_check.setChecked(True)  # Automatically check the box when a file is loaded
                
    def toggle_elastic_deformation(self, state):
        if state == Qt.Checked:
            self.elastic_check.setChecked(False)
            self.elastic_check.setEnabled(False)
        else:
            self.elastic_check.setEnabled(True)
        

    
    def start_augmentation(self):
        if not self.input_dir or not self.output_dir:
            QMessageBox.warning(self, "Missing Directory", "Please select both input and output directories.")
            return
    
        if self.coco_check.isChecked() and not self.coco_file:
            QMessageBox.warning(self, "Missing COCO JSON", "Please select a COCO JSON file for annotation augmentation.")
            return
    
        # Create 'images' subdirectory in the output directory
        images_output_dir = os.path.join(self.output_dir, "images")
        os.makedirs(images_output_dir, exist_ok=True)
    
        image_files = [f for f in os.listdir(self.input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
        total_augmentations = len(image_files) * self.aug_count_spin.value()
    
        self.progress_bar.setMaximum(total_augmentations)
        self.progress_bar.setValue(0)
    
        augmented_coco_data = {
            "images": [],
            "annotations": [],
            "categories": self.coco_data["categories"] if self.coco_data else []
        }
    
        next_image_id = 1
        next_annotation_id = 1
    
        for i, image_file in enumerate(image_files):
            input_path = os.path.join(self.input_dir, image_file)
            image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
            
            if image is None:
                print(f"Error loading image: {input_path}")
                continue
    
            # Determine image type and bit depth
            is_color = len(image.shape) == 3 and image.shape[2] == 3
            bit_depth = image.dtype
    
            original_annotations = []
            if self.coco_check.isChecked():
                original_annotations = [ann for ann in self.coco_data["annotations"] 
                                        if any(img['file_name'] == image_file and img['id'] == ann['image_id'] 
                                               for img in self.coco_data["images"])]
    
            for j in range(self.aug_count_spin.value()):
                try:
                    augmented, transform_params = self.apply_random_augmentation(image, include_annotations=self.coco_check.isChecked())
                    
                    # Ensure the augmented image has the same properties as the input
                    if not is_color and len(augmented.shape) == 3:
                        augmented = cv2.cvtColor(augmented, cv2.COLOR_BGR2GRAY)
                    elif is_color and len(augmented.shape) == 2:
                        augmented = cv2.cvtColor(augmented, cv2.COLOR_GRAY2BGR)
                    
                    augmented = augmented.astype(bit_depth)
    
                    output_filename = f"{os.path.splitext(image_file)[0]}_aug_{j+1}{os.path.splitext(image_file)[1]}"
                    output_path = os.path.join(images_output_dir, output_filename)
                    cv2.imwrite(output_path, augmented)
    
                    if self.coco_check.isChecked():
                        augmented_coco_data["images"].append({
                            "id": next_image_id,
                            "file_name": output_filename,
                            "height": augmented.shape[0],
                            "width": augmented.shape[1]
                        })
    
                        for ann in original_annotations:
                            augmented_ann = self.augment_annotation(ann, transform_params, augmented.shape[:2])
                            augmented_ann["id"] = next_annotation_id
                            augmented_ann["image_id"] = next_image_id
                            augmented_coco_data["annotations"].append(augmented_ann)
                            next_annotation_id += 1
    
                        next_image_id += 1
    
                    self.progress_bar.setValue(i * self.aug_count_spin.value() + j + 1)
                    QApplication.processEvents()
    
                except Exception as e:
                    print(f"Error processing {image_file} (augmentation {j+1}): {str(e)}")
                    continue  # Skip this augmentation and continue with the next
    
        if self.coco_check.isChecked():
            output_coco_path = os.path.join(self.output_dir, "augmented_annotations.json")
            with open(output_coco_path, 'w') as f:
                json.dump(augmented_coco_data, f, indent=2)
    
        QMessageBox.information(self, "Augmentation Complete", "Image and annotation augmentation has been completed successfully.")

    


    def apply_random_augmentation(self, image, include_annotations=False):
        augmentations = []
        
        if self.rotate_check.isChecked():
            augmentations.append(self.rotate_image)
        if self.zoom_check.isChecked():
            augmentations.append(self.zoom_image)
        if self.blur_check.isChecked():
            augmentations.append(self.blur_image)
        if self.brightness_contrast_check.isChecked():
            augmentations.append(self.adjust_brightness_contrast)
        if self.sharpen_check.isChecked():
            augmentations.append(self.sharpen_image)
        if self.flip_check.isChecked():
            augmentations.append(self.flip_image)
        if self.elastic_check.isChecked() and not include_annotations:
            augmentations.append(self.elastic_transform)
        if self.grayscale_check.isChecked():
            augmentations.append(self.convert_to_grayscale)
        if self.hist_equalize_check.isChecked():
            augmentations.append(self.apply_histogram_equalization)
        
        if not augmentations:
            return image, {}
        
        aug_func = random.choice(augmentations)
        return aug_func(image)

    def rotate_image(self, image):
        angle = random.uniform(-self.rotate_spin.value(), self.rotate_spin.value())
        h, w = image.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, -angle, 1.0)  # Negative angle for clockwise rotation
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return rotated, {"type": "rotate", "angle": angle, "center": center, "matrix": M}
    
    def zoom_image(self, image):
        scale = random.uniform(1, 1 + self.zoom_spin.value())
        h, w = image.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, 0, scale)
        zoomed = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return zoomed, {"type": "zoom", "scale": scale, "center": center, "matrix": M}
    
    
    

    def blur_image(self, image):
        kernel_size = random.choice([3, 5, 7])
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        return blurred, {"type": "blur", "kernel_size": kernel_size}
    
    def adjust_brightness_contrast(self, image):
        alpha = random.uniform(0.5, 1.5)  # Contrast control
        beta = random.uniform(-30, 30)    # Brightness control
        adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        return adjusted, {"type": "brightness_contrast", "alpha": alpha, "beta": beta}
    
    def sharpen_image(self, image):
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(image, -1, kernel)
        return sharpened, {"type": "sharpen"}
    

    

    
    def flip_image(self, image):
        flip_options = []
        if self.flip_horizontal_check.isChecked():
            flip_options.append(1)  # Horizontal flip
        if self.flip_vertical_check.isChecked():
            flip_options.append(0)  # Vertical flip
        if self.flip_horizontal_check.isChecked() and self.flip_vertical_check.isChecked():
            flip_options.append(-1)  # Both horizontal and vertical

        if not flip_options:
            return image, {"type": "flip", "flip_code": None}

        flip_code = random.choice(flip_options)
        flipped = cv2.flip(image, flip_code)
        return flipped, {"type": "flip", "flip_code": flip_code}

    def elastic_transform(self, image):
        alpha = self.elastic_alpha_spin.value()
        sigma = self.elastic_sigma_spin.value()
        shape = image.shape[:2]
        random_state = np.random.RandomState(None)
        dx = random_state.rand(*shape) * 2 - 1
        dy = random_state.rand(*shape) * 2 - 1
        dx = cv2.GaussianBlur(dx, (0, 0), sigma) * alpha
        dy = cv2.GaussianBlur(dy, (0, 0), sigma) * alpha
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        distorted_x = x + dx
        distorted_y = y + dy
        
        transformed = cv2.remap(image, distorted_x.astype(np.float32), distorted_y.astype(np.float32), 
                                interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        
        return transformed, {"type": "elastic", "dx": dx, "dy": dy, "shape": shape}
    
    def convert_to_grayscale(self, image):
        if len(image.shape) == 2:
            return image, {"type": "grayscale"}  # Already grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_3channel = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  # Convert back to 3 channels
        return gray_3channel, {"type": "grayscale"}
    
    def apply_histogram_equalization(self, image):
        def equalize_8bit(img):
            return cv2.equalizeHist(img)
    
        def equalize_16bit(img):
            # Equalize 16-bit image
            hist, bins = np.histogram(img.flatten(), 65536, [0, 65536])
            cdf = hist.cumsum()
            cdf_normalized = cdf * 65535 / cdf[-1]
            equalized = np.interp(img.flatten(), bins[:-1], cdf_normalized).reshape(img.shape)
            return equalized.astype(np.uint16)
    
        if len(image.shape) == 2:  # Grayscale image
            if image.dtype == np.uint8:
                equalized = equalize_8bit(image)
            elif image.dtype == np.uint16:
                equalized = equalize_16bit(image)
            else:
                raise ValueError(f"Unsupported image dtype: {image.dtype}")
            return equalized, {"type": "histogram_equalization", "mode": "grayscale"}
        else:  # Color image
            # Convert to YUV color space
            yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            
            # Equalize the Y channel
            if image.dtype == np.uint8:
                yuv[:,:,0] = equalize_8bit(yuv[:,:,0])
            elif image.dtype == np.uint16:
                yuv[:,:,0] = equalize_16bit(yuv[:,:,0])
            else:
                raise ValueError(f"Unsupported image dtype: {image.dtype}")
            
            # Convert back to BGR color space
            equalized = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
            return equalized, {"type": "histogram_equalization", "mode": "color"}


    def augment_annotation(self, annotation, transform_params, image_shape):
        augmented_ann = annotation.copy()
        
        if transform_params["type"] == "rotate":
            angle = transform_params["angle"]
            center = transform_params["center"]
            matrix = transform_params["matrix"]
            augmented_ann["segmentation"] = [self.rotate_polygon(annotation["segmentation"][0], angle, center, matrix)]
        elif transform_params["type"] == "zoom":
            scale = transform_params["scale"]
            center = transform_params["center"]
            matrix = transform_params["matrix"]
            augmented_ann["segmentation"] = [self.scale_polygon(annotation["segmentation"][0], scale, center, matrix)]
        elif transform_params["type"] == "flip":
            flip_code = transform_params["flip_code"]
            if flip_code is not None:
                augmented_ann["segmentation"] = [self.flip_polygon(annotation["segmentation"][0], flip_code, image_shape)]
        elif transform_params["type"] == "elastic":
            dx = transform_params["dx"]
            dy = transform_params["dy"]
            shape = transform_params["shape"]
            augmented_ann["segmentation"] = [self.elastic_transform_polygon(annotation["segmentation"][0], dx, dy, shape)]
        
        # Recalculate bbox and area for all transformation types
        if "segmentation" in augmented_ann and augmented_ann["segmentation"]:
            augmented_ann["bbox"] = self.get_bbox_from_polygon(augmented_ann["segmentation"][0])
            augmented_ann["area"] = int(self.calculate_polygon_area(augmented_ann["segmentation"][0]))
        
        return augmented_ann


    
    def calculate_polygon_area(self, polygon):
        points = np.array(polygon).reshape(-1, 2)
        return 0.5 * np.abs(np.dot(points[:, 0], np.roll(points[:, 1], 1)) - np.dot(points[:, 1], np.roll(points[:, 0], 1)))


    def rotate_polygon(self, polygon, angle, center, matrix):
        points = np.array(polygon).reshape(-1, 2)
        ones = np.ones(shape=(len(points), 1))
        points_ones = np.hstack([points, ones])
        transformed_points = matrix.dot(points_ones.T).T
        return np.round(transformed_points).astype(int).flatten().tolist()
    
    def scale_polygon(self, polygon, scale, center, matrix):
        points = np.array(polygon).reshape(-1, 2)
        ones = np.ones(shape=(len(points), 1))
        points_ones = np.hstack([points, ones])
        transformed_points = matrix.dot(points_ones.T).T
        return np.round(transformed_points).astype(int).flatten().tolist()
    
    def flip_polygon(self, polygon, flip_code, image_shape):
        points = np.array(polygon).reshape(-1, 2)
        if flip_code == 0:  # Vertical flip
            points[:, 1] = image_shape[0] - points[:, 1]
        elif flip_code == 1:  # Horizontal flip
            points[:, 0] = image_shape[1] - points[:, 0]
        elif flip_code == -1:  # Both
            points[:, 0] = image_shape[1] - points[:, 0]
            points[:, 1] = image_shape[0] - points[:, 1]
        return np.round(points).astype(int).flatten().tolist()
    


    def get_bbox_from_polygon(self, polygon):
        points = np.array(polygon).reshape(-1, 2)
        x_min, y_min = np.min(points, axis=0)
        x_max, y_max = np.max(points, axis=0)
        return [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)]




    def show_centered(self, parent):
        parent_geo = parent.geometry()
        self.move(parent_geo.center() - self.rect().center())
        self.show()

def show_image_augmenter(parent):
    dialog = ImageAugmenterDialog(parent)
    dialog.show_centered(parent)
    return dialog