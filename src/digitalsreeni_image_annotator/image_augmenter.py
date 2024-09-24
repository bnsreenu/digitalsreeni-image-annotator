import os
import random
import cv2
import numpy as np
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QFileDialog, QLabel, QMessageBox, QSpinBox, 
                             QCheckBox, QDoubleSpinBox, QProgressBar, QApplication)
from PyQt5.QtCore import Qt

class ImageAugmenterDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Image Augmenter")
        self.setGeometry(100, 100, 400, 500)
        self.setWindowFlags(self.windowFlags() | Qt.Window)
        self.setWindowModality(Qt.ApplicationModal)
        self.input_dir = ""
        self.output_dir = ""
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
        layout.addLayout(flip_layout)


        # Elastic Deformation
        self.elastic_check = QCheckBox("Elastic Deformation")
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

    def start_augmentation(self):
        if not self.input_dir or not self.output_dir:
            QMessageBox.warning(self, "Missing Directory", "Please select both input and output directories.")
            return

        image_files = [f for f in os.listdir(self.input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        total_augmentations = len(image_files) * self.aug_count_spin.value()

        self.progress_bar.setMaximum(total_augmentations)
        self.progress_bar.setValue(0)

        for i, image_file in enumerate(image_files):
            input_path = os.path.join(self.input_dir, image_file)
            image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
            
            # Determine image type and bit depth
            is_color = len(image.shape) == 3 and image.shape[2] == 3
            bit_depth = image.dtype

            for j in range(self.aug_count_spin.value()):
                augmented = self.apply_random_augmentation(image)
                
                # Ensure the augmented image has the same properties as the input
                if not is_color and len(augmented.shape) == 3:
                    augmented = cv2.cvtColor(augmented, cv2.COLOR_BGR2GRAY)
                elif is_color and len(augmented.shape) == 2:
                    augmented = cv2.cvtColor(augmented, cv2.COLOR_GRAY2BGR)
                
                augmented = augmented.astype(bit_depth)

                output_filename = f"{os.path.splitext(image_file)[0]}_aug_{j+1}{os.path.splitext(image_file)[1]}"
                output_path = os.path.join(self.output_dir, output_filename)
                cv2.imwrite(output_path, augmented)

                self.progress_bar.setValue(i * self.aug_count_spin.value() + j + 1)
                QApplication.processEvents()

        QMessageBox.information(self, "Augmentation Complete", "Image augmentation has been completed successfully.")

    def apply_random_augmentation(self, image):
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
        if self.elastic_check.isChecked():
            augmentations.append(self.elastic_transform)
        if self.grayscale_check.isChecked():
            augmentations.append(self.convert_to_grayscale)
        if self.hist_equalize_check.isChecked():
            augmentations.append(self.apply_histogram_equalization)

        
        if not augmentations:
            return image
        
        aug_func = random.choice(augmentations)
        return aug_func(image)

    def rotate_image(self, image):
        angle = random.uniform(-self.rotate_spin.value(), self.rotate_spin.value())
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        return cv2.warpAffine(image, M, (w, h))

    def zoom_image(self, image):
        scale = random.uniform(1, 1 + self.zoom_spin.value())
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), 0, scale)
        return cv2.warpAffine(image, M, (w, h))

    def blur_image(self, image):
        kernel_size = random.choice([3, 5, 7])
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    def adjust_brightness_contrast(self, image):
        alpha = random.uniform(0.5, 1.5)  # Contrast control
        beta = random.uniform(-30, 30)    # Brightness control
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    def sharpen_image(self, image):
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        return cv2.filter2D(image, -1, kernel)
    
    def flip_image(self, image):
        flip_options = []
        if self.flip_horizontal_check.isChecked():
            flip_options.append(1)  # Horizontal flip
        if self.flip_vertical_check.isChecked():
            flip_options.append(0)  # Vertical flip
        if self.flip_horizontal_check.isChecked() and self.flip_vertical_check.isChecked():
            flip_options.append(-1)  # Both horizontal and vertical

        if not flip_options:
            return image  # No flip if none selected

        flip_code = random.choice(flip_options)
        return cv2.flip(image, flip_code)

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
        mapx = np.float32(x + dx)
        mapy = np.float32(y + dy)
        return cv2.remap(image, mapx, mapy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    def convert_to_grayscale(self, image):
        if len(image.shape) == 2:
            return image  # Already grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  # Convert back to 3 channels

    def apply_histogram_equalization(self, image):
        if len(image.shape) == 2:  # Grayscale image
            return cv2.equalizeHist(image)
        else:  # Color image
            ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            ycrcb[:,:,0] = cv2.equalizeHist(ycrcb[:,:,0])
            return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    def show_centered(self, parent):
        parent_geo = parent.geometry()
        self.move(parent_geo.center() - self.rect().center())
        self.show()

def show_image_augmenter(parent):
    dialog = ImageAugmenterDialog(parent)
    dialog.show_centered(parent)
    return dialog