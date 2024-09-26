import os
import json
import shutil
import random
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, 
                             QLabel, QSpinBox, QRadioButton, QButtonGroup, QMessageBox)
from PyQt5.QtCore import Qt

class DatasetSplitterTool(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Dataset Splitter")
        self.setGeometry(100, 100, 500, 300)
        self.setWindowFlags(self.windowFlags() | Qt.Window)
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # Option selection
        options_layout = QVBoxLayout()
        self.images_only_radio = QRadioButton("Images Only")
        options_layout.addWidget(self.images_only_radio)

        images_annotations_layout = QHBoxLayout()
        self.images_annotations_radio = QRadioButton("Images and Annotations")
        images_annotations_layout.addWidget(self.images_annotations_radio)
        self.select_json_button = QPushButton("Upload COCO JSON File")
        self.select_json_button.clicked.connect(self.select_json_file)
        self.select_json_button.setEnabled(False)
        images_annotations_layout.addWidget(self.select_json_button)
        options_layout.addLayout(images_annotations_layout)

        layout.addLayout(options_layout)
        
        option_group = QButtonGroup(self)
        option_group.addButton(self.images_only_radio)
        option_group.addButton(self.images_annotations_radio)
        
        self.images_only_radio.setChecked(True)

        # Percentage inputs
        train_layout = QHBoxLayout()
        train_layout.addWidget(QLabel("Train %:"))
        self.train_percent = QSpinBox()
        self.train_percent.setRange(0, 100)
        self.train_percent.setValue(70)
        train_layout.addWidget(self.train_percent)
        layout.addLayout(train_layout)

        val_layout = QHBoxLayout()
        val_layout.addWidget(QLabel("Validation %:"))
        self.val_percent = QSpinBox()
        self.val_percent.setRange(0, 100)
        self.val_percent.setValue(30)
        val_layout.addWidget(self.val_percent)
        layout.addLayout(val_layout)

        test_layout = QHBoxLayout()
        test_layout.addWidget(QLabel("Test %:"))
        self.test_percent = QSpinBox()
        self.test_percent.setRange(0, 100)
        self.test_percent.setValue(0)
        test_layout.addWidget(self.test_percent)
        layout.addLayout(test_layout)

        # Buttons
        self.select_input_button = QPushButton("Select Input Directory")
        self.select_input_button.clicked.connect(self.select_input_directory)
        layout.addWidget(self.select_input_button)

        self.select_output_button = QPushButton("Select Output Directory")
        self.select_output_button.clicked.connect(self.select_output_directory)
        layout.addWidget(self.select_output_button)

        self.split_button = QPushButton("Split Dataset")
        self.split_button.clicked.connect(self.split_dataset)
        layout.addWidget(self.split_button)

        self.setLayout(layout)

        self.input_directory = ""
        self.output_directory = ""
        self.json_file = ""

        # Connect radio buttons to enable/disable JSON selection
        self.images_only_radio.toggled.connect(self.toggle_json_selection)
        self.images_annotations_radio.toggled.connect(self.toggle_json_selection)

    def toggle_json_selection(self):
        self.select_json_button.setEnabled(self.images_annotations_radio.isChecked())

    def select_input_directory(self):
        self.input_directory = QFileDialog.getExistingDirectory(self, "Select Input Directory")

    def select_output_directory(self):
        self.output_directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")

    def select_json_file(self):
        self.json_file, _ = QFileDialog.getOpenFileName(self, "Select COCO JSON File", "", "JSON Files (*.json)")

    def split_dataset(self):
        if not self.input_directory or not self.output_directory:
            QMessageBox.warning(self, "Error", "Please select input and output directories.")
            return

        if self.images_annotations_radio.isChecked() and not self.json_file:
            QMessageBox.warning(self, "Error", "Please select a COCO JSON file.")
            return

        train_percent = self.train_percent.value()
        val_percent = self.val_percent.value()
        test_percent = self.test_percent.value()

        if train_percent + val_percent + test_percent != 100:
            QMessageBox.warning(self, "Error", "Percentages must add up to 100%.")
            return

        if self.images_only_radio.isChecked():
            self.split_images_only()
        else:
            self.split_images_and_annotations()

    def split_images_only(self):
        image_files = [f for f in os.listdir(self.input_directory) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
        random.shuffle(image_files)

        train_split = int(len(image_files) * self.train_percent.value() / 100)
        val_split = int(len(image_files) * self.val_percent.value() / 100)

        train_images = image_files[:train_split]
        val_images = image_files[train_split:train_split + val_split]
        test_images = image_files[train_split + val_split:]

        self.copy_images(train_images, "train", images_only=True)
        self.copy_images(val_images, "val", images_only=True)
        self.copy_images(test_images, "test", images_only=True)

        QMessageBox.information(self, "Success", "Dataset split successfully!")

    def split_images_and_annotations(self):
        with open(self.json_file, 'r') as f:
            coco_data = json.load(f)

        image_files = [img['file_name'] for img in coco_data['images']]
        random.shuffle(image_files)

        train_split = int(len(image_files) * self.train_percent.value() / 100)
        val_split = int(len(image_files) * self.val_percent.value() / 100)

        train_images = image_files[:train_split]
        val_images = image_files[train_split:train_split + val_split]
        test_images = image_files[train_split + val_split:]

        self.copy_images(train_images, "train", images_only=False)
        self.copy_images(val_images, "val", images_only=False)
        self.copy_images(test_images, "test", images_only=False)

        self.split_coco_annotations(coco_data, train_images, val_images, test_images)

        QMessageBox.information(self, "Success", "Dataset and annotations split successfully!")

    def copy_images(self, image_list, subset, images_only=False):
        if images_only:
            subset_dir = os.path.join(self.output_directory, subset)
        else:
            subset_dir = os.path.join(self.output_directory, subset, "images")
        os.makedirs(subset_dir, exist_ok=True)
        
        for image in image_list:
            src = os.path.join(self.input_directory, image)
            dst = os.path.join(subset_dir, image)
            shutil.copy2(src, dst)

    def split_coco_annotations(self, coco_data, train_images, val_images, test_images):
        def create_subset_annotations(subset_images):
            subset_images_data = [img for img in coco_data['images'] if img['file_name'] in subset_images]
            subset_image_ids = [img['id'] for img in subset_images_data]
            
            return {
                "images": subset_images_data,
                "annotations": [ann for ann in coco_data['annotations'] if ann['image_id'] in subset_image_ids],
                "categories": coco_data['categories']
            }

        train_data = create_subset_annotations(train_images)
        val_data = create_subset_annotations(val_images)
        test_data = create_subset_annotations(test_images)

        self.save_coco_annotations(train_data, "train")
        self.save_coco_annotations(val_data, "val")
        self.save_coco_annotations(test_data, "test")

    def save_coco_annotations(self, data, subset):
        output_file = os.path.join(self.output_directory, subset, f"{subset}_annotations.json")
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)

    def show_centered(self, parent):
        parent_geo = parent.geometry()
        self.move(parent_geo.center() - self.rect().center())
        self.show()