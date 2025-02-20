import json
import os
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QFileDialog, QLabel, QMessageBox, QApplication)
from PyQt5.QtCore import Qt

class COCOJSONCombinerDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("COCO JSON Combiner")
        self.setGeometry(100, 100, 400, 300)
        self.setWindowFlags(self.windowFlags() | Qt.Window)
        self.setWindowModality(Qt.ApplicationModal)
        self.json_files = []
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        self.file_labels = []
        for i in range(5):
            file_layout = QHBoxLayout()
            label = QLabel(f"File {i+1}: Not selected")
            self.file_labels.append(label)
            file_layout.addWidget(label)
            select_button = QPushButton(f"Select File {i+1}")
            select_button.clicked.connect(lambda checked, x=i: self.select_file(x))
            file_layout.addWidget(select_button)
            layout.addLayout(file_layout)

        self.combine_button = QPushButton("Combine JSON Files")
        self.combine_button.clicked.connect(self.combine_json_files)
        self.combine_button.setEnabled(False)
        layout.addWidget(self.combine_button)

        self.setLayout(layout)

    def select_file(self, index):
        file_name, _ = QFileDialog.getOpenFileName(self, f"Select COCO JSON File {index+1}", "", "JSON Files (*.json)")
        if file_name:
            if file_name not in self.json_files:
                self.json_files.append(file_name)
                self.file_labels[index].setText(f"File {index+1}: {os.path.basename(file_name)}")
                self.combine_button.setEnabled(True)
            else:
                QMessageBox.warning(self, "Duplicate File", "This file has already been selected.")
        QApplication.processEvents()


    def combine_json_files(self):
        if not self.json_files:
            QMessageBox.warning(self, "No Files", "Please select at least one JSON file to combine.")
            return
    
        combined_data = {
            "images": [],
            "annotations": [],
            "categories": []
        }
        image_file_names = set()
        next_image_id = 1
        next_annotation_id = 1
    
        try:
            for file_path in self.json_files:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Combine categories
                category_id_map = {}
                for category in data.get('categories', []):
                    existing_category = next((c for c in combined_data['categories'] if c['name'] == category['name']), None)
                    if existing_category:
                        category_id_map[category['id']] = existing_category['id']
                    else:
                        new_id = len(combined_data['categories']) + 1
                        category_id_map[category['id']] = new_id
                        category['id'] = new_id
                        combined_data['categories'].append(category)
    
                # Combine images and annotations
                image_id_map = {}
                for image in data.get('images', []):
                    if image['file_name'] not in image_file_names:
                        image_file_names.add(image['file_name'])
                        image_id_map[image['id']] = next_image_id
                        image['id'] = next_image_id
                        combined_data['images'].append(image)
                        next_image_id += 1
    
                for annotation in data.get('annotations', []):
                    if annotation['image_id'] in image_id_map:
                        annotation['id'] = next_annotation_id
                        annotation['image_id'] = image_id_map[annotation['image_id']]
                        annotation['category_id'] = category_id_map[annotation['category_id']]
                        combined_data['annotations'].append(annotation)
                        next_annotation_id += 1
    
            output_file, _ = QFileDialog.getSaveFileName(self, "Save Combined JSON", "", "JSON Files (*.json)")
            if output_file:
                with open(output_file, 'w') as f:
                    json.dump(combined_data, f, indent=2)
                QMessageBox.information(self, "Success", f"Combined JSON saved to {output_file}")
    
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred while combining JSON files: {str(e)}")



    def show_centered(self, parent):
        parent_geo = parent.geometry()
        self.move(parent_geo.center() - self.rect().center())
        self.show()

def show_coco_json_combiner(parent):
    dialog = COCOJSONCombinerDialog(parent)
    dialog.show_centered(parent)
    return dialog