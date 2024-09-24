import os
import numpy as np
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QFileDialog, QLabel, QMessageBox, QComboBox, QGridLayout, QWidget,
                             QProgressDialog, QApplication)
from PyQt5.QtCore import Qt
from tifffile import TiffFile
from czifile import CziFile
from PIL import Image

class DimensionDialog(QDialog):
    def __init__(self, shape, file_name, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Assign Dimensions")
        self.shape = shape
        self.initUI(file_name)

    def initUI(self, file_name):
        layout = QVBoxLayout()
        
        file_name_label = QLabel(f"File: {file_name}")
        file_name_label.setWordWrap(True)
        layout.addWidget(file_name_label)
        
        dim_widget = QWidget()
        dim_layout = QGridLayout(dim_widget)
        self.combos = []
        dimensions = ['T', 'Z', 'C', 'S', 'H', 'W']
        for i, dim in enumerate(self.shape):
            dim_layout.addWidget(QLabel(f"Dimension {i} (size {dim}):"), i, 0)
            combo = QComboBox()
            combo.addItems(dimensions)
            dim_layout.addWidget(combo, i, 1)
            self.combos.append(combo)
        layout.addWidget(dim_widget)
        
        self.button = QPushButton("OK")
        self.button.clicked.connect(self.accept)
        layout.addWidget(self.button)
        
        self.setLayout(layout)

    def get_dimensions(self):
        return [combo.currentText() for combo in self.combos]


class StackToSlicesDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Stack to Slices")
        self.setGeometry(100, 100, 400, 200)
        self.setWindowFlags(self.windowFlags() | Qt.Window)
        self.setWindowModality(Qt.ApplicationModal)
        self.dimensions = None
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        self.file_label = QLabel("No file selected")
        layout.addWidget(self.file_label)

        select_button = QPushButton("Select Stack File")
        select_button.clicked.connect(self.select_file)
        layout.addWidget(select_button)

        self.convert_button = QPushButton("Convert to Slices")
        self.convert_button.clicked.connect(self.convert_to_slices)
        self.convert_button.setEnabled(False)
        layout.addWidget(self.convert_button)

        self.setLayout(layout)

    def select_file(self):
        self.file_name, _ = QFileDialog.getOpenFileName(self, "Select Stack File", "", "Image Files (*.tif *.tiff *.czi)")
        if self.file_name:
            self.file_label.setText(f"Selected file: {os.path.basename(self.file_name)}")
            QApplication.processEvents()
            self.process_file()

    def process_file(self):
        if self.file_name.lower().endswith(('.tif', '.tiff')):
            self.process_tiff()
        elif self.file_name.lower().endswith('.czi'):
            self.process_czi()

    def process_tiff(self):
        with TiffFile(self.file_name) as tif:
            image_array = tif.asarray()
        
        self.get_dimensions(image_array.shape)

    def process_czi(self):
        with CziFile(self.file_name) as czi:
            image_array = czi.asarray()
        
        self.get_dimensions(image_array.shape)

    def get_dimensions(self, shape):
        dialog = DimensionDialog(shape, os.path.basename(self.file_name), self)
        dialog.setWindowModality(Qt.ApplicationModal)
        if dialog.exec_():
            self.dimensions = dialog.get_dimensions()
            self.convert_button.setEnabled(True)
        else:
            self.dimensions = None
            self.convert_button.setEnabled(False)
        QApplication.processEvents()

    def convert_to_slices(self):
        if not hasattr(self, 'file_name') or not self.dimensions:
            QMessageBox.warning(self, "Invalid Input", "Please select a file and assign dimensions first.")
            return

        output_dir = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if not output_dir:
            return

        if self.file_name.lower().endswith(('.tif', '.tiff')):
            with TiffFile(self.file_name) as tif:
                image_array = tif.asarray()
        elif self.file_name.lower().endswith('.czi'):
            with CziFile(self.file_name) as czi:
                image_array = czi.asarray()

        self.save_slices(image_array, output_dir)

    def save_slices(self, image_array, output_dir):
        base_name = os.path.splitext(os.path.basename(self.file_name))[0]
        
        slice_indices = [i for i, dim in enumerate(self.dimensions) if dim not in ['H', 'W']]

        total_slices = np.prod([image_array.shape[i] for i in slice_indices])
        
        progress = QProgressDialog("Saving slices...", "Cancel", 0, total_slices, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setWindowTitle("Progress")
        progress.setMinimumDuration(0)
        progress.setValue(0)
        progress.show()

        try:
            for idx, _ in enumerate(np.ndindex(tuple(image_array.shape[i] for i in slice_indices))):
                if progress.wasCanceled():
                    break

                full_idx = [slice(None)] * len(self.dimensions)
                for i, val in zip(slice_indices, _):
                    full_idx[i] = val
                
                slice_array = image_array[tuple(full_idx)]
                
                if slice_array.ndim > 2:
                    slice_array = slice_array.squeeze()
                
                if slice_array.dtype == np.uint16:
                    mode = 'I;16'
                elif slice_array.dtype == np.uint8:
                    mode = 'L'
                else:
                    slice_array = ((slice_array - slice_array.min()) / (slice_array.max() - slice_array.min()) * 65535).astype(np.uint16)
                    mode = 'I;16'

                slice_name = f"{base_name}_{'_'.join([f'{self.dimensions[i]}{val+1}' for i, val in zip(slice_indices, _)])}.png"
                img = Image.fromarray(slice_array, mode=mode)
                img.save(os.path.join(output_dir, slice_name))

                progress.setValue(idx + 1)
                QApplication.processEvents()

            if progress.wasCanceled():
                QMessageBox.warning(self, "Conversion Interrupted", "The conversion process was interrupted.")
            else:
                QMessageBox.information(self, "Conversion Complete", f"All slices have been saved to {output_dir}")
        
        finally:
            progress.close()

    def show_centered(self, parent):
        parent_geo = parent.geometry()
        self.move(parent_geo.center() - self.rect().center())
        self.show()

def show_stack_to_slices(parent):
    dialog = StackToSlicesDialog(parent)
    dialog.show_centered(parent)
    return dialog