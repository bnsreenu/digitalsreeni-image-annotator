import os
import numpy as np
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                             QSpinBox, QProgressBar, QMessageBox, QListWidget, QDialogButtonBox,
                             QGridLayout, QComboBox, QApplication)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtCore import QTimer, QEventLoop
from tifffile import TiffFile, imsave
from PIL import Image
import traceback

class DimensionDialog(QDialog):
    def __init__(self, shape, file_name, parent=None):
        super().__init__(parent)
        self.shape = shape
        self.file_name = file_name
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        layout.addWidget(QLabel(f"File: {self.file_name}"))
        layout.addWidget(QLabel(f"Image shape: {self.shape}"))
        layout.addWidget(QLabel("Assign dimensions:"))

        grid_layout = QGridLayout()
        self.combos = []
        dimensions = ['T', 'Z', 'C', 'H', 'W']
        for i, dim in enumerate(self.shape):
            grid_layout.addWidget(QLabel(f"Dimension {i} (size {dim}):"), i, 0)
            combo = QComboBox()
            combo.addItems(dimensions)
            grid_layout.addWidget(combo, i, 1)
            self.combos.append(combo)
        layout.addLayout(grid_layout)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

    def get_dimensions(self):
        return [combo.currentText() for combo in self.combos]

class PatchingThread(QThread):
    progress = pyqtSignal(int)
    error = pyqtSignal(str)
    finished = pyqtSignal()
    dimension_required = pyqtSignal(object, str)


    def __init__(self, input_files, output_dir, patch_size, overlap, dimensions):
        super().__init__()
        self.input_files = input_files
        self.output_dir = output_dir
        self.patch_size = patch_size
        self.overlap = overlap  # Changed to tuple (to handle overlap_x, overlap_y independently) - Sreeni
        self.dimensions = dimensions

    def run(self):
        try:
            total_files = len(self.input_files)
            for i, file_path in enumerate(self.input_files):
                self.patch_image(file_path)
                self.progress.emit(int((i + 1) / total_files * 100))
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))
            traceback.print_exc()

    def patch_image(self, file_path):
        file_name = os.path.basename(file_path)
        file_name_without_ext, file_extension = os.path.splitext(file_name)

        if file_extension.lower() in ['.tif', '.tiff']:
            with TiffFile(file_path) as tif:
                images = tif.asarray()
                if images.ndim > 2:
                    if file_path not in self.dimensions:
                        self.dimension_required.emit(images.shape, file_name)
                        self.wait()
                    dimensions = self.dimensions.get(file_path)
                    if dimensions:
                        if 'H' in dimensions and 'W' in dimensions:
                            h_index = dimensions.index('H')
                            w_index = dimensions.index('W')
                            for idx in np.ndindex(images.shape[:h_index] + images.shape[h_index+2:]):
                                slice_idx = idx[:h_index] + (slice(None), slice(None)) + idx[h_index:]
                                image = images[slice_idx]
                                slice_name = '_'.join([f'{dim}{i+1}' for dim, i in zip(dimensions, idx) if dim not in ['H', 'W']])
                                self.save_patches(image, f"{file_name_without_ext}_{slice_name}", file_extension)
                        else:
                            raise ValueError("You must assign both H and W dimensions.")
                    else:
                        raise ValueError("Dimensions were not properly assigned.")
                else:
                    self.save_patches(images, file_name_without_ext, file_extension)
        else:
            with Image.open(file_path) as img:
                image = np.array(img)
                self.save_patches(image, file_name_without_ext, file_extension)

    def save_patches(self, image, base_name, extension):
        h, w = image.shape[:2]
        patch_h, patch_w = self.patch_size
        overlap_x, overlap_y = self.overlap

        for i in range(0, h - overlap_y, patch_h - overlap_y):
            for j in range(0, w - overlap_x, patch_w - overlap_x):
                if i + patch_h <= h and j + patch_w <= w:  # Only save full-sized patches
                    patch = image[i:i+patch_h, j:j+patch_w]
                    patch_name = f"{base_name}_patch_{i}_{j}{extension}"
                    output_path = os.path.join(self.output_dir, patch_name)

                    if extension.lower() in ['.tif', '.tiff']:
                        imsave(output_path, patch)
                    else:
                        Image.fromarray(patch).save(output_path)

class ImagePatcherTool(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowModality(Qt.ApplicationModal)
        self.dimensions = {}
        self.input_files = []
        self.output_dir = ""
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Input files selection
        input_layout = QHBoxLayout()
        self.input_label = QLabel("Input Files:")
        self.input_button = QPushButton("Select Files")
        self.input_button.clicked.connect(self.select_input_files)
        input_layout.addWidget(self.input_label)
        input_layout.addWidget(self.input_button)
        layout.addLayout(input_layout)

        # Output directory selection
        output_layout = QHBoxLayout()
        self.output_label = QLabel("Output Directory:")
        self.output_button = QPushButton("Select Directory")
        self.output_button.clicked.connect(self.select_output_directory)
        output_layout.addWidget(self.output_label)
        output_layout.addWidget(self.output_button)
        layout.addLayout(output_layout)

        # Patch size inputs
        patch_layout = QHBoxLayout()
        patch_layout.addWidget(QLabel("Patch Size (W x H):"))
        self.patch_w = QSpinBox()
        self.patch_w.setRange(1, 10000)
        self.patch_w.setValue(256)
        self.patch_h = QSpinBox()
        self.patch_h.setRange(1, 10000)
        self.patch_h.setValue(256)
        patch_layout.addWidget(self.patch_w)
        patch_layout.addWidget(self.patch_h)
        layout.addLayout(patch_layout)

        # Overlap inputs
        overlap_layout = QHBoxLayout()
        overlap_layout.addWidget(QLabel("Overlap (X, Y):"))
        self.overlap_x = QSpinBox()
        self.overlap_x.setRange(0, 1000)
        self.overlap_x.setValue(0)
        self.overlap_y = QSpinBox()
        self.overlap_y.setRange(0, 1000)
        self.overlap_y.setValue(0)
        overlap_layout.addWidget(self.overlap_x)
        overlap_layout.addWidget(self.overlap_y)
        layout.addLayout(overlap_layout)

        # Patch info label
        self.patch_info_label = QLabel()
        self.patch_info_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.patch_info_label)

        # Start button
        self.start_button = QPushButton("Start Patching")
        self.start_button.clicked.connect(self.start_patching)
        layout.addWidget(self.start_button)

        # Progress bar
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        self.setWindowTitle('Image Patcher Tool')
        self.setGeometry(300, 300, 400, 300)

        # Connect value changed signals
        self.patch_w.valueChanged.connect(self.update_patch_info)
        self.patch_h.valueChanged.connect(self.update_patch_info)
        self.overlap_x.valueChanged.connect(self.update_patch_info)
        self.overlap_y.valueChanged.connect(self.update_patch_info)

    def select_input_files(self):
        file_dialog = QFileDialog()
        self.input_files, _ = file_dialog.getOpenFileNames(self, "Select Input Files", "", "Image Files (*.png *.jpg *.bmp *.tif *.tiff)")
        self.input_label.setText(f"Input Files: {len(self.input_files)} selected")
        QApplication.processEvents()
        self.process_tiff_files()
        self.update_patch_info()
        
    def process_tiff_files(self):
        for file_path in self.input_files:
            if file_path.lower().endswith(('.tif', '.tiff')):
                self.check_tiff_dimensions(file_path)
            QApplication.processEvents()
            
            
            
    def check_tiff_dimensions(self, file_path):
        with TiffFile(file_path) as tif:
            images = tif.asarray()
            if images.ndim > 2:
                file_name = os.path.basename(file_path)
                dialog = DimensionDialog(images.shape, file_name, self)
                dialog.setWindowModality(Qt.ApplicationModal)
                result = dialog.exec_()
                if result == QDialog.Accepted:
                    dimensions = dialog.get_dimensions()
                    if 'H' in dimensions and 'W' in dimensions:
                        self.dimensions[file_path] = dimensions
                    else:
                        QMessageBox.warning(self, "Invalid Dimensions", f"You must assign both H and W dimensions for {file_name}.")
                QApplication.processEvents()



    def select_output_directory(self):
        file_dialog = QFileDialog()
        self.output_dir = file_dialog.getExistingDirectory(self, "Select Output Directory")
        dir_name = os.path.basename(self.output_dir) if self.output_dir else ""
        self.output_label.setText(f"Output Directory: {dir_name}")
        QApplication.processEvents()
        self.update_patch_info()
        
        
    def start_patching(self):
        if not self.input_files:
            QMessageBox.warning(self, "No Input Files", "Please select input files.")
            return
        if not self.output_dir:
            QMessageBox.warning(self, "No Output Directory", "Please select an output directory.")
            return

        patch_size = (self.patch_h.value(), self.patch_w.value())
        overlap = (self.overlap_x.value(), self.overlap_y.value())

        self.patching_thread = PatchingThread(self.input_files, self.output_dir, patch_size, overlap, self.dimensions)
        self.patching_thread.progress.connect(self.update_progress)
        self.patching_thread.error.connect(self.show_error)
        self.patching_thread.finished.connect(self.patching_finished)
        self.patching_thread.dimension_required.connect(self.get_dimensions)
        self.patching_thread.start()
    
        self.start_button.setEnabled(False)


    def get_dimensions(self, shape, file_name):
        dialog = DimensionDialog(shape, file_name, self)
        dialog.setWindowModality(Qt.ApplicationModal)
        result = dialog.exec_()
        
        if result == QDialog.Accepted:
            dimensions = dialog.get_dimensions()
            if 'H' in dimensions and 'W' in dimensions:
                self.dimensions[file_name] = dimensions
            else:
                QMessageBox.warning(self, "Invalid Dimensions", f"You must assign both H and W dimensions for {file_name}.")
        QApplication.processEvents()
        self.patching_thread.wake()



    def get_patch_info(self):
        patch_info = {}
        patch_w = self.patch_w.value()
        patch_h = self.patch_h.value()
        overlap_x = self.overlap_x.value()
        overlap_y = self.overlap_y.value()

        for file_path in self.input_files:
            file_name = os.path.basename(file_path)
            if file_path.lower().endswith(('.tif', '.tiff')):
                with TiffFile(file_path) as tif:
                    images = tif.asarray()
                    if images.ndim > 2:
                        dimensions = self.dimensions.get(file_path)
                        if dimensions:
                            h_index = dimensions.index('H')
                            w_index = dimensions.index('W')
                            h, w = images.shape[h_index], images.shape[w_index]
                        else:
                            h, w = images.shape[-2], images.shape[-1]
                    else:
                        h, w = images.shape
            else:
                with Image.open(file_path) as img:
                    w, h = img.size

            patches_x = (w - overlap_x) // (patch_w - overlap_x)
            patches_y = (h - overlap_y) // (patch_h - overlap_y)
            leftover_x = w - (patches_x * (patch_w - overlap_x) + overlap_x)
            leftover_y = h - (patches_y * (patch_h - overlap_y) + overlap_y)

            patch_info[file_name] = {
                'patches_x': patches_x,
                'patches_y': patches_y,
                'leftover_x': leftover_x,
                'leftover_y': leftover_y
            }

        return patch_info

    def update_patch_info(self):
        if not self.input_files:
            self.patch_info_label.setText("No input files selected")
            return

        patch_info = self.get_patch_info()
        if patch_info:
            info_text = "Patch Information:\n\n"
            for file_name, info in patch_info.items():
                info_text += f"File: {file_name}\n"
                info_text += f"Patches in X: {info['patches_x']}, Y: {info['patches_y']}\n"
                info_text += f"Leftover pixels in X: {info['leftover_x']}, Y: {info['leftover_y']}\n\n"
            self.patch_info_label.setText(info_text)
        else:
            self.patch_info_label.setText("Unable to calculate patch information")





    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def show_error(self, error_message):
        QMessageBox.critical(self, "Error", f"An error occurred during patching:\n{error_message}")
        self.start_button.setEnabled(True)

    def patching_finished(self):
        self.start_button.setEnabled(True)
        QMessageBox.information(self, "Patching Complete", "Image patching has been completed.")

    def show_centered(self, parent):
        parent_geo = parent.geometry()
        self.move(parent_geo.center() - self.rect().center())
        self.show()

def show_image_patcher(parent=None):
    dialog = ImagePatcherTool(parent)
    dialog.show_centered(parent)
    return dialog