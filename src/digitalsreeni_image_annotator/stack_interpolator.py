import os
import numpy as np
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, 
                            QLabel, QComboBox, QMessageBox, QProgressDialog, QRadioButton,
                            QButtonGroup, QGroupBox, QDoubleSpinBox, QApplication)
from PyQt5.QtCore import Qt
from scipy.interpolate import RegularGridInterpolator
from skimage import io
import tifffile

class StackInterpolator(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Stack Interpolator")
        self.setGeometry(100, 100, 600, 400)
        self.setWindowFlags(self.windowFlags() | Qt.Window)
        self.setWindowModality(Qt.ApplicationModal)  # Added window modality
        
        # Initialize variables
        self.input_path = ""
        self.output_directory = ""
        
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        layout.setSpacing(10)  # Add consistent spacing

        # Input selection
        input_group = QGroupBox("Input Selection")
        input_layout = QVBoxLayout()
        
        # Radio buttons for input type
        self.dir_radio = QRadioButton("Directory of Image Files")
        self.stack_radio = QRadioButton("TIFF Stack")
        
        input_group_buttons = QButtonGroup(self)
        input_group_buttons.addButton(self.dir_radio)
        input_group_buttons.addButton(self.stack_radio)
        
        input_layout.addWidget(self.dir_radio)
        input_layout.addWidget(self.stack_radio)
        self.dir_radio.setChecked(True)
        
        input_group.setLayout(input_layout)
        layout.addWidget(input_group)

        # Interpolation method
        method_group = QGroupBox("Interpolation Settings")
        method_layout = QVBoxLayout()
        
        method_combo_layout = QHBoxLayout()
        method_combo_layout.addWidget(QLabel("Method:"))
        self.method_combo = QComboBox()
        self.method_combo.addItems([
            "linear",
            "nearest",
            "slinear",
            "cubic",
            "quintic",
            "pchip"
        ])
        method_combo_layout.addWidget(self.method_combo)
        method_layout.addLayout(method_combo_layout)
        
        method_group.setLayout(method_layout)
        layout.addWidget(method_group)

        # Original dimensions group
        orig_group = QGroupBox("Original Dimensions")
        orig_layout = QVBoxLayout()
        
        orig_xy_layout = QHBoxLayout()
        orig_xy_layout.addWidget(QLabel("XY Pixel Size:"))
        self.orig_xy_size = QDoubleSpinBox()
        self.orig_xy_size.setRange(0.001, 1000.0)
        self.orig_xy_size.setValue(1.0)
        self.orig_xy_size.setDecimals(3)
        orig_xy_layout.addWidget(self.orig_xy_size)
        
        orig_z_layout = QHBoxLayout()
        orig_z_layout.addWidget(QLabel("Z Spacing:"))
        self.orig_z_size = QDoubleSpinBox()
        self.orig_z_size.setRange(0.001, 1000.0)
        self.orig_z_size.setValue(1.0)
        self.orig_z_size.setDecimals(3)
        orig_z_layout.addWidget(self.orig_z_size)
        
        orig_layout.addLayout(orig_xy_layout)
        orig_layout.addLayout(orig_z_layout)
        orig_group.setLayout(orig_layout)
        layout.addWidget(orig_group)

        # New dimensions group
        new_group = QGroupBox("New Dimensions")
        new_layout = QVBoxLayout()
        
        new_xy_layout = QHBoxLayout()
        new_xy_layout.addWidget(QLabel("XY Pixel Size:"))
        self.new_xy_size = QDoubleSpinBox()
        self.new_xy_size.setRange(0.001, 1000.0)
        self.new_xy_size.setValue(1.0)
        self.new_xy_size.setDecimals(3)
        new_xy_layout.addWidget(self.new_xy_size)
        
        new_z_layout = QHBoxLayout()
        new_z_layout.addWidget(QLabel("Z Spacing:"))
        self.new_z_size = QDoubleSpinBox()
        self.new_z_size.setRange(0.001, 1000.0)
        self.new_z_size.setValue(1.0)
        self.new_z_size.setDecimals(3)
        new_z_layout.addWidget(self.new_z_size)
        
        new_layout.addLayout(new_xy_layout)
        new_layout.addLayout(new_z_layout)
        new_group.setLayout(new_layout)
        layout.addWidget(new_group)

        # Units selector
        unit_group = QGroupBox("Unit Settings")
        unit_layout = QHBoxLayout()
        unit_layout.addWidget(QLabel("Unit:"))
        self.size_unit = QComboBox()
        self.size_unit.addItems(["nm", "µm", "mm"])
        self.size_unit.setCurrentText("µm")
        unit_layout.addWidget(self.size_unit)
        unit_group.setLayout(unit_layout)
        layout.addWidget(unit_group)

        # Input/Output buttons
        button_group = QGroupBox("File Selection")
        button_layout = QVBoxLayout()
        
        # Input selection
        input_file_layout = QHBoxLayout()
        self.input_label = QLabel("No input selected")
        self.select_input_btn = QPushButton("Select Input")
        self.select_input_btn.clicked.connect(self.select_input)
        input_file_layout.addWidget(self.select_input_btn)
        input_file_layout.addWidget(self.input_label)
        button_layout.addLayout(input_file_layout)
        
        # Output selection
        output_file_layout = QHBoxLayout()
        self.output_label = QLabel("No output directory selected")
        self.select_output_btn = QPushButton("Select Output Directory")
        self.select_output_btn.clicked.connect(self.select_output)
        output_file_layout.addWidget(self.select_output_btn)
        output_file_layout.addWidget(self.output_label)
        button_layout.addLayout(output_file_layout)
        
        button_group.setLayout(button_layout)
        layout.addWidget(button_group)

        # Interpolate button
        self.interpolate_btn = QPushButton("Interpolate")
        self.interpolate_btn.clicked.connect(self.interpolate_stack)
        layout.addWidget(self.interpolate_btn)

        self.setLayout(layout)

    def select_input(self):
        try:
            if self.dir_radio.isChecked():
                path = QFileDialog.getExistingDirectory(
                    self,
                    "Select Directory with Images",
                    "",
                    QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
                )
            else:
                path, _ = QFileDialog.getOpenFileName(
                    self,
                    "Select TIFF Stack",
                    "",
                    "TIFF Files (*.tif *.tiff)",
                    options=QFileDialog.Options()
                )
            
            if path:
                self.input_path = path
                self.input_label.setText(f"Selected: {os.path.basename(path)}")
                self.input_label.setToolTip(path)
                QApplication.processEvents()
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error selecting input: {str(e)}")

    def select_output(self):
        try:
            directory = QFileDialog.getExistingDirectory(
                self,
                "Select Output Directory",
                "",
                QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
            )
            
            if directory:
                self.output_directory = directory
                self.output_label.setText(f"Selected: {os.path.basename(directory)}")
                self.output_label.setToolTip(directory)
                QApplication.processEvents()
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error selecting output directory: {str(e)}")

    def load_images(self):
        try:
            progress = QProgressDialog("Loading images...", "Cancel", 0, 100, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.show()
            QApplication.processEvents()
    
            if self.stack_radio.isChecked():
                progress.setLabelText("Loading TIFF stack...")
                progress.setValue(20)
                QApplication.processEvents()
                
                # Load stack preserving original dtype
                stack = io.imread(self.input_path)
                print(f"Loaded stack dtype: {stack.dtype}")
                print(f"Value range: [{stack.min()}, {stack.max()}]")
                
                progress.setValue(90)
                QApplication.processEvents()
                return stack
            else:
                valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
                files = sorted([f for f in os.listdir(self.input_path) 
                              if f.lower().endswith(valid_extensions)])
                
                if not files:
                    raise ValueError("No valid image files found in directory")
    
                progress.setMaximum(len(files))
                
                # Load first image to get dimensions and dtype
                first_img = io.imread(os.path.join(self.input_path, files[0]))
                stack = np.zeros((len(files), *first_img.shape), dtype=first_img.dtype)
                stack[0] = first_img
                
                print(f"Created stack with dtype: {stack.dtype}")
                print(f"First image range: [{first_img.min()}, {first_img.max()}]")
    
                # Load remaining images
                for i, fname in enumerate(files[1:], 1):
                    progress.setValue(i)
                    progress.setLabelText(f"Loading image {i+1}/{len(files)}")
                    QApplication.processEvents()
                    
                    if progress.wasCanceled():
                        raise InterruptedError("Loading cancelled by user")
                    
                    img = io.imread(os.path.join(self.input_path, fname))
                    if img.shape != first_img.shape:
                        raise ValueError(f"Image {fname} has different dimensions from the first image")
                    if img.dtype != first_img.dtype:
                        raise ValueError(f"Image {fname} has different bit depth from the first image")
                    stack[i] = img
    
                return stack
                
        except Exception as e:
            raise ValueError(f"Error loading images: {str(e)}")
        finally:
            progress.close()
            QApplication.processEvents()
    
    def interpolate_stack(self):
        if not self.input_path or not self.output_directory:
            QMessageBox.warning(self, "Missing Paths", "Please select both input and output paths")
            return
            
        try:
            # Create progress dialog
            progress = QProgressDialog("Processing...", "Cancel", 0, 100, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.setWindowTitle("Interpolation Progress")
            progress.setMinimumDuration(0)
            progress.setMinimumWidth(400)
            progress.show()
            QApplication.processEvents()
    
            # Load images
            progress.setLabelText("Loading images...")
            progress.setValue(10)
            QApplication.processEvents()
            
            input_stack = self.load_images()
            original_dtype = input_stack.dtype
            type_range = np.iinfo(original_dtype) if np.issubdtype(original_dtype, np.integer) else None
            
            print(f"Original data type: {original_dtype}")
            print(f"Original shape: {input_stack.shape}")
            print(f"Original range: {input_stack.min()} - {input_stack.max()}")
            
            # Normalize input data to float64 for interpolation
            input_stack_normalized = input_stack.astype(np.float64)
            if type_range is not None:
                input_stack_normalized = input_stack_normalized / type_range.max
            
            progress.setLabelText("Calculating dimensions...")
            progress.setValue(20)
            QApplication.processEvents()
    
            # Calculate dimensions and coordinates
            z_old = np.arange(input_stack.shape[0]) * self.orig_z_size.value()
            y_old = np.arange(input_stack.shape[1]) * self.orig_xy_size.value()
            x_old = np.arange(input_stack.shape[2]) * self.orig_xy_size.value()
    
            z_new = np.arange(z_old[0], z_old[-1] + self.new_z_size.value(), self.new_z_size.value())
            y_new = np.arange(0, input_stack.shape[1] * self.orig_xy_size.value(), self.new_xy_size.value())
            x_new = np.arange(0, input_stack.shape[2] * self.orig_xy_size.value(), self.new_xy_size.value())
    
            y_new = y_new[y_new < y_old[-1] + self.new_xy_size.value()]
            x_new = x_new[x_new < x_old[-1] + self.new_xy_size.value()]
    
            new_shape = (len(z_new), len(y_new), len(x_new))
            print(f"New dimensions will be: {new_shape}")
    
            # Initialize output array
            interpolated_data = np.zeros(new_shape, dtype=np.float64)
            
            method = self.method_combo.currentText()
            
            # For higher-order methods, use a hybrid approach
            if method in ['cubic', 'quintic', 'pchip']:
                progress.setLabelText("Using hybrid interpolation approach...")
                progress.setValue(30)
                QApplication.processEvents()
                
                from scipy.interpolate import interp1d
                
                # Process each XY point
                total_points = input_stack.shape[1] * input_stack.shape[2]
                points_processed = 0
                
                temp_stack = np.zeros((len(z_new), input_stack.shape[1], input_stack.shape[2]), dtype=np.float64)
                
                for y in range(input_stack.shape[1]):
                    for x in range(input_stack.shape[2]):
                        if progress.wasCanceled():
                            return
                        
                        points_processed += 1
                        if points_processed % 1000 == 0:
                            progress_val = 30 + (points_processed / total_points * 30)
                            progress.setValue(int(progress_val))
                            progress.setLabelText(f"Interpolating Z dimension: {points_processed}/{total_points} points")
                            QApplication.processEvents()
                        
                        z_profile = input_stack_normalized[:, y, x]
                        f = interp1d(z_old, z_profile, kind=method, bounds_error=False, fill_value='extrapolate')
                        temp_stack[:, y, x] = f(z_new)
                
                progress.setLabelText("Interpolating XY planes...")
                progress.setValue(60)
                QApplication.processEvents()
                
                for z in range(len(z_new)):
                    if progress.wasCanceled():
                        return
                    
                    progress.setValue(60 + int((z / len(z_new)) * 30))
                    progress.setLabelText(f"Processing XY plane {z+1}/{len(z_new)}")
                    QApplication.processEvents()
                    
                    interpolator = RegularGridInterpolator(
                        (y_old, x_old),
                        temp_stack[z],
                        method='linear',
                        bounds_error=False,
                        fill_value=0
                    )
                    
                    yy, xx = np.meshgrid(y_new, x_new, indexing='ij')
                    pts = np.stack([yy.ravel(), xx.ravel()], axis=-1)
                    
                    interpolated_data[z] = interpolator(pts).reshape(len(y_new), len(x_new))
                
                del temp_stack
                
            else:  # For linear and nearest neighbor
                progress.setLabelText("Creating interpolator...")
                progress.setValue(30)
                QApplication.processEvents()
                
                interpolator = RegularGridInterpolator(
                    (z_old, y_old, x_old),
                    input_stack_normalized,
                    method=method,
                    bounds_error=False,
                    fill_value=0
                )
                
                slices_per_batch = max(1, len(z_new) // 20)
                total_batches = (len(z_new) + slices_per_batch - 1) // slices_per_batch
                
                for batch_idx in range(total_batches):
                    if progress.wasCanceled():
                        return
                    
                    start_idx = batch_idx * slices_per_batch
                    end_idx = min((batch_idx + 1) * slices_per_batch, len(z_new))
                    
                    progress.setLabelText(f"Interpolating batch {batch_idx + 1}/{total_batches}")
                    progress_value = int(40 + (batch_idx/total_batches)*40)
                    progress.setValue(progress_value)
                    QApplication.processEvents()
                    
                    zz, yy, xx = np.meshgrid(
                        z_new[start_idx:end_idx],
                        y_new,
                        x_new,
                        indexing='ij'
                    )
                    
                    pts = np.stack([zz.ravel(), yy.ravel(), xx.ravel()], axis=-1)
                    
                    interpolated_data[start_idx:end_idx] = interpolator(pts).reshape(
                        end_idx - start_idx,
                        len(y_new),
                        len(x_new)
                    )
    
            # Convert back to original dtype
            progress.setLabelText("Converting to original bit depth...")
            progress.setValue(90)
            QApplication.processEvents()
    
            if np.issubdtype(original_dtype, np.integer):
                # Scale back to original range
                interpolated_data = np.clip(interpolated_data, 0, 1)
                interpolated_data = (interpolated_data * type_range.max).astype(original_dtype)
            else:
                interpolated_data = interpolated_data.astype(original_dtype)
    
            print(f"Final dtype: {interpolated_data.dtype}")
            print(f"Final range: [{interpolated_data.min()}, {interpolated_data.max()}]")
    
            # Save output
            progress.setLabelText("Saving interpolated stack...")
            progress.setValue(95)
            QApplication.processEvents()
    
            if self.stack_radio.isChecked():
                output_name = os.path.splitext(os.path.basename(self.input_path))[0]
            else:
                output_name = "interpolated_stack"
    
            output_path = os.path.join(self.output_directory, f"{output_name}_interpolated.tif")
    
            # Convert sizes to micrometers for metadata
            unit = self.size_unit.currentText()
            xy_size = self.new_xy_size.value()
            z_size = self.new_z_size.value()
            
            if unit == "nm":
                xy_size /= 1000
                z_size /= 1000
            elif unit == "mm":
                xy_size *= 1000
                z_size *= 1000
    
            # Save with metadata
            tifffile.imwrite(
                output_path,
                interpolated_data,
                imagej=True,
                metadata={
                    'axes': 'ZYX',
                    'spacing': z_size,
                    'unit': 'um',
                    'finterval': xy_size
                },
                resolution=(1.0/xy_size, 1.0/xy_size)
            )
    
            progress.setValue(100)
            QApplication.processEvents()
    
            QMessageBox.information(
                self,
                "Success",
                f"Interpolation completed successfully!\n"
                f"Output saved to:\n{output_path}\n"
                f"New dimensions: {interpolated_data.shape}\n"
                f"Bit depth: {interpolated_data.dtype}\n"
                f"XY Pixel size: {self.new_xy_size.value()} {unit}\n"
                f"Z Spacing: {self.new_z_size.value()} {unit}"
            )
    
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            print(f"Error occurred: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            progress.close()
            QApplication.processEvents()


    def show_centered(self, parent):
        parent_geo = parent.geometry()
        self.move(parent_geo.center() - self.rect().center())
        self.show()
        QApplication.processEvents()  # Ensure UI updates
        
        
# Helper function to create the dialog
def show_stack_interpolator(parent):
    dialog = StackInterpolator(parent)
    dialog.show_centered(parent)
    return dialog