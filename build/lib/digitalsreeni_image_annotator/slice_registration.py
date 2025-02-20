from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, 
                            QLabel, QComboBox, QMessageBox, QProgressDialog, QRadioButton,
                            QButtonGroup, QSpinBox, QApplication, QGroupBox, QDoubleSpinBox)
from PyQt5.QtCore import Qt
from pystackreg import StackReg
from skimage import io
import tifffile
from PIL import Image
import numpy as np
import os

class SliceRegistrationTool(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Slice Registration")
        self.setGeometry(100, 100, 600, 400)
        self.setWindowFlags(self.windowFlags() | Qt.Window)
        self.setWindowModality(Qt.ApplicationModal)  # Add modal behavior
        
        # Initialize variables first
        self.input_path = ""
        self.output_directory = ""
        
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        layout.setSpacing(10)  # Add consistent spacing

        # Input selection
        input_group = QGroupBox("Input Selection")
        input_layout = QVBoxLayout()
        
        self.dir_radio = QRadioButton("Directory of Image Files")
        self.stack_radio = QRadioButton("TIFF Stack")
        
        input_group = QButtonGroup(self)
        input_group.addButton(self.dir_radio)
        input_group.addButton(self.stack_radio)
        
        input_layout.addWidget(self.dir_radio)
        input_layout.addWidget(self.stack_radio)
        self.dir_radio.setChecked(True)
        
        # Input/Output file selection with labels
        self.input_label = QLabel("No input selected")
        self.output_label = QLabel("No output directory selected")
        
        file_select_layout = QVBoxLayout()
        
        input_file_layout = QHBoxLayout()
        self.select_input_btn = QPushButton("Select Input")
        self.select_input_btn.clicked.connect(self.select_input)
        input_file_layout.addWidget(self.select_input_btn)
        input_file_layout.addWidget(self.input_label)
        
        output_file_layout = QHBoxLayout()
        self.select_output_btn = QPushButton("Select Output Directory")
        self.select_output_btn.clicked.connect(self.select_output)
        output_file_layout.addWidget(self.select_output_btn)
        output_file_layout.addWidget(self.output_label)
        
        file_select_layout.addLayout(input_file_layout)
        file_select_layout.addLayout(output_file_layout)
        input_layout.addLayout(file_select_layout)
        
        layout.addLayout(input_layout)

        # Transform type
        transform_group = QGroupBox("Transformation Settings")
        transform_layout = QVBoxLayout()
        
        transform_combo_layout = QHBoxLayout()
        transform_combo_layout.addWidget(QLabel("Type:"))
        self.transform_combo = QComboBox()
        self.transform_combo.addItems([
            "Translation (X-Y Translation Only)",
            "Rigid Body (Translation + Rotation)",
            "Scaled Rotation (Translation + Rotation + Scaling)",
            "Affine (Translation + Rotation + Scaling + Shearing)",
            "Bilinear (Non-linear; Does not preserve straight lines)"
        ])
        transform_combo_layout.addWidget(self.transform_combo)
        transform_layout.addLayout(transform_combo_layout)
        transform_group.setLayout(transform_layout)
        layout.addWidget(transform_group)

        # Reference type
        ref_group = QGroupBox("Reference Settings")
        ref_layout = QVBoxLayout()
        
        ref_combo_layout = QHBoxLayout()
        ref_combo_layout.addWidget(QLabel("Reference:"))
        self.ref_combo = QComboBox()
        self.ref_combo.addItems([
            "Previous Frame",
            "First Frame",
            "Mean of All Frames",
            "Mean of First N Frames",
            "Mean of First N Frames + Moving Average"
        ])
        ref_combo_layout.addWidget(self.ref_combo)
        ref_layout.addLayout(ref_combo_layout)
        
        # N frames settings
        n_frames_layout = QHBoxLayout()
        n_frames_layout.addWidget(QLabel("N Frames:"))
        self.n_frames_spin = QSpinBox()
        self.n_frames_spin.setRange(1, 100)
        self.n_frames_spin.setValue(10)
        self.n_frames_spin.setEnabled(False)
        n_frames_layout.addWidget(self.n_frames_spin)
        ref_layout.addLayout(n_frames_layout)
        
        # Moving average settings
        moving_avg_layout = QHBoxLayout()
        moving_avg_layout.addWidget(QLabel("Moving Average Window:"))
        self.moving_avg_spin = QSpinBox()
        self.moving_avg_spin.setRange(1, 100)
        self.moving_avg_spin.setValue(10)
        self.moving_avg_spin.setEnabled(False)
        moving_avg_layout.addWidget(self.moving_avg_spin)
        ref_layout.addLayout(moving_avg_layout)
        
        ref_group.setLayout(ref_layout)
        layout.addWidget(ref_group)

        # Connect reference combo box
        self.ref_combo.currentTextChanged.connect(self.on_ref_changed)

        # Add spacing group
        spacing_group = QGroupBox("Pixel/Voxel Size")
        spacing_layout = QVBoxLayout()
        
        # XY pixel size
        xy_size_layout = QHBoxLayout()
        xy_size_layout.addWidget(QLabel("XY Pixel Size:"))
        self.xy_size_value = QDoubleSpinBox()
        self.xy_size_value.setRange(0.001, 1000.0)
        self.xy_size_value.setValue(1.0)
        self.xy_size_value.setDecimals(3)
        xy_size_layout.addWidget(self.xy_size_value)
        
        # Z spacing
        z_size_layout = QHBoxLayout()
        z_size_layout.addWidget(QLabel("Z Spacing:"))
        self.z_size_value = QDoubleSpinBox()
        self.z_size_value.setRange(0.001, 1000.0)
        self.z_size_value.setValue(1.0)
        self.z_size_value.setDecimals(3)
        z_size_layout.addWidget(self.z_size_value)
        
        # Unit selector
        unit_layout = QHBoxLayout()
        unit_layout.addWidget(QLabel("Unit:"))
        self.size_unit = QComboBox()
        self.size_unit.addItems(["nm", "µm", "mm"])
        self.size_unit.setCurrentText("µm")
        unit_layout.addWidget(self.size_unit)
        
        spacing_layout.addLayout(xy_size_layout)
        spacing_layout.addLayout(z_size_layout)
        spacing_layout.addLayout(unit_layout)
        spacing_group.setLayout(spacing_layout)
        layout.addWidget(spacing_group)

        # Register button
        self.register_btn = QPushButton("Register")
        self.register_btn.clicked.connect(self.register_slices)
        layout.addWidget(self.register_btn)

        self.setLayout(layout)

    def on_ref_changed(self, text):
        uses_n_frames = text in ["Mean of First N Frames", "Mean of First N Frames + Moving Average"]
        self.n_frames_spin.setEnabled(uses_n_frames)
        self.moving_avg_spin.setEnabled(text == "Mean of First N Frames + Moving Average")
        QApplication.processEvents()  # Ensure UI updates
        

    def on_transform_changed(self, text):
        if text == "Bilinear" and self.ref_combo.currentText() == "Previous":
            QMessageBox.warning(self, "Warning", 
                "Bilinear transformation cannot be used with 'Previous' reference. "
                "Please select a different reference type.")
            self.transform_combo.setCurrentText("Rigid Body")


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

    def register_slices(self):
        if not self.input_path or not self.output_directory:
            QMessageBox.warning(self, "Error", "Please select both input and output paths")
            return
    
        try:
            progress = QProgressDialog(self)
            progress.setWindowTitle("Registration Progress")
            progress.setLabelText("Loading images...")
            progress.setMinimum(0)
            progress.setMaximum(100)
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumWidth(400)
            progress.show()
            QApplication.processEvents()
    
            # Load images using scikit-image's imread
            if self.stack_radio.isChecked():
                progress.setLabelText("Loading TIFF stack...")
                img0 = io.imread(self.input_path)
            else:
                progress.setLabelText("Loading images from directory...")
                image_files = sorted([f for f in os.listdir(self.input_path) 
                                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
                first_img = io.imread(os.path.join(self.input_path, image_files[0]))
                img0 = np.zeros((len(image_files), *first_img.shape), dtype=first_img.dtype)
                img0[0] = first_img
                for i, fname in enumerate(image_files[1:], 1):
                    img0[i] = io.imread(os.path.join(self.input_path, fname))
    
            # Store original properties
            original_dtype = img0.dtype
            print(f"Original image properties:")
            print(f"Dtype: {original_dtype}")
            print(f"Range: {img0.min()} - {img0.max()}")
            print(f"Shape: {img0.shape}")
    
            progress.setValue(30)
            progress.setLabelText("Performing registration...")
            QApplication.processEvents()
    
            # Set up StackReg with selected transformation
            transform_types = {
                "Translation (X-Y Translation Only)": StackReg.TRANSLATION,
                "Rigid Body (Translation + Rotation)": StackReg.RIGID_BODY,
                "Scaled Rotation (Translation + Rotation + Scaling)": StackReg.SCALED_ROTATION,
                "Affine (Translation + Rotation + Scaling + Shearing)": StackReg.AFFINE,
                "Bilinear (Non-linear; Does not preserve straight lines)": StackReg.BILINEAR
            }
            
            transform_type = transform_types[self.transform_combo.currentText()]
            sr = StackReg(transform_type)
    
            # Register images
            selected_ref = self.ref_combo.currentText()
            progress.setLabelText(f"Registering images using {selected_ref}...")
            progress.setValue(40)
            QApplication.processEvents()
    
            # Register and transform
            if selected_ref == "Previous Frame":
                out_registered = sr.register_transform_stack(img0, reference='previous')
            elif selected_ref == "First Frame":
                out_registered = sr.register_transform_stack(img0, reference='first')
            elif selected_ref == "Mean of All Frames":
                out_registered = sr.register_transform_stack(img0, reference='mean')
            elif selected_ref == "Mean of First N Frames":
                n_frames = self.n_frames_spin.value()
                out_registered = sr.register_transform_stack(img0, reference='first', n_frames=n_frames)
            elif selected_ref == "Mean of First N Frames + Moving Average":
                n_frames = self.n_frames_spin.value()
                moving_avg = self.moving_avg_spin.value()
                out_registered = sr.register_transform_stack(img0, reference='first', 
                                                           n_frames=n_frames, 
                                                           moving_average=moving_avg)
    
            progress.setValue(80)
            progress.setLabelText("Saving registered images...")
            QApplication.processEvents()
    
            # Convert back to original dtype without changing values
            out_registered = out_registered.astype(original_dtype)
    
            print(f"Output image properties:")
            print(f"Dtype: {out_registered.dtype}")
            print(f"Range: {out_registered.min()} - {out_registered.max()}")
            print(f"Shape: {out_registered.shape}")
    
            # Save output
            if self.stack_radio.isChecked():
                output_name = os.path.splitext(os.path.basename(self.input_path))[0]
            else:
                output_name = "registered_stack"
                
            output_path = os.path.join(self.output_directory, f"{output_name}_registered.tif")
    
            # Get pixel sizes in micrometers (convert if necessary)
            xy_size = self.xy_size_value.value()
            z_size = self.z_size_value.value()
            unit = self.size_unit.currentText()
            
            # Convert to micrometers based on selected unit
            if unit == "nm":
                xy_size = xy_size / 1000
                z_size = z_size / 1000
            elif unit == "mm":
                xy_size = xy_size * 1000
                z_size = z_size * 1000
    
            # Save the stack
            tifffile.imwrite(
                output_path, 
                out_registered,
                imagej=True,
                metadata={
                    'axes': 'ZYX',
                    'spacing': z_size,  # Z spacing in micrometers
                    'unit': 'um',
                    'finterval': xy_size  # XY pixel size in micrometers
                },
                resolution=(1.0/xy_size, 1.0/xy_size)  # XY Resolution in pixels per unit
            )
            
            progress.setValue(100)
            QApplication.processEvents()
            
            # Include both XY and Z size info in success message
            QMessageBox.information(self, "Success", 
                                  f"Registration completed successfully!\n"
                                  f"Output saved to:\n{output_path}\n"
                                  f"XY Pixel size: {self.xy_size_value.value()} {unit}\n"
                                  f"Z Spacing: {self.z_size_value.value()} {unit}")
    
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", str(e))
            
            
    def update_progress(self, progress_dialog, current_iteration, end_iteration):
        """Helper function to update progress during registration"""
        if end_iteration > 0:
            percent = int(40 + (current_iteration / end_iteration) * 40)  # Scale to 40-80% range
            progress_dialog.setValue(percent)
            progress_dialog.setLabelText(f"Processing image {current_iteration}/{end_iteration}...")
            QApplication.processEvents()
        

    def load_images(self):
        print("Starting image loading...")
        try:
            if self.stack_radio.isChecked():
                print(f"Loading TIFF stack from: {self.input_path}")
                # Explicitly use scikit-image's imread for TIFF stacks
                stack = io.imread(self.input_path)
                if stack.dtype != np.float32:
                    stack = stack.astype(np.float32)
                print(f"Loaded TIFF stack shape: {stack.shape}")
                return stack
            else:
                # Load individual images
                valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
                images = []
                files = sorted([f for f in os.listdir(self.input_path) 
                              if f.lower().endswith(valid_extensions)])
                
                print(f"Found {len(files)} image files")
                
                if not files:
                    raise ValueError("No valid image files found in directory")
    
                # Check first image size
                first_path = os.path.join(self.input_path, files[0])
                print(f"Loading first image: {first_path}")
                first_img = np.array(Image.open(first_path))
                ref_shape = first_img.shape
                images.append(first_img)
                print(f"First image shape: {ref_shape}")
    
                # Load remaining images and check sizes
                for f in files[1:]:
                    img_path = os.path.join(self.input_path, f)
                    print(f"Loading: {f}")
                    img = np.array(Image.open(img_path))
                    if img.shape != ref_shape:
                        raise ValueError(f"Image {f} has different dimensions from the first image")
                    images.append(img)
    
                stack = np.stack(images)
                print(f"Final stack shape: {stack.shape}")
                return stack
                
        except Exception as e:
            print(f"Error in load_images: {str(e)}")
            raise


    def show_centered(self, parent):
        parent_geo = parent.geometry()
        self.move(parent_geo.center() - self.rect().center())
        self.show()
        QApplication.processEvents()  # Ensure window displays properly

