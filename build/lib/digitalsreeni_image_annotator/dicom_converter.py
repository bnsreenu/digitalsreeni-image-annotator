import os
import json
import numpy as np
from datetime import datetime
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, 
                            QLabel, QProgressDialog, QRadioButton, QButtonGroup, 
                            QMessageBox, QApplication, QGroupBox)
from PyQt5.QtCore import Qt
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import tifffile

class DicomConverter(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("DICOM to TIFF Converter")
        self.setGeometry(100, 100, 600, 300)
        self.setWindowFlags(self.windowFlags() | Qt.Window)
        self.setWindowModality(Qt.ApplicationModal)  # Add modal behavior
        
        # Initialize variables first
        self.input_file = ""
        self.output_directory = ""
        
        self.initUI()
        
    def initUI(self):
        layout = QVBoxLayout()
        layout.setSpacing(10)  # Add consistent spacing

        # File Selection Group
        file_group = QGroupBox("File Selection")
        file_layout = QVBoxLayout()

        # Input file selection
        input_layout = QHBoxLayout()
        self.input_label = QLabel("No DICOM file selected")
        self.input_label.setMinimumWidth(100)
        self.input_label.setMaximumWidth(300)
        self.input_label.setWordWrap(True)
        self.select_input_btn = QPushButton("Select DICOM File")
        self.select_input_btn.clicked.connect(self.select_input)
        input_layout.addWidget(self.select_input_btn)
        input_layout.addWidget(self.input_label, 1)
        file_layout.addLayout(input_layout)

        # Output directory selection
        output_layout = QHBoxLayout()
        self.output_label = QLabel("No output directory selected")
        self.output_label.setMinimumWidth(100)
        self.output_label.setMaximumWidth(300)
        self.output_label.setWordWrap(True)
        self.select_output_btn = QPushButton("Select Output Directory")
        self.select_output_btn.clicked.connect(self.select_output)
        output_layout.addWidget(self.select_output_btn)
        output_layout.addWidget(self.output_label, 1)
        file_layout.addLayout(output_layout)

        file_group.setLayout(file_layout)
        layout.addWidget(file_group)

        # Output Format Group
        format_group = QGroupBox("Output Format")
        format_layout = QVBoxLayout()
        
        self.stack_radio = QRadioButton("Single TIFF Stack")
        self.individual_radio = QRadioButton("Individual TIFF Files")
        self.stack_radio.setChecked(True)
        
        format_layout.addWidget(self.stack_radio)
        format_layout.addWidget(self.individual_radio)
        format_group.setLayout(format_layout)
        layout.addWidget(format_group)

        # Metadata info
        metadata_group = QGroupBox("Metadata Information")
        metadata_layout = QVBoxLayout()
        metadata_label = QLabel("DICOM metadata will be saved as JSON file in the output directory")
        metadata_label.setStyleSheet("color: gray; font-style: italic;")
        metadata_label.setWordWrap(True)
        metadata_layout.addWidget(metadata_label)
        metadata_group.setLayout(metadata_layout)
        layout.addWidget(metadata_group)

        # Convert button
        self.convert_btn = QPushButton("Convert")
        self.convert_btn.clicked.connect(self.convert_dicom)
        layout.addWidget(self.convert_btn)

        self.setLayout(layout)
        
    def select_input(self):
        try:
            file_filter = "DICOM files (*.dcm *.DCM);;All files (*.*)"
            file_name, _ = QFileDialog.getOpenFileName(
                self,
                "Select DICOM File",
                "",
                file_filter,
                options=QFileDialog.Options()
            )
            
            if file_name:
                self.input_file = file_name
                self.input_label.setText(self.truncate_path(file_name))
                self.input_label.setToolTip(file_name)
                QApplication.processEvents()
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error selecting input file: {str(e)}")
        
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
                self.output_label.setText(self.truncate_path(directory))
                self.output_label.setToolTip(directory)
                QApplication.processEvents()
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error selecting output directory: {str(e)}")
            
    def truncate_path(self, path, max_length=40):
        if len(path) <= max_length:
            return path
        
        filename = os.path.basename(path)
        directory = os.path.dirname(path)
        
        if len(filename) > max_length - 5:
            return f"...{filename[-(max_length-5):]}"
        
        available_length = max_length - len(filename) - 5
        return f"...{directory[-available_length:]}{os.sep}{filename}"

    def extract_metadata(self, ds):
        """Extract relevant metadata from DICOM dataset."""
        metadata = {
            "PatientID": getattr(ds, "PatientID", "Unknown"),
            "PatientName": str(getattr(ds, "PatientName", "Unknown")),
            "StudyDate": getattr(ds, "StudyDate", "Unknown"),
            "SeriesDescription": getattr(ds, "SeriesDescription", "Unknown"),
            "Modality": getattr(ds, "Modality", "Unknown"),
            "Manufacturer": getattr(ds, "Manufacturer", "Unknown"),
            "InstitutionName": getattr(ds, "InstitutionName", "Unknown"),
            "PixelSpacing": getattr(ds, "PixelSpacing", [1, 1]),
            "SliceThickness": getattr(ds, "SliceThickness", 1),
            "ImageOrientation": getattr(ds, "ImageOrientationPatient", [1,0,0,0,1,0]),
            "ImagePosition": getattr(ds, "ImagePositionPatient", [0,0,0]),
            "WindowCenter": getattr(ds, "WindowCenter", None),
            "WindowWidth": getattr(ds, "WindowWidth", None),
            "RescaleIntercept": getattr(ds, "RescaleIntercept", 0),
            "RescaleSlope": getattr(ds, "RescaleSlope", 1),
            "BitsAllocated": getattr(ds, "BitsAllocated", 16),
            "PixelRepresentation": getattr(ds, "PixelRepresentation", 0),
            "ConversionDate": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        return metadata

    def apply_window_level(self, image, ds):
        """Apply window/level if present in DICOM."""
        try:
            if hasattr(ds, 'WindowCenter') and hasattr(ds, 'WindowWidth'):
                return apply_voi_lut(image, ds)
        except:
            pass
        return image


    def convert_dicom(self):
        if not self.input_file or not self.output_directory:
            QMessageBox.warning(self, "Error", "Please select both input file and output directory")
            return
            
        try:
            # Create progress dialog
            progress = QProgressDialog("Processing DICOM file...", "Cancel", 0, 100, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumWidth(400)
            progress.show()
            
            # Verify DICOM file
            if not pydicom.misc.is_dicom(self.input_file):
                raise ValueError("Selected file is not a valid DICOM file")
            
            # Read DICOM data
            print("Reading DICOM file...")
            progress.setLabelText("Reading DICOM file...")
            progress.setValue(20)
            
            ds = pydicom.dcmread(self.input_file)
            series_metadata = self.extract_metadata(ds)
            
            # Process pixel data
            print("Processing pixel data...")
            progress.setLabelText("Processing pixel data...")
            progress.setValue(40)
            
            pixel_array = ds.pixel_array
            original_dtype = pixel_array.dtype
            print(f"Original data type: {original_dtype}")
            print(f"Original data range: {pixel_array.min()} to {pixel_array.max()}")
            
            # Apply rescale slope and intercept
            if hasattr(ds, 'RescaleSlope') or hasattr(ds, 'RescaleIntercept'):
                slope = getattr(ds, 'RescaleSlope', 1)
                intercept = getattr(ds, 'RescaleIntercept', 0)
                print(f"Applying rescale slope ({slope}) and intercept ({intercept})")
                pixel_array = (pixel_array * slope + intercept)
            
            # Apply window/level
            print("Applying window/level adjustments...")
            pixel_array = self.apply_window_level(pixel_array, ds)
            print(f"Adjusted data range: {pixel_array.min()} to {pixel_array.max()}")
            
            print(f"Image shape: {pixel_array.shape}")
            print(f"Original dtype: {original_dtype}")
            
            # Save metadata
            progress.setLabelText("Saving metadata...")
            progress.setValue(60)
            
            metadata_file = os.path.join(self.output_directory, 
                                       os.path.splitext(os.path.basename(self.input_file))[0] + 
                                       "_metadata.json")
            with open(metadata_file, 'w') as f:
                json.dump(series_metadata, f, indent=2)
            
            # Get physical sizes from metadata
            pixel_spacing = series_metadata.get("PixelSpacing", [1, 1])
            slice_thickness = series_metadata.get("SliceThickness", 1)
            
            print(f"Pixel spacing: {pixel_spacing}")
            print(f"Slice thickness: {slice_thickness}")
            
            # Save TIFF
            progress.setLabelText("Saving TIFF file(s)...")
            progress.setValue(80)
            
            # Convert back to original dtype if needed
            if np.issubdtype(original_dtype, np.integer):
                print("Converting back to original integer dtype...")
                data_min = pixel_array.min()
                data_max = pixel_array.max()
                
                if data_max != data_min:
                    pixel_array = ((pixel_array - data_min) / (data_max - data_min) * 
                                 np.iinfo(original_dtype).max).astype(original_dtype)
                else:
                    pixel_array = np.zeros_like(pixel_array, dtype=original_dtype)
                
                print(f"Final data range: {pixel_array.min()} to {pixel_array.max()}")
            
            # Prepare ImageJ metadata
            imagej_metadata = {
                'axes': 'YX',  # Will be updated to ZYX for 3D data
                'spacing': float(slice_thickness),  # Only used for 3D data
                'unit': 'um',
                'finterval': float(pixel_spacing[0])  # XY pixel size
            }
            
            base_name = os.path.splitext(os.path.basename(self.input_file))[0]
            
            if self.stack_radio.isChecked():
                # Save as single stack
                output_file = os.path.join(self.output_directory, f"{base_name}.tif")
                
                # Update axes for 3D data
                if len(pixel_array.shape) > 2:
                    imagej_metadata['axes'] = 'ZYX'
                
                print(f"Saving stack with metadata: {imagej_metadata}")
                
                tifffile.imwrite(
                    output_file,
                    pixel_array,
                    imagej=True,
                    metadata=imagej_metadata,
                    resolution=(1.0/float(pixel_spacing[0]), 1.0/float(pixel_spacing[1]))
                )
                
                print(f"Saved stack to: {output_file}")
                print(f"Stack shape: {pixel_array.shape}")
                
            # Replace the individual slices saving section in convert_dicom method with this:
            else:
                # For multi-slice DICOM, save individual slices
                if len(pixel_array.shape) > 2:
                    imagej_metadata['axes'] = 'YX'  # Reset to 2D for individual slices
                    
                    total_slices = pixel_array.shape[0]
                    for i in range(total_slices):
                        progress.setLabelText(f"Saving slice {i+1}/{total_slices}...")
                        # Fix: Convert float to integer for progress value
                        progress_value = int(80 + (i/total_slices)*15)
                        progress.setValue(progress_value)
                        QApplication.processEvents()
                        
                        if progress.wasCanceled():
                            print("Operation cancelled by user")
                            return
                        
                        output_file = os.path.join(self.output_directory, 
                                                 f"{base_name}_slice_{i+1:03d}.tif")
                        
                        print(f"Saving slice {i+1} with metadata: {imagej_metadata}")
                        
                        tifffile.imwrite(
                            output_file,
                            pixel_array[i],
                            imagej=True,
                            metadata=imagej_metadata,
                            resolution=(1.0/float(pixel_spacing[0]), 1.0/float(pixel_spacing[1]))
                        )
                        
                    print(f"Saved {total_slices} individual slices")
                    
                else:
                    # Single slice DICOM
                    output_file = os.path.join(self.output_directory, f"{base_name}.tif")
                    
                    print(f"Saving single slice with metadata: {imagej_metadata}")
                    
                    tifffile.imwrite(
                        output_file,
                        pixel_array,
                        imagej=True,
                        metadata=imagej_metadata,
                        resolution=(1.0/float(pixel_spacing[0]), 1.0/float(pixel_spacing[1]))
                    )
                    
                    print(f"Saved single slice to: {output_file}")
            
            progress.setValue(100)
            
            # Construct success message
            msg = "Conversion complete!\n\n"
            msg += f"DICOM file: {os.path.basename(self.input_file)}\n"
            msg += f"Output directory: {self.truncate_path(self.output_directory)}\n\n"
            
            if self.stack_radio.isChecked():
                msg += f"Saved as: {os.path.basename(output_file)}\n"
            else:
                if len(pixel_array.shape) > 2:
                    msg += f"Saved {pixel_array.shape[0]} individual slices\n"
                else:
                    msg += f"Saved as: {os.path.basename(output_file)}\n"
            
            msg += f"\nMetadata saved as: {os.path.basename(metadata_file)}\n"
            msg += f"Pixel spacing: {pixel_spacing[0]}x{pixel_spacing[1]} µm\n"
            if len(pixel_array.shape) > 2:
                msg += f"Slice thickness: {slice_thickness} µm"
            
            QMessageBox.information(self, "Success", msg)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            print(f"Error occurred: {str(e)}")
            import traceback
            traceback.print_exc()


    def show_centered(self, parent):
        parent_geo = parent.geometry()
        self.move(parent_geo.center() - self.rect().center())
        self.show()
        QApplication.processEvents()  # Ensure window displays properly
        
def show_dicom_converter(parent):
    dialog = DicomConverter(parent)
    dialog.show_centered(parent)
    return dialog
