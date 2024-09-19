import os
import json
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QFileDialog, QListWidget, QInputDialog, 
                             QLabel, QButtonGroup, QListWidgetItem, QScrollArea, 
                             QSlider, QMenu, QMessageBox, QColorDialog, QDialog,
                             QGridLayout, QComboBox, QAbstractItemView)
from PyQt5.QtGui import QPixmap, QColor, QIcon, QImage, QPalette, QFont
from PyQt5.QtCore import Qt, QSize
import numpy as np
from tifffile import TiffFile
from czifile import CziFile
from PIL import Image
import tifffile


from .image_label import ImageLabel
from .utils import calculate_area, calculate_bbox
from .help_window import HelpWindow

from PyQt5.QtWidgets import QStyleFactory
from .soft_dark_stylesheet import soft_dark_stylesheet
from .default_stylesheet import default_stylesheet
import mmap  #optimizing working with large tiff files.

import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


class DimensionDialog(QDialog):
    def __init__(self, shape, file_name, parent=None, default_dimensions=None):
        super().__init__(parent)
        self.setWindowTitle("Assign Dimensions")
        layout = QVBoxLayout(self)
        
        # Add file name label
        file_name_label = QLabel(f"File: {file_name}")
        file_name_label.setWordWrap(True)
        layout.addWidget(file_name_label)
        
        # Add dimension assignment widgets
        dim_widget = QWidget()
        dim_layout = QGridLayout(dim_widget)
        self.combos = []
        self.shape = shape
        dimensions = ['T', 'Z', 'C', 'S', 'H', 'W']
        for i, dim in enumerate(shape):
            dim_layout.addWidget(QLabel(f"Dimension {i} (size {dim}):"), i, 0)
            combo = QComboBox()
            combo.addItems(dimensions)
            if default_dimensions and i < len(default_dimensions):
                combo.setCurrentText(default_dimensions[i])
            dim_layout.addWidget(combo, i, 1)
            self.combos.append(combo)
        layout.addWidget(dim_widget)
        
        self.button = QPushButton("OK")
        self.button.clicked.connect(self.accept)
        layout.addWidget(self.button)
        
        self.setMinimumWidth(300)

    def get_dimensions(self):
        return [combo.currentText() for combo in self.combos]

class ImageAnnotator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Annotator")
        self.setGeometry(100, 100, 1400, 800)
    
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QHBoxLayout(self.central_widget)
        
        # Initialize image_label early
        self.image_label = ImageLabel()
        self.image_label.set_main_window(self)
    
        # Initialize attributes
        self.current_image = None
        self.current_class = None
        self.image_file_name = ""
        self.all_annotations = {}
        self.all_images = []
        self.image_paths = {}
        self.loaded_json = None
        self.class_mapping = {}
        self.editing_mode = False
        self.current_slice = None
        self.slices = []
        self.current_stack = None
        self.image_dimensions = {}
        self.image_slices = {}
        # SAM2
        self.sam2_model = None
        self.sam2_predictor = None
        self.initialize_sam2()
        
        # Font size control
        self.font_sizes = {"Small": 8, "Medium": 10, "Large": 12}
        self.current_font_size = "Medium"
    
        # Dark mode control
        self.dark_mode = False
    
        # Setup UI components
        self.setup_ui()
        
        # Apply theme and font (this includes stylesheet and font size application)
        self.apply_theme_and_font()
    
    def setup_ui(self):
        self.setup_sidebar()
        self.setup_image_area()
        self.setup_image_list()
        self.setup_slice_list()
        self.update_ui_for_current_tool()  
        
    def initialize_sam2(self):
        # Get the directory of the current script
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Construct absolute paths to the model files
        sam2_checkpoint = os.path.join(current_dir, "sam2_models", "sam2_hiera_small.pt")
        model_cfg_path = os.path.join(current_dir, "sam2_models", "sam2_hiera_s.yaml")
        
        # Ensure the files exist
        if not os.path.exists(sam2_checkpoint) or not os.path.exists(model_cfg_path):
            QMessageBox.warning(self, "SAM2 Model Error", "SAM2 model files not found. Some features may not work.")
            return

        device = torch.device("cpu")
        
        try:
            # Build the SAM2 model using the config file path
            self.sam2_model = build_sam2(model_cfg_path, sam2_checkpoint, device=device)
            self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)
        except Exception as e:
            QMessageBox.warning(self, "SAM2 Model Error", f"Failed to initialize SAM2 model: {str(e)}")
            self.sam2_model = None
            self.sam2_predictor = None
        
    def toggle_sam_magic_wand(self):
        if self.sam_magic_wand_button.isChecked():
            if self.current_class is None:
                QMessageBox.warning(self, "No Class Selected", "Please select a class before using SAM2 Magic Wand.")
                self.sam_magic_wand_button.setChecked(False)
                return
            self.image_label.setCursor(Qt.CrossCursor)
            self.image_label.sam_magic_wand_active = True
        else:
            self.image_label.setCursor(Qt.ArrowCursor)
            self.image_label.sam_magic_wand_active = False
            self.image_label.sam_bbox = None
        
        self.image_label.clear_temp_sam_prediction()  # Clear temporary prediction
    
    def generate_sam2_prediction(self, bbox):
        if self.current_image is None or self.sam2_predictor is None:
            print("Current image or SAM2 predictor is None")
            return None, 0
    
        print(f"Current image format: {self.current_image.format()}")
        image = self.qimage_to_numpy(self.current_image)
        print(f"Numpy image shape: {image.shape}, dtype: {image.dtype}")
    
        # Ensure the image is RGB
        if image.ndim == 2:
            image = np.stack((image,) * 3, axis=-1)
        elif image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
            
        print(f"Numpy image shape after RGB conversion: {image.shape}, dtype: {image.dtype}")
    
        self.sam2_predictor.set_image(image)
    
        masks, scores, _ = self.sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=np.array([bbox]),
            multimask_output=True
        )
        
        print(f"SAM2 prediction generated {len(masks)} masks")
        
        if len(masks) > 0:
            best_mask_index = np.argmax(scores)
            return masks[best_mask_index], scores[best_mask_index]
        else:
            print("No masks generated by SAM2")
            return None, 0
    
    def qimage_to_numpy(self, qimage):
        width = qimage.width()
        height = qimage.height()
        fmt = qimage.format()
    
        if fmt == QImage.Format_Grayscale8:
            # Handle grayscale format
            buffer = qimage.constBits().asarray(height * width)
            image = np.frombuffer(buffer, dtype=np.uint8).reshape((height, width))
            # Convert grayscale to RGB
            return np.stack((image,) * 3, axis=-1)
        
        elif fmt in [QImage.Format_RGB32, QImage.Format_ARGB32, QImage.Format_ARGB32_Premultiplied]:
            # Handle RGB32 and ARGB32 formats
            buffer = qimage.constBits().asarray(height * width * 4)
            image = np.frombuffer(buffer, dtype=np.uint8).reshape((height, width, 4))
            return image[:, :, :3]  # Return only RGB channels
        
        elif fmt == QImage.Format_RGB888:
            # Handle RGB888 format (no alpha channel, 3 bytes per pixel)
            buffer = qimage.constBits().asarray(height * width * 3)
            image = np.frombuffer(buffer, dtype=np.uint8).reshape((height, width, 3))
            return image  # Already RGB
        
        elif fmt == QImage.Format_Indexed8:
            # Handle Indexed8 format (palette-based)
            buffer = qimage.constBits().asarray(height * width)
            image = np.frombuffer(buffer, dtype=np.uint8).reshape((height, width))
            # Convert palette-based to RGB using the color table
            color_table = qimage.colorTable()
            rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
            for y in range(height):
                for x in range(width):
                    rgb_image[y, x] = QColor(color_table[image[y, x]]).getRgb()[:3]
            return rgb_image
    
        elif fmt == QImage.Format_RGB16:
            # Handle RGB16 format (16-bit RGB)
            buffer = qimage.constBits().asarray(height * width * 2)
            image = np.frombuffer(buffer, dtype=np.uint16).reshape((height, width))
            # Convert 16-bit to 8-bit
            image = (image / 256).astype(np.uint8)
            # Convert to RGB
            return np.stack((image,) * 3, axis=-1)
        
        else:
            # For any other format, convert to RGB32 first
            converted_image = qimage.convertToFormat(QImage.Format_RGB32)
            buffer = converted_image.constBits().asarray(height * width * 4)
            image = np.frombuffer(buffer, dtype=np.uint8).reshape((height, width, 4))
            return image[:, :, :3]  # Return only RGB channels

        
    def apply_sam2_prediction(self):
        if self.image_label.sam_bbox is None:
            print("SAM bbox is None")
            return
    
        x1, y1, x2, y2 = self.image_label.sam_bbox
        bbox = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
        print(f"Applying SAM2 prediction with bbox: {bbox}")
        
        try:
            mask, score = self.generate_sam2_prediction(bbox)
            print(f"SAM2 prediction generated with score: {score}")
            
            if mask is not None:
                print(f"Mask shape: {mask.shape}, Mask sum: {mask.sum()}")
                contours = self.mask_to_polygon(mask)
                print(f"Contours generated: {len(contours)} contour(s)")
                
                if not contours:
                    print("No valid contours found")
                    return
                
                temp_annotation = {
                    "segmentation": contours[0],  # Take the first contour
                    "category_id": self.class_mapping[self.current_class],
                    "category_name": self.current_class,
                    "score": score
                }
                print(f"Temporary annotation: {temp_annotation}")
                
                self.image_label.temp_sam_prediction = temp_annotation
                self.image_label.update()
            else:
                print("Failed to generate mask")
        except Exception as e:
            print(f"Error in applying SAM2 prediction: {str(e)}")
            import traceback
            traceback.print_exc()
    
        # Reset SAM bounding box
        self.image_label.sam_bbox = None
        self.image_label.update()
    
    def mask_to_polygon(self, mask):
        import cv2
        contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygons = []
        for contour in contours:
            if cv2.contourArea(contour) > 10:  # Filter out very small contours
                polygon = contour.flatten().tolist()
                if len(polygon) >= 6:  # Ensure the polygon has at least 3 points
                    polygons.append(polygon)
        print(f"Generated {len(polygons)} valid polygons")
        return polygons
    
    def accept_sam_prediction(self):
        if self.image_label.temp_sam_prediction:
            new_annotation = self.image_label.temp_sam_prediction
            self.image_label.annotations.setdefault(new_annotation["category_name"], []).append(new_annotation)
            self.add_annotation_to_list(new_annotation)
            self.save_current_annotations()
            self.update_slice_list_colors()
            self.image_label.temp_sam_prediction = None
            self.image_label.update()
            print("SAM prediction accepted and added to annotations.")
    
    def setup_slice_list(self):
        self.slice_list = QListWidget()
        self.slice_list.itemClicked.connect(self.switch_slice)
        self.image_list_layout.addWidget(QLabel("Slices:"))
        self.image_list_layout.addWidget(self.slice_list)

    def open_images(self):
        file_names, _ = QFileDialog.getOpenFileNames(self, "Open Images", "", "Image Files (*.png *.jpg *.bmp *.tif *.tiff *.czi)")
        if file_names:
            self.image_list.clear()
            self.image_paths.clear()
            self.all_images.clear()
            self.slice_list.clear()
            self.slices.clear()
            self.current_stack = None
            self.current_slice = None
            self.add_images_to_list(file_names)
            
            
    def convert_to_8bit_rgb(self, image_array):
        if image_array.ndim == 2:
            # Grayscale image
            image_8bit = self.normalize_array(image_array)
            return np.stack((image_8bit,) * 3, axis=-1)
        elif image_array.ndim == 3:
            if image_array.shape[2] == 3:
                # Already RGB, just normalize
                return self.normalize_array(image_array)
            elif image_array.shape[2] > 3:
                # Multi-channel image, use first three channels
                rgb_array = image_array[:, :, :3]
                return self.normalize_array(rgb_array)
        
        raise ValueError(f"Unsupported image shape: {image_array.shape}")
            
            

    def add_images_to_list(self, file_names):
        for file_name in file_names:
            base_name = os.path.basename(file_name)
            if base_name not in self.image_paths:
                image_info = {
                    "file_name": base_name,
                    "height": 0,
                    "width": 0,
                    "id": len(self.all_images) + 1
                }
                self.all_images.append(image_info)
                self.image_list.addItem(base_name)
                self.image_paths[base_name] = file_name

        if not self.current_image and self.all_images:
            self.switch_image(self.image_list.item(0))
            
            
    def switch_slice(self, item):
        if item is None:
            return
    
        self.save_current_annotations()
        self.image_label.clear_temp_sam_prediction()
    
        slice_name = item.text()
        for name, qimage in self.slices:
            if name == slice_name:
                self.current_image = qimage
                self.current_slice = name
                self.display_image()
                self.load_image_annotations()
                self.update_annotation_list()
                self.clear_highlighted_annotation()
                self.image_label.reset_annotation_state()
                self.image_label.clear_current_annotation()
                self.update_image_info()
                break
    
        # Ensure the UI is updated
        self.image_label.update()
        self.update_slice_list_colors()

    def switch_image(self, item):
        if item is None:
            return
    
        #print(f"Switching image to: {item.text()}")
        self.save_current_annotations()
        self.image_label.clear_temp_sam_prediction()
    
        file_name = item.text()
        image_info = next((img for img in self.all_images if img["file_name"] == file_name), None)
        
        if image_info:
            image_path = self.image_paths.get(file_name)
            if image_path and os.path.exists(image_path):
                self.image_file_name = file_name
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                
                if base_name in self.image_slices:
                    #print(f"Image has slices: {len(self.image_slices[base_name])}")
                    self.slices = self.image_slices[base_name]
                    self.update_slice_list()
                    
                    if self.slices:
                        first_slice = self.slices[0][0]
                        #print(f"Activating first slice: {first_slice}")
                        self.current_slice = first_slice
                        self.current_image = self.slices[0][1]
                        self.activate_slice(first_slice)
                    else:
                        print("No slices found")
                        self.slices = []
                        self.slice_list.clear()
                        self.current_slice = None
                        self.current_image = None
                else:
                    print("Loading single image")
                    self.current_slice = None
                    self.slices = []
                    self.slice_list.clear()
                    self.load_image(image_path)
                
                self.display_image()
                self.load_image_annotations()
                self.update_annotation_list()
                self.clear_highlighted_annotation()
                self.image_label.reset_annotation_state()
                self.image_label.clear_current_annotation()
                self.update_image_info()
                self.update_slice_list_colors()
            else:
                print(f"Image path not found: {image_path}")
                self.current_image = None
                self.current_slice = None
                self.image_label.clear()
                self.update_image_info()
        else:
            print(f"Image info not found for: {file_name}")
            self.current_image = None
            self.current_slice = None
            self.image_label.clear()
            self.update_image_info()
    
        #print(f"After switch_image - Current slice: {self.current_slice}")
        #print(f"After switch_image - Current image_file_name: {self.image_file_name}")
            
            
    def activate_current_slice(self):
        if self.current_slice:
            # Ensure the current slice is selected in the slice list
            items = self.slice_list.findItems(self.current_slice, Qt.MatchExactly)
            if items:
                self.slice_list.setCurrentItem(items[0])
            
            # Load annotations for the current slice
            self.load_image_annotations()
            
            # Update the image label
            self.image_label.update()
            
            # Update the annotation list
            self.update_annotation_list()

    def load_image(self, image_path):
        extension = os.path.splitext(image_path)[1].lower()
        if extension in ['.tif', '.tiff']:
            self.load_tiff(image_path)
        elif extension == '.czi':
            self.load_czi(image_path)
        else:
            self.load_regular_image(image_path)



    def load_tiff(self, image_path):
        print(f"Loading TIFF file: {image_path}")
        with TiffFile(image_path) as tif:
            print(f"TIFF tags: {tif.pages[0].tags}")
            
            # Try to access metadata if available
            try:
                metadata = tif.pages[0].tags['ImageDescription'].value
                print(f"TIFF metadata: {metadata}")
            except KeyError:
                print("No ImageDescription metadata found")
            
            # Check if it's a multi-page TIFF
            if len(tif.pages) > 1:
                print(f"Multi-page TIFF detected. Number of pages: {len(tif.pages)}")
                # Read all pages into a 3D array
                image_array = tif.asarray()
            else:
                print("Single-page TIFF detected.")
                image_array = tif.pages[0].asarray()
            
            print(f"Image array shape: {image_array.shape}")
            print(f"Image array dtype: {image_array.dtype}")
            print(f"Image min: {image_array.min()}, max: {image_array.max()}")
    
            # Print additional information about the TIFF structure
            print(f"TIFF structure: {tif.series}")
            for i, page in enumerate(tif.pages):
                print(f"Page {i}: shape={page.shape}, dtype={page.dtype}")
    
        self.process_multidimensional_image(image_array, image_path)



    def load_czi(self, image_path):
        print(f"Loading CZI file: {image_path}")
        with CziFile(image_path) as czi:
            image_array = czi.asarray()
            print(f"CZI array shape: {image_array.shape}")
            print(f"CZI array dtype: {image_array.dtype}")
            print(f"CZI array min: {image_array.min()}, max: {image_array.max()}")
            
            # Print information about each channel
            if len(image_array.shape) > 2:
                for c in range(image_array.shape[-3]):  # Assuming channel is the third-to-last dimension
                    channel = image_array[..., c, :, :]
                    print(f"Channel {c} - min: {channel.min()}, max: {channel.max()}, mean: {channel.mean()}")
        
        self.process_multidimensional_image(image_array, image_path)
    
    
    def load_regular_image(self, image_path):
        self.current_image = QImage(image_path)
        self.slices = []
        self.slice_list.clear()
        self.current_slice = None
    
    def process_multidimensional_image(self, image_array, image_path):
        file_name = os.path.basename(image_path)
        print(f"Processing file: {file_name}")
        print(f"Image array shape: {image_array.shape}")
        print(f"Image array dtype: {image_array.dtype}")
    
        if file_name not in self.image_dimensions:
            if image_array.ndim > 2:
                default_dimensions = ['Z', 'H', 'W'] if image_array.ndim == 3 else ['T', 'Z', 'H', 'W']
                default_dimensions = default_dimensions[-image_array.ndim:]
                while True:
                    dialog = DimensionDialog(image_array.shape, file_name, self, default_dimensions)
                    dialog.setWindowFlags(dialog.windowFlags() & ~Qt.WindowContextHelpButtonHint)
                    if dialog.exec_():
                        dimensions = dialog.get_dimensions()
                        print(f"Assigned dimensions: {dimensions}")
                        if 'H' in dimensions and 'W' in dimensions:
                            self.image_dimensions[file_name] = dimensions
                            break
                        else:
                            QMessageBox.warning(self, "Invalid Dimensions", "You must assign both H and W dimensions.")
                    else:
                        return
            else:
                self.image_dimensions[file_name] = ['H', 'W']
    
        print(f"Final assigned dimensions: {self.image_dimensions[file_name]}")
    
        if self.image_dimensions[file_name]:
            self.create_slices(image_array, self.image_dimensions[file_name], image_path)
        else:
            rgb_image = self.convert_to_8bit_rgb(image_array)
            self.current_image = self.array_to_qimage(rgb_image)
            self.slices = []
            self.slice_list.clear()
    
        if self.slices:
            self.current_image = self.slices[0][1]
            self.current_slice = self.slices[0][0]
            self.slice_list.setCurrentRow(0)
            self.load_image_annotations()
            self.image_label.update()
    
        self.update_image_info()

    
    def create_slices(self, image_array, dimensions, image_path):
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        slices = []
        self.slice_list.clear()
    
        print(f"Creating slices for {base_name}")
        print(f"Dimensions: {dimensions}")
        print(f"Image array shape: {image_array.shape}")
    
        # Handle 2D images
        if image_array.ndim == 2:
            normalized_array = self.normalize_array(image_array)
            qimage = self.array_to_qimage(normalized_array)
            slice_name = f"{base_name}"
            slices.append((slice_name, qimage))
            self.slice_list.addItem(QListWidgetItem(slice_name))
        else:
            # For 3D or higher dimensional arrays
            height_index = dimensions.index('H')
            width_index = dimensions.index('W')
            slice_indices = [i for i, dim in enumerate(dimensions) if dim not in ['H', 'W']]
    
            for idx in np.ndindex(tuple(image_array.shape[i] for i in slice_indices)):
                full_idx = [slice(None)] * len(dimensions)
                for i, val in zip(slice_indices, idx):
                    full_idx[i] = val
                
                slice_array = image_array[tuple(full_idx)]
                rgb_slice = self.convert_to_8bit_rgb(slice_array)
                qimage = self.array_to_qimage(rgb_slice)
                
                slice_name = f"{base_name}_{'_'.join([f'{dimensions[i]}{val+1}' for i, val in zip(slice_indices, idx)])}"
                slices.append((slice_name, qimage))
                
                item = QListWidgetItem(slice_name)
                if slice_name in self.all_annotations:
                    item.setForeground(QColor(Qt.green))
                else:
                    item.setForeground(QColor(Qt.black))
                self.slice_list.addItem(item)
    
        self.image_slices[base_name] = slices
        self.slices = slices
    
        if slices:
            self.current_image = slices[0][1]
            self.current_slice = slices[0][0]
            self.slice_list.setCurrentRow(0)
            
            self.activate_slice(self.current_slice)
    
            slice_info = f"Total slices: {len(slices)}"
            for dim, size in zip(dimensions, image_array.shape):
                if dim not in ['H', 'W']:
                    slice_info += f", {dim}: {size}"
            self.update_image_info(additional_info=slice_info)
        else:
            print("No slices were created")
    
        return slices
    
    # def normalize_array(self, array):
    #     print(f"Normalizing array. Shape: {array.shape}, dtype: {array.dtype}")
    #     print(f"Array min: {array.min()}, max: {array.max()}, mean: {array.mean()}")
        
    #     array_float = array.astype(np.float32)
        
    #     if array.dtype == np.uint16:
    #         array_normalized = (array_float - array.min()) / (array.max() - array.min())
    #     elif array.dtype == np.uint8:
    #         p2, p98 = np.percentile(array, (2, 98))
    #         array_normalized = np.clip(array_float, p2, p98)
    #         array_normalized = (array_normalized - p2) / (p98 - p2)
    #     else:
    #         array_normalized = (array_float - array.min()) / (array.max() - array.min())
        
    #     # Apply gamma correction
    #     gamma = 0.8
    #     array_normalized = np.power(array_normalized, gamma)
        
    #     return (array_normalized * 255).astype(np.uint8)
    
    
    def normalize_array(self, array):
        print(f"Normalizing array. Shape: {array.shape}, dtype: {array.dtype}")
        print(f"Array min: {array.min()}, max: {array.max()}, mean: {array.mean()}")
        
        array_float = array.astype(np.float32)
        
        if array.dtype == np.uint16:
            array_normalized = (array_float - array.min()) / (array.max() - array.min())
        elif array.dtype == np.uint8:
            # For 8-bit images, use a simple contrast stretching
            p_low, p_high = np.percentile(array_float, (0, 100)) #Change these to 1, 99 or something to stretch the contrast for visualizing 8 bit images
            array_normalized = np.clip(array_float, p_low, p_high)
            array_normalized = (array_normalized - p_low) / (p_high - p_low)
        else:
            array_normalized = (array_float - array.min()) / (array.max() - array.min())
        
        # Apply gamma correction
        gamma = 1.0  # Adjust this value to fine-tune brightness (> 1 for darker, < 1 for brighter)
        array_normalized = np.power(array_normalized, gamma)
        
        return (array_normalized * 255).astype(np.uint8)
            
    def adjust_contrast(self, image, low_percentile=1, high_percentile=99):
        if image.dtype != np.uint8:
            p_low, p_high = np.percentile(image, (low_percentile, high_percentile))
            image_adjusted = np.clip(image, p_low, p_high)
            image_adjusted = (image_adjusted - p_low) / (p_high - p_low)
            return (image_adjusted * 255).astype(np.uint8)
        return image

            
    def activate_slice(self, slice_name):
        #print(f"Activating slice: {slice_name}")
        self.current_slice = slice_name
        self.image_file_name = slice_name  # Add this line
        self.load_image_annotations()
        self.update_annotation_list()
        self.image_label.update()
        
        items = self.slice_list.findItems(slice_name, Qt.MatchExactly)
        if items:
            self.slice_list.setCurrentItem(items[0])
            #print(f"Slice {slice_name} selected in list")
        else:
            print(f"Slice {slice_name} not found in list")
        
        #print(f"Current slice after activation: {self.current_slice}")
        #print(f"Current image_file_name after activation: {self.image_file_name}")

    
    def array_to_qimage(self, array):
        if array.ndim == 2:
            height, width = array.shape
            return QImage(array.data, width, height, width, QImage.Format_Grayscale8)
        elif array.ndim == 3 and array.shape[2] == 3:
            height, width, _ = array.shape
            bytes_per_line = 3 * width
            return QImage(array.data, width, height, bytes_per_line, QImage.Format_RGB888)
        else:
            raise ValueError(f"Unsupported array shape {array.shape} for conversion to QImage")

    def update_slice_list(self):
        self.slice_list.clear()
        for i, (slice_name, _) in enumerate(self.slices):
            item = QListWidgetItem(slice_name)
            if slice_name in self.all_annotations:
                item.setForeground(QColor(Qt.green))
            else:
                item.setForeground(QColor(Qt.black) if not self.dark_mode else QColor(Qt.white))
            self.slice_list.addItem(item)
            
            # Highlight the current slice
            if slice_name == self.current_slice:
                self.slice_list.setCurrentItem(item)
                       

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Up or event.key() == Qt.Key_Down:
            current_row = self.slice_list.currentRow()
            if event.key() == Qt.Key_Up and current_row > 0:
                self.slice_list.setCurrentRow(current_row - 1)
            elif event.key() == Qt.Key_Down and current_row < self.slice_list.count() - 1:
                self.slice_list.setCurrentRow(current_row + 1)
            self.switch_slice(self.slice_list.currentItem())
        else:
            super().keyPressEvent(event)
    
    def save_annotations(self):
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Annotations", "", "JSON Files (*.json)")
        if file_name:
            self.save_current_annotations()
    
            coco_format = {
                "images": [],
                "categories": [{"id": id, "name": name} for name, id in self.class_mapping.items()],
                "annotations": []
            }
            
            annotation_id = 1
            image_id = 1
    
            # #print(f"Total annotations to process: {len(self.all_annotations)}")
            # #print(f"Total slices: {len(self.slices)}")
            # #print("Slices:", [slice_name for slice_name, _ in self.slices])
            # #print("Image paths:", self.image_paths)
            
            # Create a mapping of slice names to their QImage objects
            slice_map = {slice_name: qimage for slice_name, qimage in self.slices}
            
            # Handle all images and slices
            for image_name, annotations in self.all_annotations.items():
               # #print(f"Processing: {image_name}")
                
                # Check if it's a slice (either in slice_map or has underscores and no file extension)
                is_slice = image_name in slice_map or ('_' in image_name and '.' not in image_name)
                
                if is_slice:
                   # #print(f"{image_name} is a slice")
                    qimage = slice_map.get(image_name)
                    if qimage is None:
                        # If the slice is not in slice_map, it might be a CZI slice or a TIFF slice
                        # We need to find the corresponding QImage in self.slices or self.image_slices
                        matching_slices = [s for s in self.slices if s[0] == image_name]
                        if matching_slices:
                            qimage = matching_slices[0][1]
                        else:
                            # Check in self.image_slices
                            for stack_slices in self.image_slices.values():
                                matching_slices = [s for s in stack_slices if s[0] == image_name]
                                if matching_slices:
                                    qimage = matching_slices[0][1]
                                    break
                        if qimage is None:
                           # print(f"No image data found for slice {image_name}, skipping")
                            continue
                    file_name_img = f"{image_name}.png"
                else:
                   # #print(f"{image_name} is an individual image")
                    # Check if the image_name exists in image_paths
                    image_path = next((path for name, path in self.image_paths.items() if image_name in name), None)
                    if not image_path:
                      #  #print(f"No image path found for {image_name}, skipping")
                        continue
                    if image_path.lower().endswith(('.tif', '.tiff', '.czi')):
                       # #print(f"Skipping main tiff/czi file: {image_name}")
                        continue
                    qimage = QImage(image_path)
                    file_name_img = image_name
    
             #   #print(f"Adding image info for: {file_name_img}")
                image_info = {
                    "file_name": file_name_img,
                    "height": qimage.height(),
                    "width": qimage.width(),
                    "id": image_id
                }
                coco_format["images"].append(image_info)
                
               # #print(f"Adding annotations for: {file_name_img}")
                for class_name, class_annotations in annotations.items():
                    for ann in class_annotations:
                        coco_ann = self.create_coco_annotation(ann, image_id, annotation_id)
                        coco_format["annotations"].append(coco_ann)
                        annotation_id += 1
                
                image_id += 1
    
          #  #print(f"Total images processed: {len(coco_format['images'])}")
          #  #print(f"Total annotations added: {len(coco_format['annotations'])}")
    
            # Save JSON file
            with open(file_name, 'w') as f:
                json.dump(coco_format, f, indent=2)
    
            # Save slice PNGs
            save_dir = os.path.dirname(file_name)
            self.save_slices(save_dir)
    
            QMessageBox.information(self, "Save Complete", "Annotations and slice images have been saved successfully.")
    
          #  #print("Save process completed")


        
    def save_slices(self, directory):
        slices_saved = False
        for image_slices in self.image_slices.values():
            for slice_name, qimage in image_slices:
                if slice_name in self.all_annotations:
                    file_path = os.path.join(directory, f"{slice_name}.png")
                    qimage.save(file_path, "PNG")
                    slices_saved = True
        
        if slices_saved:
            QMessageBox.information(self, "Slices Saved", "Annotated slices have been saved as separate PNG files.")
        else:
            QMessageBox.information(self, "No Slices Saved", "No annotated slices were found to save.")
            
    def create_coco_annotation(self, ann, image_id, annotation_id):
        coco_ann = {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": ann["category_id"],
            "area": calculate_area(ann),
            "iscrowd": 0
        }
        
        if "segmentation" in ann:
            coco_ann["segmentation"] = [ann["segmentation"]]
            coco_ann["bbox"] = calculate_bbox(ann["segmentation"])
        elif "bbox" in ann:
            coco_ann["bbox"] = ann["bbox"]
        
        return coco_ann

    def update_all_annotation_lists(self):
        for image_name in self.all_annotations.keys():
            self.update_annotation_list(image_name)
        self.update_annotation_list()  # Update for the current image/slice

    def update_annotation_list(self, image_name=None):
        self.annotation_list.clear()
        current_name = image_name or self.current_slice or self.image_file_name
        annotations = self.all_annotations.get(current_name, {})
        for class_name, class_annotations in annotations.items():
            color = self.image_label.class_colors.get(class_name, QColor(Qt.white))
            for i, annotation in enumerate(class_annotations, start=1):
                item_text = f"{class_name} - {i}"
                item = QListWidgetItem(item_text)
                item.setData(Qt.UserRole, annotation)
                item.setForeground(color)
                self.annotation_list.addItem(item)
                

            
    def update_slice_list_colors(self):
        for i in range(self.slice_list.count()):
            item = self.slice_list.item(i)
            slice_name = item.text()
            if slice_name in self.all_annotations and any(self.all_annotations[slice_name].values()):
                item.setForeground(QColor(Qt.green))
            else:
                item.setForeground(QColor(Qt.black) if not self.dark_mode else QColor(Qt.white))
                
    # def update_annotation_list_colors(self):
    #     for i in range(self.annotation_list.count()):
    #         item = self.annotation_list.item(i)
    #         annotation = item.data(Qt.UserRole)
    #         class_name = annotation['category_name']
    #         color = self.image_label.class_colors.get(class_name, QColor(Qt.white))
    #         item.setForeground(color)
            
    def update_annotation_list_colors(self, class_name=None, color=None):
        for i in range(self.annotation_list.count()):
            item = self.annotation_list.item(i)
            annotation = item.data(Qt.UserRole)
            # Update only the item for the specific class if class_name is provided
            if class_name is None or annotation['category_name'] == class_name:
                item_color = color if class_name else self.image_label.class_colors.get(annotation['category_name'], QColor(Qt.white))
                item.setForeground(item_color)

    def load_image_annotations(self):
        #print(f"Loading annotations for: {self.current_slice or self.image_file_name}")
        self.image_label.annotations.clear()
        current_name = self.current_slice or self.image_file_name
        if current_name in self.all_annotations:
            self.image_label.annotations = self.all_annotations[current_name].copy()
            #print(f"Loaded {len(self.image_label.annotations)} annotations")
        else:
            print("No annotations found")
        self.image_label.update()

    def save_current_annotations(self):
        if self.current_slice:
            current_name = self.current_slice
        elif self.image_file_name:
            current_name = self.image_file_name
        else:
            #print("Error: No current slice or image file name set")
            return
    
        #print(f"Saving annotations for: {current_name}")
        if self.image_label.annotations:
            self.all_annotations[current_name] = self.image_label.annotations.copy()
            #print(f"Saved {len(self.image_label.annotations)} annotations for {current_name}")
        elif current_name in self.all_annotations:
            del self.all_annotations[current_name]
            #print(f"Removed annotations for {current_name}")
    
        self.update_slice_list_colors()
    
        #print(f"All annotations now: {self.all_annotations.keys()}")
        #print(f"Current slice: {self.current_slice}")
        #print(f"Current image_file_name: {self.image_file_name}")
                
    def setup_class_list(self):
        """Set up the class list widget."""
        self.class_list = QListWidget()
        self.class_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.class_list.customContextMenuRequested.connect(self.show_class_context_menu)
        self.class_list.itemClicked.connect(self.on_class_selected)
        self.sidebar_layout.addWidget(QLabel("Classes:"))
        self.sidebar_layout.addWidget(self.class_list)

        self.add_class_button = QPushButton("Add Class")
        self.add_class_button.clicked.connect(self.add_class)
        self.sidebar_layout.addWidget(self.add_class_button)

    def setup_tool_buttons(self):
        """Set up the tool buttons."""
        self.tool_group = QButtonGroup(self)
        self.tool_group.setExclusive(False)
        self.polygon_button = QPushButton("Polygon Tool")
        self.polygon_button.setCheckable(True)
        self.rectangle_button = QPushButton("Rectangle Tool")
        self.rectangle_button.setCheckable(True)
        self.tool_group.addButton(self.polygon_button)
        self.tool_group.addButton(self.rectangle_button)
        self.sidebar_layout.addWidget(self.polygon_button)
        self.sidebar_layout.addWidget(self.rectangle_button)
        self.polygon_button.clicked.connect(self.toggle_tool)
        self.rectangle_button.clicked.connect(self.toggle_tool)
    
        self.sam_magic_wand_button = QPushButton("SAM2 Magic Wand")
        self.sam_magic_wand_button.setCheckable(True)
        self.tool_group.addButton(self.sam_magic_wand_button)
        self.sidebar_layout.addWidget(self.sam_magic_wand_button)
        self.sam_magic_wand_button.clicked.connect(self.toggle_tool)  #
        

    def setup_annotation_list(self):
        """Set up the annotation list widget."""
        self.annotation_list = QListWidget()
        self.annotation_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.annotation_list.itemSelectionChanged.connect(self.update_highlighted_annotations)
        self.sidebar_layout.addWidget(QLabel("Annotations:"))
        self.sidebar_layout.addWidget(self.annotation_list)

        self.delete_button = QPushButton("Delete Selected Annotations")
        self.delete_button.clicked.connect(self.delete_selected_annotations)
        self.sidebar_layout.addWidget(self.delete_button)

    def setup_sidebar(self):
        self.sidebar = QWidget()
        self.sidebar_layout = QVBoxLayout(self.sidebar)
        self.layout.addWidget(self.sidebar, 1)

        # Add existing buttons and widgets
        self.load_annotations_button = QPushButton("Import Saved Annotations")
        self.load_annotations_button.clicked.connect(self.load_annotations)
        self.sidebar_layout.addWidget(self.load_annotations_button)

        self.open_button = QPushButton("Open New Image Set")
        self.open_button.clicked.connect(self.open_images)
        self.sidebar_layout.addWidget(self.open_button)

        self.add_images_button = QPushButton("Add More Images")
        self.add_images_button.clicked.connect(self.add_images)
        self.sidebar_layout.addWidget(self.add_images_button)

        self.setup_class_list()
        self.setup_tool_buttons()
        self.setup_annotation_list()

        self.save_button = QPushButton("Save Annotations")
        self.save_button.clicked.connect(self.save_annotations)
        self.sidebar_layout.addWidget(self.save_button)

        self.sidebar_layout.addStretch(1)

        # Add font size selector
        self.setup_font_size_selector()

        # Dark mode toggle
        self.dark_mode_button = QPushButton("Toggle Dark Mode")
        self.dark_mode_button.clicked.connect(self.toggle_dark_mode)
        self.sidebar_layout.addWidget(self.dark_mode_button)

        # Help button
        self.help_button = QPushButton("Help")
        self.help_button.clicked.connect(self.show_help)
        self.sidebar_layout.addWidget(self.help_button)
        
        
    def setup_font_size_selector(self):
        font_size_label = QLabel("Font Size:")
        self.font_size_selector = QComboBox()
        self.font_size_selector.addItems(["Small", "Medium", "Large"])
        self.font_size_selector.setCurrentText("Medium")
        self.font_size_selector.currentTextChanged.connect(self.on_font_size_changed)
        
        self.sidebar_layout.addWidget(font_size_label)
        self.sidebar_layout.addWidget(self.font_size_selector)
        
    def on_font_size_changed(self, size):
        self.current_font_size = size
        self.apply_theme_and_font()
        

    def apply_theme_and_font(self):
        font_size = self.font_sizes[self.current_font_size]
        if self.dark_mode:
            style = soft_dark_stylesheet
        else:
            style = default_stylesheet
        
        # Combine the theme stylesheet with font size
        combined_style = f"{style}\nQWidget {{ font-size: {font_size}pt; }}"
        self.setStyleSheet(combined_style)
        
        # Apply font size to all widgets
        for widget in self.findChildren(QWidget):
            font = widget.font()
            font.setPointSize(font_size)
            widget.setFont(font)
        
        self.image_label.setFont(QFont("Arial", font_size))
        self.update_ui_colors()  # Ensure custom colors are reapplied
        self.update()

        
    def toggle_dark_mode(self):
        self.dark_mode = not self.dark_mode
        self.apply_theme_and_font()
        
    def apply_stylesheet(self):
        if self.dark_mode:
            self.setStyleSheet(soft_dark_stylesheet)
        else:
            self.setStyleSheet(default_stylesheet)
            
    def update_ui_colors(self):
        # Update colors for elements that need to retain their functionality
        self.update_annotation_list_colors()
        self.update_slice_list_colors()
        self.image_label.update()
        
    def setup_image_area(self):
        """Set up the main image area."""
        self.image_widget = QWidget()
        self.image_layout = QVBoxLayout(self.image_widget)
        self.layout.addWidget(self.image_widget, 3)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # Use the already initialized image_label
        self.image_label.setAlignment(Qt.AlignCenter)
        self.scroll_area.setWidget(self.image_label)
        
        self.image_layout.addWidget(self.scroll_area)


        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setMinimum(10)
        self.zoom_slider.setMaximum(500)
        self.zoom_slider.setValue(100)
        self.zoom_slider.setTickPosition(QSlider.TicksBelow)
        self.zoom_slider.setTickInterval(50)
        self.zoom_slider.valueChanged.connect(self.zoom_image)
        self.image_layout.addWidget(self.zoom_slider)
        self.image_info_label = QLabel()
        self.image_layout.addWidget(self.image_info_label)

    def setup_image_list(self):
        """Set up the image list area."""
        self.image_list_widget = QWidget()
        self.image_list_layout = QVBoxLayout(self.image_list_widget)
        self.layout.addWidget(self.image_list_widget, 1)

        self.image_list_label = QLabel("Images:")
        self.image_list_layout.addWidget(self.image_list_label)

        self.image_list = QListWidget()
        self.image_list.itemClicked.connect(self.switch_image)
        self.image_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.image_list.customContextMenuRequested.connect(self.show_image_context_menu)
        self.image_list_layout.addWidget(self.image_list)

        self.clear_all_button = QPushButton("Clear All Images and Annotations")
        self.clear_all_button.clicked.connect(self.clear_all)
        self.image_list_layout.addWidget(self.clear_all_button)


    # In the ImageAnnotator class, update the show_help method:
    def show_help(self):
        self.help_window = HelpWindow(dark_mode=self.dark_mode, font_size=self.font_sizes[self.current_font_size])
        self.help_window.show()

            
    def add_images(self):
        file_names, _ = QFileDialog.getOpenFileNames(self, "Add Images", "", "Image Files (*.png *.jpg *.bmp *.tif *.tiff *.czi)")
        if file_names:
            self.add_images_to_list(file_names)
            
            
    def clear_all(self):
        reply = self.show_question('Clear All',
                                   "Are you sure you want to clear all images and annotations? This action cannot be undone.")
    
        if reply == QMessageBox.Yes:
            # Clear images
            self.image_list.clear()
            self.image_paths.clear()
            self.all_images.clear()
            self.current_image = None
            self.image_file_name = ""
            self.image_label.clear()
    
            # Clear annotations
            self.all_annotations.clear()
            self.annotation_list.clear()
            self.image_label.annotations.clear()
            self.clear_highlighted_annotation()
    
            # Reset class-related data
            self.class_list.clear()
            self.image_label.class_colors.clear()
            self.class_mapping.clear()
        
            # Clear slices
            self.image_slices.clear()
            self.slices = []
            self.slice_list.clear()
            self.current_slice = None
            self.current_stack = None
            
            # Update UI
            self.image_label.update()
            self.update_image_info()
            self.show_info("Clear All", "All images and annotations have been cleared.")
            
            
    def show_warning(self, title, message):
        QMessageBox.warning(self, title, message)

    def show_info(self, title, message):
        QMessageBox.information(self, title, message)
        
    def update_image_info(self, additional_info=None):
        if self.current_image:
            width = self.current_image.width()
            height = self.current_image.height()
            info = f"Image: {width}x{height}"
            if additional_info:
                info += f", {additional_info}"
            self.image_info_label.setText(info)
        else:
            self.image_info_label.setText("No image loaded")
    
    def show_question(self, title, message):
        return QMessageBox.question(self, title, message,
                                    QMessageBox.Yes | QMessageBox.No,
                                    QMessageBox.No)
            
    def show_image_context_menu(self, position):
        menu = QMenu()
        delete_action = menu.addAction("Remove Image")

        action = menu.exec_(self.image_list.mapToGlobal(position))
        
        if action == delete_action:
            self.remove_image()
    
    def remove_image(self):
        current_item = self.image_list.currentItem()
        if current_item:
            file_name = current_item.text()
            
            # Remove from all data structures
            self.image_list.takeItem(self.image_list.row(current_item))
            self.image_paths.pop(file_name, None)
            self.all_images = [img for img in self.all_images if img["file_name"] != file_name]
            self.all_annotations.pop(file_name, None)

            # If the removed image was the current image, switch to another image
            if self.image_file_name == file_name:
                if self.image_list.count() > 0:
                    self.switch_image(self.image_list.item(0))
                else:
                    self.current_image = None
                    self.image_file_name = ""
                    self.image_label.clear()
                    self.annotation_list.clear()       


    def load_annotations(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Load Annotations", "", "JSON Files (*.json)")
        if file_name:
            with open(file_name, 'r') as f:
                self.loaded_json = json.load(f)
            
            # Load categories
            self.class_list.clear()
            self.image_label.class_colors.clear()
            self.class_mapping.clear()
            for category in self.loaded_json["categories"]:
                class_name = category["name"]
                self.class_mapping[class_name] = category["id"]
                
                # Assign a color if not already assigned
                if class_name not in self.image_label.class_colors:
                    color = QColor(Qt.GlobalColor(len(self.image_label.class_colors) % 16 + 7))
                    self.image_label.class_colors[class_name] = color
                
                # Add item to class list with color indicator
                item = QListWidgetItem(class_name)
                self.update_class_item_color(item, self.image_label.class_colors[class_name])
                self.class_list.addItem(item)
             
            # Create a mapping of image IDs to file names
            image_id_to_filename = {img["id"]: img["file_name"] for img in self.loaded_json["images"]}
            
            # Load image information
            json_images = {img["file_name"]: img for img in self.loaded_json["images"]}
            
            # Update existing images and add new ones from JSON
            updated_all_images = []
            for i in range(self.image_list.count()):
                item = self.image_list.item(i)
                file_name = item.text()
                if file_name in json_images:
                    updated_image = self.all_images[i].copy()
                    updated_image.update(json_images[file_name])
                    updated_all_images.append(updated_image)
                    del json_images[file_name]
                else:
                    updated_all_images.append(self.all_images[i])
            
            # Add remaining images from JSON
            for img in json_images.values():
                updated_all_images.append(img)
                self.image_list.addItem(img["file_name"])
            
            self.all_images = updated_all_images
            
            # Load annotations
            self.all_annotations.clear()
            for annotation in self.loaded_json["annotations"]:
                image_id = annotation["image_id"]
                file_name = image_id_to_filename.get(image_id)
                if file_name:
                    if file_name not in self.all_annotations:
                        self.all_annotations[file_name] = {}
                    
                    category = next((cat for cat in self.loaded_json["categories"] if cat["id"] == annotation["category_id"]), None)
                    if category:
                        category_name = category["name"]
                        if category_name not in self.all_annotations[file_name]:
                            self.all_annotations[file_name][category_name] = []
                        
                        ann = {
                            "category_id": annotation["category_id"],
                            "category_name": category_name,
                        }
                        
                        if "segmentation" in annotation:
                            ann["segmentation"] = annotation["segmentation"][0]
                            ann["type"] = "polygon"
                        elif "bbox" in annotation:
                            # Store bbox as is, don't convert to rectangle
                            ann["bbox"] = annotation["bbox"]
                            ann["type"] = "bbox"
                        
                        self.all_annotations[file_name][category_name].append(ann)
            
            # Check for missing images
            missing_images = [img["file_name"] for img in self.loaded_json["images"] if img["file_name"] not in self.image_paths]
            if missing_images:
                self.show_warning("Missing Images", "The following images are missing:\n" + "\n".join(missing_images))
            
            # Reload the current image if it exists, otherwise load the first image
            if self.image_file_name and self.image_file_name in self.all_annotations:
                self.switch_image(self.image_list.findItems(self.image_file_name, Qt.MatchExactly)[0])
            elif self.all_images:
                self.switch_image(self.image_list.item(0))




    def clear_highlighted_annotation(self):
        self.image_label.highlighted_annotation = None
        self.image_label.update()
        
    def update_highlighted_annotations(self):
        selected_items = self.annotation_list.selectedItems()
        self.image_label.highlighted_annotations = [item.data(Qt.UserRole) for item in selected_items]
        self.image_label.update()
        
    def delete_selected_annotations(self):
        selected_items = self.annotation_list.selectedItems()
        if not selected_items:
            return
    
        reply = QMessageBox.question(self, 'Delete Annotations',
                                     f"Are you sure you want to delete {len(selected_items)} annotation(s)?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            for item in selected_items:
                annotation = item.data(Qt.UserRole)
                category_name = annotation['category_name']
                self.image_label.annotations[category_name].remove(annotation)
                self.annotation_list.takeItem(self.annotation_list.row(item))
            
            self.image_label.highlighted_annotations.clear()
            self.image_label.update()
            
            # Update all_annotations
            current_name = self.current_slice or self.image_file_name
            self.all_annotations[current_name] = self.image_label.annotations
            
            # Update slice list colors
            self.update_slice_list_colors()

        


    def display_image(self):
        if self.current_image:
            if isinstance(self.current_image, QImage):
                pixmap = QPixmap.fromImage(self.current_image)
            elif isinstance(self.current_image, QPixmap):
                pixmap = self.current_image
            else:
                print(f"Unexpected image type: {type(self.current_image)}")
                return
            
            if not pixmap.isNull():
                self.image_label.setPixmap(pixmap)
                self.image_label.adjustSize()
            else:
                print("Error: Null pixmap")
        else:
            self.image_label.clear()
            print("No current image to display")

    def add_class(self):
        class_name, ok = QInputDialog.getText(self, "Add Class", "Enter class name:")
        if ok and class_name and class_name not in self.class_mapping:
            color = QColor(Qt.GlobalColor(len(self.image_label.class_colors) % 16 + 7))
            self.image_label.class_colors[class_name] = color
            self.class_mapping[class_name] = len(self.class_mapping) + 1
            
            item = QListWidgetItem(class_name)
            self.update_class_item_color(item, color)
            self.class_list.addItem(item)
            
            self.class_list.setCurrentItem(item)
            self.current_class = class_name

    def toggle_tool(self):
        sender = self.sender()
        other_buttons = [btn for btn in self.tool_group.buttons() if btn != sender]
    
        if sender.isChecked():
            # Uncheck all other buttons
            for btn in other_buttons:
                btn.setChecked(False)
            
            # Set the current tool based on the checked button
            if sender == self.polygon_button:
                self.image_label.current_tool = "polygon"
            elif sender == self.rectangle_button:
                self.image_label.current_tool = "rectangle"
            elif sender == self.sam_magic_wand_button:
                self.image_label.current_tool = "sam_magic_wand"
            
            # If a class is not selected, select the first one (if available)
            if self.current_class is None and self.class_list.count() > 0:
                self.class_list.setCurrentRow(0)
                self.current_class = self.class_list.currentItem().text()
            elif self.class_list.count() == 0:
                QMessageBox.warning(self, "No Class Selected", "Please add a class before using annotation tools.")
                sender.setChecked(False)
                self.image_label.current_tool = None
        else:
            self.image_label.current_tool = None
    
        # Update SAM magic wand state
        self.image_label.sam_magic_wand_active = (self.image_label.current_tool == "sam_magic_wand")
        self.image_label.setCursor(Qt.CrossCursor if self.image_label.sam_magic_wand_active else Qt.ArrowCursor)
    
        # Update UI based on the current tool
        self.update_ui_for_current_tool()
        
    def update_ui_for_current_tool(self):
        # Disable finish_polygon_button if it still exists in your code
        if hasattr(self, 'finish_polygon_button'):
            self.finish_polygon_button.setEnabled(self.image_label.current_tool in ["polygon", "rectangle"])
    
        # Update button states
        self.polygon_button.setChecked(self.image_label.current_tool == "polygon")
        self.rectangle_button.setChecked(self.image_label.current_tool == "rectangle")
        self.sam_magic_wand_button.setChecked(self.image_label.current_tool == "sam_magic_wand")        

    def on_class_selected(self):
        selected_item = self.class_list.currentItem()
        if selected_item:
            self.current_class = selected_item.text()

    def show_class_context_menu(self, position):
        menu = QMenu()
        rename_action = menu.addAction("Rename Class")
        change_color_action = menu.addAction("Change Color")
        delete_action = menu.addAction("Delete Class")

        action = menu.exec_(self.class_list.mapToGlobal(position))
        
        if action == rename_action:
            self.rename_class()
        elif action == change_color_action:
            self.change_class_color()
        elif action == delete_action:
            self.delete_class()
            
    def change_class_color(self):
        current_item = self.class_list.currentItem()
        if current_item:
            class_name = current_item.text()
            current_color = self.image_label.class_colors.get(class_name, QColor(Qt.white))
            color = QColorDialog.getColor(current_color, self, f"Select Color for {class_name}")
            
            if color.isValid():
                self.image_label.class_colors[class_name] = color
                self.update_class_item_color(current_item, color)
                self.update_annotation_list_colors(class_name, color)
                self.image_label.update()
                

        
    def update_class_item_color(self, item, color):
        pixmap = QPixmap(16, 16)
        pixmap.fill(color)
        item.setIcon(QIcon(pixmap))

    def rename_class(self):
        current_item = self.class_list.currentItem()
        if current_item:
            old_name = current_item.text()
            new_name, ok = QInputDialog.getText(self, "Rename Class", "Enter new class name:", text=old_name)
            if ok and new_name and new_name != old_name:
                # Update class mapping
                old_id = self.class_mapping[old_name]
                self.class_mapping[new_name] = old_id
                del self.class_mapping[old_name]
    
                # Update class colors
                self.image_label.class_colors[new_name] = self.image_label.class_colors.pop(old_name)
    
                # Update annotations for all images and slices
                for image_name, image_annotations in self.all_annotations.items():
                    if old_name in image_annotations:
                        image_annotations[new_name] = image_annotations.pop(old_name)
                        for annotation in image_annotations[new_name]:
                            annotation['category_name'] = new_name
    
                # Update current image annotations
                if old_name in self.image_label.annotations:
                    self.image_label.annotations[new_name] = self.image_label.annotations.pop(old_name)
                    for annotation in self.image_label.annotations[new_name]:
                        annotation['category_name'] = new_name
    
                # Update current class if it's the renamed one
                if self.current_class == old_name:
                    self.current_class = new_name
    
                # Update annotation list for all images and slices
                self.update_all_annotation_lists()
    
                # Update class list
                current_item.setText(new_name)
    
                # Update the image label
                self.image_label.update()
    
                #print(f"Class renamed from '{old_name}' to '{new_name}'")

    def delete_class(self):
        current_item = self.class_list.currentItem()
        if current_item:
            class_name = current_item.text()
            # Remove class color
            self.image_label.class_colors.pop(class_name, None)
            
            # Remove class from mapping
            del self.class_mapping[class_name]
            
            # Remove annotations for this class from all images
            for image_annotations in self.all_annotations.values():
                image_annotations.pop(class_name, None)
            
            # Remove annotations for this class from current image
            self.image_label.annotations.pop(class_name, None)
            
            # Update annotation list
            self.update_annotation_list()

            # Remove class from list
            self.class_list.takeItem(self.class_list.row(current_item))
            
            self.image_label.update()

    def finish_polygon(self):
        if self.image_label.current_tool == "polygon" and len(self.image_label.current_annotation) > 2:
            if self.current_class is None:
                QMessageBox.warning(self, "No Class Selected", "Please select a class before finishing the annotation.")
                return
            
            new_annotation = {
                "segmentation": [coord for point in self.image_label.current_annotation for coord in point],
                "category_id": self.class_mapping[self.current_class],
                "category_name": self.current_class,
            }
            self.image_label.annotations.setdefault(self.current_class, []).append(new_annotation)
            self.add_annotation_to_list(new_annotation)
            self.image_label.clear_current_annotation()
            self.image_label.reset_annotation_state()
            self.image_label.update()
            
            # Save the current annotations
            self.save_current_annotations()
            
            # Update the slice list colors
            self.update_slice_list_colors()


    def highlight_annotation(self, item):
        self.image_label.highlighted_annotation = item.data(Qt.UserRole)
        self.image_label.update()

    def delete_annotation(self):
        current_item = self.annotation_list.currentItem()
        if current_item:
            annotation = current_item.data(Qt.UserRole)
            category_name = annotation['category_name']
            self.image_label.annotations[category_name].remove(annotation)
            self.annotation_list.takeItem(self.annotation_list.row(current_item))
            self.image_label.highlighted_annotation = None
            self.image_label.update()

    def add_annotation_to_list(self, annotation):
        class_name = annotation['category_name']
        color = self.image_label.class_colors.get(class_name, QColor(Qt.white))
        annotations = self.image_label.annotations.get(class_name, [])
        item_text = f"{class_name} - {len(annotations)}"
        item = QListWidgetItem(item_text)
        item.setData(Qt.UserRole, annotation)
        item.setForeground(color)
        self.annotation_list.addItem(item)
        
    


    def zoom_in(self):
        new_zoom = min(self.image_label.zoom_factor + 0.1, 5.0)
        self.set_zoom(new_zoom)

    def zoom_out(self):
        new_zoom = max(self.image_label.zoom_factor - 0.1, 0.1)
        self.set_zoom(new_zoom)

    def set_zoom(self, zoom_factor):
        self.image_label.set_zoom(zoom_factor)
        self.zoom_slider.setValue(int(zoom_factor * 100))

    def zoom_image(self):
        zoom_factor = self.zoom_slider.value() / 100
        self.set_zoom(zoom_factor)

    def disable_tools(self):
        self.polygon_button.setEnabled(False)
        self.rectangle_button.setEnabled(False)
        #self.finish_polygon_button.setEnabled(False)

    def enable_tools(self):
        self.polygon_button.setEnabled(True)
        self.rectangle_button.setEnabled(True)

            
    def finish_rectangle(self):
        if self.image_label.current_rectangle:
            x1, y1, x2, y2 = self.image_label.current_rectangle
            new_annotation = {
                "segmentation": [x1, y1, x2, y1, x2, y2, x1, y2],
                "category_id": self.class_mapping[self.current_class],
                "category_name": self.current_class,
            }
            self.image_label.annotations.setdefault(self.current_class, []).append(new_annotation)
            self.add_annotation_to_list(new_annotation)
            self.image_label.start_point = None
            self.image_label.end_point = None
            self.image_label.current_rectangle = None
            self.image_label.update()
            
            # Save the current annotations
            self.save_current_annotations()
            
            # Update the slice list colors
            self.update_slice_list_colors()

    def enter_edit_mode(self, annotation):
        self.editing_mode = True
        self.disable_tools()
        # self.finish_polygon_button.setText("Finish Editing")
        # self.finish_polygon_button.setEnabled(True)
        # self.finish_polygon_button.clicked.disconnect()
        # self.finish_polygon_button.clicked.connect(self.exit_edit_mode)
        QMessageBox.information(self, "Edit Mode", "You are now in edit mode. Click and drag points to move them, Shift+Click to delete points, or click on edges to add new points.")

    def exit_edit_mode(self):
        self.editing_mode = False
        self.enable_tools()
        # self.finish_polygon_button.setText("Finish Polygon")
        # self.finish_polygon_button.setEnabled(False)
        # self.finish_polygon_button.clicked.disconnect()
        # self.finish_polygon_button.clicked.connect(self.finish_polygon)
        self.image_label.editing_polygon = None
        self.image_label.editing_point_index = None
        self.image_label.hover_point_index = None
        self.update_annotation_list()
        self.image_label.update()

