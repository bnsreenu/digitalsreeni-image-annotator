import os
import json
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QFileDialog, QListWidget, QInputDialog, 
                             QLabel, QButtonGroup, QListWidgetItem, QScrollArea, 
                             QSlider, QMenu, QMessageBox, QColorDialog, QDialog,
                             QGridLayout, QComboBox, QAbstractItemView, QProgressDialog,
                             QApplication, QAction, QLineEdit, QTextEdit)
from PyQt5.QtGui import QPixmap, QColor, QIcon, QImage, QFont, QKeySequence
from PyQt5.QtCore import Qt
import numpy as np
from tifffile import TiffFile
from czifile import CziFile


from .image_label import ImageLabel
from .utils import calculate_area, calculate_bbox
from .help_window import HelpWindow

from .soft_dark_stylesheet import soft_dark_stylesheet
from .default_stylesheet import default_stylesheet

from .dataset_splitter import DatasetSplitterTool
from .annotation_statistics import show_annotation_statistics
from .coco_json_combiner import show_coco_json_combiner
from .stack_to_slices import show_stack_to_slices
from .image_patcher import show_image_patcher
from .image_augmenter import show_image_augmenter
from .sam_utils import SAMUtils
from .snake_game import SnakeGame

from shapely.geometry import Polygon, MultiPolygon, Point
from shapely.ops import unary_union
from shapely.validation import make_valid
import shapely

from .export_formats import (
    export_coco_json, export_yolo_v8, export_labeled_images, 
    export_semantic_labels, export_pascal_voc_bbox, export_pascal_voc_both
)

from .import_formats import import_coco_json, import_yolo_v8
from .import_formats import process_import_format

import shutil 
import copy
from ultralytics import SAM

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
        
        self.create_menu_bar()
        
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
        self.image_shapes = {}
        
        #For paint brush and eraser
        self.paint_brush_size = 10
        self.eraser_size = 10
        # Initialize SAM utils
        self.current_sam_model = None
        self.sam_utils = SAMUtils()
    
        # Create sam_magic_wand_button
        self.sam_magic_wand_button = QPushButton("Magic Wand")
        self.sam_magic_wand_button.setCheckable(True)
        self.sam_magic_wand_button.setEnabled(False)  # Initially disable the button
    
        # Initialize tool group
        self.tool_group = QButtonGroup(self)
        self.tool_group.setExclusive(False)
    
        # Font size control
        self.font_sizes = {"Small": 8, "Medium": 10, "Large": 12, "XL": 14, "XXL": 16}   #Also, add the otions in create_menu_bar method
        self.current_font_size = "Medium"
    
        # Dark mode control
        self.dark_mode = False
    
        # Setup UI components
        self.setup_ui()
        
        # Apply theme and font (this includes stylesheet and font size application)
        self.apply_theme_and_font()
    
        # Connect sam_magic_wand_button
        self.sam_magic_wand_button.clicked.connect(self.toggle_tool)


    def setup_ui(self):
        # Initialize the main layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QHBoxLayout(self.central_widget)
    
        # Initialize tool group
        self.tool_group = QButtonGroup(self)
        self.tool_group.setExclusive(False)
    
        # Setup UI components
        self.setup_sidebar()
        self.setup_image_area()
        self.setup_image_list()
        self.setup_slice_list()
        self.update_ui_for_current_tool()    



    def update_window_title(self):
        base_title = "Image Annotator"
        if hasattr(self, 'current_project_file'):
            project_name = os.path.basename(self.current_project_file)
            project_name = os.path.splitext(project_name)[0]  # Remove the file extension
            self.setWindowTitle(f"{base_title} - {project_name}")
        else:
            self.setWindowTitle(base_title)
        

                
    def new_project(self):
        project_file, _ = QFileDialog.getSaveFileName(self, "Create New Project", "", "Image Annotator Project (*.iap)")
        if project_file:
            # Ensure the file has the correct extension
            if not project_file.lower().endswith('.iap'):
                project_file += '.iap'
            
            self.current_project_file = project_file
            self.current_project_dir = os.path.dirname(project_file)
            
            # Create the images directory
            images_dir = os.path.join(self.current_project_dir, "images")
            os.makedirs(images_dir, exist_ok=True)
            
            # Clear existing data without showing messages
            self.clear_all(new_project=True, show_messages=False)
            
            # Save the empty project without showing a message
            self.save_project(show_message=False)
            
            # Keep only this message
            self.show_info("New Project", f"New project created at {self.current_project_file}")
            self.update_window_title()
            
          
    
    def open_project(self):
        project_file, _ = QFileDialog.getOpenFileName(self, "Open Project", "", "Image Annotator Project (*.iap)")
        if project_file:
            with open(project_file, 'r') as f:
                project_data = json.load(f)
            
            self.clear_all(show_messages=False)
            self.current_project_file = project_file
            self.current_project_dir = os.path.dirname(project_file)
            
            self.all_images = project_data.get('images', [])
            self.all_annotations = project_data.get('annotations', {})
            self.image_paths = project_data.get('image_paths', {})
            
            # Load classes
            for class_info in project_data['classes']:
                self.add_class(class_info['name'], QColor(class_info['color']))
            
            # Select the last added class
            if self.class_list.count() > 0:
                last_class_index = self.class_list.count() - 1
                self.select_class(last_class_index)
            
            # Load all annotations first
            self.all_annotations.clear()
            for image_info in project_data['images']:
                if image_info.get('is_multi_slice', False):
                    for slice_info in image_info.get('slices', []):
                        self.all_annotations[slice_info['name']] = slice_info['annotations']
                else:
                    self.all_annotations[image_info['file_name']] = image_info.get('annotations', {})
            
            # Now load images
            missing_images = []
            for image_info in project_data['images']:
                image_path = os.path.join(self.current_project_dir, "images", image_info['file_name'])
                
                if not os.path.exists(image_path):
                    missing_images.append(image_info['file_name'])
                    continue
                
                # Update image_paths
                self.image_paths[image_info['file_name']] = image_path
                
                if image_info.get('is_multi_slice', False):
                    # Load multi-slice image with stored dimensions
                    dimensions = image_info.get('dimensions', [])
                    shape = image_info.get('shape', [])
                    self.load_multi_slice_image(image_path, dimensions, shape)
                else:
                    self.add_images_to_list([image_path])
            
            self.update_ui()
            
            # Select the first image if available
            if self.image_list.count() > 0:
                self.image_list.setCurrentRow(0)
                self.switch_image(self.image_list.item(0))
                
            self.update_window_title()
            
            # Check for missing images
            if missing_images:
                self.handle_missing_images(missing_images)

    def handle_missing_images(self, missing_images):
        message = "The following images have annotations but were not found in the project directory:\n\n"
        message += "\n".join(missing_images[:10])  # Show first 10 missing images
        if len(missing_images) > 10:
            message += f"\n... and {len(missing_images) - 10} more."
        message += "\n\nWould you like to locate these images now?"
        
        reply = QMessageBox.question(self, "Missing Images", message, 
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        
        if reply == QMessageBox.Yes:
            self.load_missing_images(missing_images)
        else:
            self.remove_missing_images(missing_images)
            

    def remove_missing_images(self, missing_images):
        for image_name in missing_images:
            # Remove from all_images
            self.all_images = [img for img in self.all_images if img['file_name'] != image_name]
            
            # Remove from image_paths
            self.image_paths.pop(image_name, None)
            
            # Remove from all_annotations
            self.all_annotations.pop(image_name, None)
            
            # If it's a multi-slice image, remove all related slices
            base_name = os.path.splitext(image_name)[0]
            if base_name in self.image_slices:
                for slice_name, _ in self.image_slices[base_name]:
                    self.all_annotations.pop(slice_name, None)
                del self.image_slices[base_name]
        
        self.update_ui()
        QMessageBox.information(self, "Images Removed", 
                                f"{len(missing_images)} missing images and their annotations have been removed from the project.")


        
    
    def prompt_load_missing_images(self, missing_images):
        message = "The following images have annotations but were not found in the project directory:\n\n"
        message += "\n".join(missing_images[:10])  # Show first 10 missing images
        if len(missing_images) > 10:
            message += f"\n... and {len(missing_images) - 10} more."
        message += "\n\nWould you like to locate these images now?"
        
        reply = QMessageBox.question(self, "Load Missing Images", message, 
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        
        if reply == QMessageBox.Yes:
            self.load_missing_images(missing_images)
    


    def load_missing_images(self, missing_images):
        files, _ = QFileDialog.getOpenFileNames(self, "Select Missing Images", "", "Image Files (*.png *.jpg *.bmp *.tif *.tiff *.czi)")
        if files:
            images_loaded = 0
            for file_path in files:
                file_name = os.path.basename(file_path)
                if file_name in missing_images:
                    dst_path = os.path.join(self.current_project_dir, "images", file_name)
                    shutil.copy2(file_path, dst_path)
                    self.image_paths[file_name] = dst_path
                    
                    # Add the image to all_images if it's not already there
                    if not any(img['file_name'] == file_name for img in self.all_images):
                        self.all_images.append({
                            "file_name": file_name,
                            "height": 0,  
                            "width": 0,   
                            "id": len(self.all_images) + 1,
                            "is_multi_slice": False
                        })
                    images_loaded += 1
                    missing_images.remove(file_name)
            
            self.update_image_list()
            if images_loaded > 0:
                self.image_list.setCurrentRow(0)  # Select the first image
                self.switch_image(self.image_list.item(0))  # Display the first image
            QMessageBox.information(self, "Images Loaded", 
                                    f"Successfully copied and loaded {images_loaded} out of {len(files)} selected images.")
            
            # If there are still missing images, prompt again
            if missing_images:
                self.prompt_load_missing_images(missing_images)


    def update_image_list(self):
        self.image_list.clear()
        for image_info in self.all_images:
            self.image_list.addItem(image_info['file_name'])

    def select_class(self, index):
        if 0 <= index < self.class_list.count():
            item = self.class_list.item(index)
            self.class_list.setCurrentItem(item)
            self.current_class = item.text()
            print(f"Selected class: {self.current_class}")
        else:
            print("Invalid class index")
    
    
    def close_project(self):
        if hasattr(self, 'current_project_file'):
            reply = QMessageBox.question(self, 'Close Project',
                                         "Do you want to save the current project before closing?",
                                         QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
            
            if reply == QMessageBox.Yes:
                self.save_project(show_message=False)  # Save without showing a message
            elif reply == QMessageBox.Cancel:
                return  # User cancelled the operation
    
        # Clear all data
        self.clear_all(new_project=True, show_messages=False)
        
        # Reset project-related attributes
        if hasattr(self, 'current_project_file'):
            del self.current_project_file
        if hasattr(self, 'current_project_dir'):
            del self.current_project_dir
    
        # Update the window title
        self.update_window_title()
    
        # No message box for project closed
    
    
    def delete_selected_class(self):
        selected_items = self.class_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "No Selection", "Please select a class to delete.")
            return
        
        class_name = selected_items[0].text()
        reply = QMessageBox.question(self, 'Delete Class',
                                     f"Are you sure you want to delete the class '{class_name}'?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.delete_class(class_name)  # Sreeni note: Implement this method to handle class deletion
        
    def check_missing_images(self):
        missing_images = [img['file_name'] for img in self.all_images if img['file_name'] not in self.image_paths or not os.path.exists(self.image_paths[img['file_name']])]
        if missing_images:
            self.prompt_load_missing_images(missing_images)
        

    def convert_to_serializable(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, list):
            return [self.convert_to_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self.convert_to_serializable(value) for key, value in obj.items()}
        else:
            return obj
        
    
    def save_project(self, show_message=True):
        if not hasattr(self, 'current_project_file') or not self.current_project_file:
            self.current_project_file, _ = QFileDialog.getSaveFileName(self, "Save Project", "", "Image Annotator Project (*.iap)")
            if not self.current_project_file:
                return  # User cancelled the save dialog
            
        self.current_project_dir = os.path.dirname(self.current_project_file)
    
        # Check if images are in the correct directory structure
        images_dir = os.path.join(self.current_project_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        
        images_to_copy = []
        for file_name, src_path in self.image_paths.items():
            dst_path = os.path.join(images_dir, file_name)
            if os.path.abspath(src_path) != os.path.abspath(dst_path):
                if not os.path.exists(dst_path):
                    images_to_copy.append((file_name, src_path, dst_path))
    
        if images_to_copy:
            reply = QMessageBox.question(self, 'Image Directory Structure',
                                         f"The project structure requires all images to be in an 'images' subdirectory. "
                                         f"{len(images_to_copy)} images need to be copied to the correct location. "
                                         f"Do you want to copy these images?",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            
            if reply == QMessageBox.Yes:
                for file_name, src_path, dst_path in images_to_copy:
                    try:
                        shutil.copy2(src_path, dst_path)
                        self.image_paths[file_name] = dst_path
                    except Exception as e:
                        QMessageBox.warning(self, "Copy Failed", f"Failed to copy {file_name}: {str(e)}")
                        return
            else:
                QMessageBox.warning(self, "Save Cancelled", "Project cannot be saved without the correct directory structure.")
                return

    
        # Prepare image data
        images_data = []
        for image_info in self.all_images:
            file_name = image_info['file_name']
            image_data = {
                'file_name': file_name,
                'width': image_info['width'],
                'height': image_info['height'],
                'is_multi_slice': image_info['is_multi_slice']
            }
    
            if image_data['is_multi_slice']:
                base_name_without_ext = os.path.splitext(file_name)[0]
                image_data['slices'] = []
                for slice_name, _ in self.image_slices.get(base_name_without_ext, []):
                    slice_data = {
                        'name': slice_name,
                        'annotations': self.convert_to_serializable(self.all_annotations.get(slice_name, {}))
                    }
                    image_data['slices'].append(slice_data)
                
                image_data['dimensions'] = self.convert_to_serializable(self.image_dimensions.get(base_name_without_ext, []))
                image_data['shape'] = self.convert_to_serializable(self.image_shapes.get(base_name_without_ext, []))
            else:
                image_data['annotations'] = {}
                for class_name, annotations in self.all_annotations.get(file_name, {}).items():
                    image_data['annotations'][class_name] = [ann.copy() for ann in annotations]
    
            images_data.append(image_data)
    
        # Create project data
        project_data = {
            'classes': [
                {'name': name, 'color': color.name()} 
                for name, color in self.image_label.class_colors.items()
            ],
            'images': images_data,
            'image_paths': {k: v for k, v in self.image_paths.items() if os.path.exists(v)}
        }
    
        # Save project data
        with open(self.current_project_file, 'w') as f:
            json.dump(self.convert_to_serializable(project_data), f, indent=2)
    
        if show_message:
            self.show_info("Project Saved", f"Project saved to {self.current_project_file}")
    
        # Update the window title
        self.update_window_title()
        
        # Update image_paths to reflect the correct locations
        for file_name in self.image_paths.keys():
            self.image_paths[file_name] = os.path.join(images_dir, file_name)
        


    def save_project_as(self):
        new_project_file, _ = QFileDialog.getSaveFileName(self, "Save Project As", "", "Image Annotator Project (*.iap)")
        if new_project_file:
            # Ensure the file has the correct extension
            if not new_project_file.lower().endswith('.iap'):
                new_project_file += '.iap'
            
            # Store the original project file
            original_project_file = getattr(self, 'current_project_file', None)
            
            # Set the new project file as the current one
            self.current_project_file = new_project_file
            self.current_project_dir = os.path.dirname(new_project_file)
            
            # Save the project with the new name
            self.save_project(show_message=False)
            
            # Update the window title
            self.update_window_title()
            
            # Show a success message
            QMessageBox.information(self, "Project Saved As", f"Project saved as:\n{new_project_file}")
            
            # If this was originally a new unsaved project, update the original project file
            if original_project_file is None:
                self.current_project_file = new_project_file

    def auto_save(self):
        if not hasattr(self, 'current_project_file'):
            reply = QMessageBox.question(self, 'No Project', 
                                         "You need to save the project before auto-saving. Would you like to save now?",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if reply == QMessageBox.Yes:
                self.save_project()
            else:
                return
        
        if hasattr(self, 'current_project_file'):
            self.save_project(show_message=False)
            print("Project auto-saved.")

            
    def load_multi_slice_image(self, image_path, dimensions=None, shape=None):
        
        file_name = os.path.basename(image_path)
        base_name = os.path.splitext(file_name)[0]
        print(f"Loading multi-slice image: {image_path}")
        print(f"Base name: {base_name}")
    
        if dimensions and shape:
            print(f"Using stored dimensions: {dimensions}")
            print(f"Using stored shape: {shape}")
            self.image_dimensions[base_name] = dimensions
            self.image_shapes[base_name] = shape
            if image_path.lower().endswith(('.tif', '.tiff')):
                self.load_tiff(image_path, dimensions, shape)
            elif image_path.lower().endswith('.czi'):
                self.load_czi(image_path, dimensions, shape)
        else:
            print("No stored dimensions or shape, loading as new image")
            if image_path.lower().endswith(('.tif', '.tiff')):
                self.load_tiff(image_path)
            elif image_path.lower().endswith('.czi'):
                self.load_czi(image_path)
    
        print(f"Loaded multi-slice image: {file_name}")
        print(f"Dimensions: {self.image_dimensions.get(base_name, 'Not found')}")
        print(f"Shape: {self.image_shapes.get(base_name, 'Not found')}")
        print(f"Number of slices: {len(self.slices)}")
    
        if self.slices:
            self.current_image = self.slices[0][1]
            self.current_slice = self.slices[0][0]
            
            self.update_slice_list()
            self.slice_list.setCurrentRow(0)
            self.activate_slice(self.current_slice)
            print(f"Activated first slice: {self.current_slice}")
        else:
            print("No slices were loaded")
            self.current_image = None
            self.current_slice = None
    
        self.update_slice_list()
        self.image_label.update()
        
       # print(f"Loaded slices: {[slice_name for slice_name, _ in self.slices]}")
        

            
    def activate_sam_magic_wand(self):
        # Uncheck all other tools
        for button in self.tool_group.buttons():
            if button != self.sam_magic_wand_button:
                button.setChecked(False)
        
        # Set the current tool
        self.image_label.current_tool = "sam_magic_wand"
        self.image_label.sam_magic_wand_active = True
        self.image_label.setCursor(Qt.CrossCursor)
        
        # Update UI based on the current tool
        self.update_ui_for_current_tool()
    
        # If a class is not selected, select the first one (if available)
        if self.current_class is None and self.class_list.count() > 0:
            self.class_list.setCurrentRow(0)
            self.current_class = self.class_list.currentItem().text()
        elif self.class_list.count() == 0:
            QMessageBox.warning(self, "No Class Selected", "Please add a class before using annotation tools.")
            self.sam_magic_wand_button.setChecked(False)
            self.deactivate_sam_magic_wand()
    
    def deactivate_sam_magic_wand(self):
        self.image_label.current_tool = None
        self.image_label.sam_magic_wand_active = False
        self.image_label.setCursor(Qt.ArrowCursor)
        
        # Update UI based on the current tool
        self.update_ui_for_current_tool()
            
    def toggle_sam_assisted(self):
        if not self.current_sam_model:
            QMessageBox.warning(self, "No SAM Model Selected", "Please pick a SAM model before using the SAM-Assisted tool.")
            self.sam_magic_wand_button.setChecked(False)
            return
    
        if self.sam_magic_wand_button.isChecked():
            self.activate_sam_magic_wand()
        else:
            self.deactivate_sam_magic_wand()
    
        self.image_label.clear_temp_sam_prediction()  # Clear temporary prediction
    
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
    

    
    def apply_sam_prediction(self):
        if self.image_label.sam_bbox is None:
            print("SAM bbox is None")
            return
    
        x1, y1, x2, y2 = self.image_label.sam_bbox
        bbox = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
        print(f"Applying SAM prediction with bbox: {bbox}")
    
        prediction = self.sam_utils.apply_sam_prediction(self.current_image, bbox)
    
        if prediction:
            temp_annotation = {
                "segmentation": prediction["segmentation"],
                "category_id": self.class_mapping[self.current_class],
                "category_name": self.current_class,
                "score": prediction["score"]
            }
    
            self.image_label.temp_sam_prediction = temp_annotation
            self.image_label.update()
        else:
            print("Failed to generate prediction")
    
        # Reset SAM bounding box
        self.image_label.sam_bbox = None
        self.image_label.update()
    
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
        first_added_item = None
        for file_name in file_names:
            base_name = os.path.basename(file_name)
            if base_name not in self.image_paths:
                image_info = {
                    "file_name": base_name,
                    "height": 0,
                    "width": 0,
                    "id": len(self.all_images) + 1,
                    "is_multi_slice": False
                }
                
                # Detect multi-slice images and set dimensions
                if file_name.lower().endswith(('.tif', '.tiff', '.czi')):
                    self.load_multi_slice_image(file_name)
                    base_name_without_ext = os.path.splitext(base_name)[0]
                    if base_name_without_ext in self.image_slices and self.image_slices[base_name_without_ext]:
                        first_slice_name, first_slice = self.image_slices[base_name_without_ext][0]
                        image_info["height"] = first_slice.height()
                        image_info["width"] = first_slice.width()
                        image_info["is_multi_slice"] = True
                        image_info["dimensions"] = self.image_dimensions.get(base_name_without_ext, [])
                        image_info["shape"] = self.image_shapes.get(base_name_without_ext, [])
                else:
                    # For regular images
                    image = QImage(file_name)
                    image_info["height"] = image.height()
                    image_info["width"] = image.width()
                
                self.all_images.append(image_info)
                item = QListWidgetItem(base_name)
                self.image_list.addItem(item)
                if first_added_item is None:
                    first_added_item = item
                
                # Update image_paths with the original file path
                self.image_paths[base_name] = file_name
    
        if first_added_item:
            self.image_list.setCurrentItem(first_added_item)
            self.switch_image(first_added_item)
        self.auto_save()  # Auto-save after adding images
    

    def update_all_images(self, new_image_info):
        for info in new_image_info:
            if not any(img['file_name'] == info['file_name'] for img in self.all_images):
                self.all_images.append(info)

    def closeEvent(self, event):
        if not self.image_label.check_unsaved_changes():
            event.ignore()
            return
        event.accept()

        if self.image_label.temp_paint_mask is not None or self.image_label.temp_eraser_mask is not None:
            reply = QMessageBox.question(self, 'Unsaved Changes',
                                         "You have unsaved changes. Do you want to save them before closing?",
                                         QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
            if reply == QMessageBox.Yes:
                if self.image_label.temp_paint_mask is not None:
                    self.image_label.commit_paint_annotation()
                if self.image_label.temp_eraser_mask is not None:
                    self.image_label.commit_eraser_changes()
            elif reply == QMessageBox.Cancel:
                event.ignore()
                return
    
        # Perform any other cleanup or saving operations here
        event.accept()

            
    def switch_slice(self, item):
        if item is None:
            return
        if not self.image_label.check_unsaved_changes():
            return
        
        # Check for unsaved changes
        if self.image_label.temp_paint_mask is not None or self.image_label.temp_eraser_mask is not None:
            reply = QMessageBox.question(self, 'Unsaved Changes',
                                         "You have unsaved changes. Do you want to save them?",
                                         QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
            if reply == QMessageBox.Yes:
                if self.image_label.temp_paint_mask is not None:
                    self.image_label.commit_paint_annotation()
                if self.image_label.temp_eraser_mask is not None:
                    self.image_label.commit_eraser_changes()
            elif reply == QMessageBox.Cancel:
                return
            else:
                self.image_label.discard_paint_annotation()
                self.image_label.discard_eraser_changes()
    
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
                self.image_label.highlighted_annotations.clear()  # Add this line
                self.image_label.reset_annotation_state()
                self.image_label.clear_current_annotation()
                self.update_image_info()
                break
    
        # Ensure the UI is updated
        self.image_label.update()
        self.update_slice_list_colors()
        
        # Reset zoom level to default (1.0)
        self.set_zoom(1.0)


    def switch_image(self, item):
        if item is None:
            return
        if not self.image_label.check_unsaved_changes():    
            return
        
        # Check for unsaved changes
        if self.image_label.temp_paint_mask is not None or self.image_label.temp_eraser_mask is not None:
            reply = QMessageBox.question(self, 'Unsaved Changes',
                                         "You have unsaved changes. Do you want to save them?",
                                         QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
            if reply == QMessageBox.Yes:
                if self.image_label.temp_paint_mask is not None:
                    self.image_label.commit_paint_annotation()
                if self.image_label.temp_eraser_mask is not None:
                    self.image_label.commit_eraser_changes()
            elif reply == QMessageBox.Cancel:
                return
            else:
                self.image_label.discard_paint_annotation()
                self.image_label.discard_eraser_changes()
    
        self.save_current_annotations()
        self.image_label.clear_temp_sam_prediction()
        self.image_label.exit_editing_mode()
    
        file_name = item.text()
        print(f"\nSwitching to image: {file_name}")
    
        image_info = next((img for img in self.all_images if img["file_name"] == file_name), None)
        
        if image_info:
            self.image_file_name = file_name
            image_path = self.image_paths.get(file_name)
            
            if not image_path:
                image_path = os.path.join(self.current_project_dir, "images", file_name)
    
            if image_path and os.path.exists(image_path):
                if image_info.get('is_multi_slice', False):
                    base_name = os.path.splitext(file_name)[0]
                    if base_name in self.image_slices:
                        self.slices = self.image_slices[base_name]
                        if self.slices:
                            self.current_image = self.slices[0][1]
                            self.current_slice = self.slices[0][0]
                            self.update_slice_list()
                            self.activate_slice(self.current_slice)
                    else:
                        self.load_multi_slice_image(image_path, image_info.get('dimensions'), image_info.get('shape'))
                else:
                    self.load_regular_image(image_path)
                    self.display_image()
                    self.clear_slice_list()
                
                self.load_image_annotations()
                self.update_annotation_list()
                self.clear_highlighted_annotation()
                
                self.image_label.highlighted_annotations.clear()  
                self.update_annotation_list()
                self.image_label.update()
                self.image_label.reset_annotation_state()
                self.image_label.clear_current_annotation()
                self.update_image_info()
            else:
                self.current_image = None
                self.image_label.clear()
                self.load_image_annotations()
                self.update_annotation_list()
                self.update_image_info()
            
            self.image_list.setCurrentItem(item)
            self.set_zoom(1.0)
            self.image_label.update()
            self.update_slice_list_colors()
        else:
            self.current_image = None
            self.current_slice = None
            self.image_label.clear()
            self.update_image_info()
            self.clear_slice_list()
        
            
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



    def load_tiff(self, image_path, dimensions=None, shape=None, force_dimension_dialog=False):
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
    
        if dimensions and shape and not force_dimension_dialog:
            # Use stored dimensions and shape
            print(f"Using stored dimensions: {dimensions}")
            print(f"Using stored shape: {shape}")
            image_array = image_array.reshape(shape)
        else:
            # Process as before for new images or when forcing dimension dialog
            print("Processing as new image or forcing dimension dialog.")
            dimensions = None
    
        self.process_multidimensional_image(image_array, image_path, dimensions, force_dimension_dialog)
    
    def load_czi(self, image_path, dimensions=None, shape=None, force_dimension_dialog=False):
        print(f"Loading CZI file: {image_path}")
        with CziFile(image_path) as czi:
            image_array = czi.asarray()
            print(f"CZI array shape: {image_array.shape}")
            print(f"CZI array dtype: {image_array.dtype}")
            print(f"CZI array min: {image_array.min()}, max: {image_array.max()}")
            

    
        if dimensions and shape and not force_dimension_dialog:
            # Use stored dimensions and shape
            print(f"Using stored dimensions: {dimensions}")
            print(f"Using stored shape: {shape}")
            image_array = image_array.reshape(shape)
        else:
            # Process as before for new images or when forcing dimension dialog
            print("Processing as new image or forcing dimension dialog.")
            dimensions = None
    
        self.process_multidimensional_image(image_array, image_path, dimensions, force_dimension_dialog)
        
    
    def load_regular_image(self, image_path):
        self.current_image = QImage(image_path)
        self.slices = []
        self.slice_list.clear()
        self.current_slice = None
    
    def process_multidimensional_image(self, image_array, image_path, dimensions=None, force_dimension_dialog=False):
        file_name = os.path.basename(image_path)
        base_name = os.path.splitext(file_name)[0]
        print(f"Processing file: {file_name}")
        print(f"Image array shape: {image_array.shape}")
        print(f"Image array dtype: {image_array.dtype}")
    
        if dimensions is None or force_dimension_dialog:
            if image_array.ndim > 2:
                default_dimensions = ['Z', 'H', 'W'] if image_array.ndim == 3 else ['T', 'Z', 'H', 'W']
                default_dimensions = default_dimensions[-image_array.ndim:]
                
                # Show a progress dialog
                progress = QProgressDialog("Assigning dimensions...", "Cancel", 0, 100, self)
                progress.setWindowModality(Qt.WindowModal)
                progress.setMinimumDuration(0)
                progress.setValue(10)
                QApplication.processEvents()
    
                while True:
                    dialog = DimensionDialog(image_array.shape, file_name, self, default_dimensions)
                    dialog.setWindowFlags(dialog.windowFlags() & ~Qt.WindowContextHelpButtonHint)
                    progress.setValue(50)
                    QApplication.processEvents()
                    if dialog.exec_():
                        dimensions = dialog.get_dimensions()
                        print(f"Assigned dimensions: {dimensions}")
                        if 'H' in dimensions and 'W' in dimensions:
                            self.image_dimensions[base_name] = dimensions
                            break
                        else:
                            QMessageBox.warning(self, "Invalid Dimensions", "You must assign both H and W dimensions.")
                    else:
                        progress.close()
                        return
                progress.setValue(100)
                progress.close()
            else:
                dimensions = ['H', 'W']
                self.image_dimensions[base_name] = dimensions
    
        self.image_shapes[base_name] = image_array.shape
        print(f"Final assigned dimensions: {self.image_dimensions[base_name]}")
        print(f"Image shape: {self.image_shapes[base_name]}")
    
        if self.image_dimensions[base_name]:
            self.create_slices(image_array, self.image_dimensions[base_name], image_path)
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
    
        # Update UI
        self.update_slice_list()
        self.update_annotation_list()
        self.image_label.update()

    
    def create_slices(self, image_array, dimensions, image_path):
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        slices = []
        self.slice_list.clear()
    
        print(f"Creating slices for {base_name}")
        print(f"Dimensions: {dimensions}")
        print(f"Image array shape: {image_array.shape}")
    
        # Create and show progress dialog
        progress = QProgressDialog("Loading slices...", "Cancel", 0, 100, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)  # Show immediately
    
        # Handle 2D images
        if image_array.ndim == 2:
            progress.setValue(50)  # Update progress
            QApplication.processEvents()  # Allow GUI to update
            normalized_array = self.normalize_array(image_array)
            qimage = self.array_to_qimage(normalized_array)
            slice_name = f"{base_name}"
            slices.append((slice_name, qimage))
            self.add_slice_to_list(slice_name)
        else:
            # For 3D or higher dimensional arrays
            slice_indices = [i for i, dim in enumerate(dimensions) if dim not in ['H', 'W']]
    
            total_slices = np.prod([image_array.shape[i] for i in slice_indices])
            for idx, _ in enumerate(np.ndindex(tuple(image_array.shape[i] for i in slice_indices))):
                if progress.wasCanceled():
                    break
    
                full_idx = [slice(None)] * len(dimensions)
                for i, val in zip(slice_indices, _):
                    full_idx[i] = val
                
                slice_array = image_array[tuple(full_idx)]
                rgb_slice = self.convert_to_8bit_rgb(slice_array)
                qimage = self.array_to_qimage(rgb_slice)
                
                slice_name = f"{base_name}_{'_'.join([f'{dimensions[i]}{val+1}' for i, val in zip(slice_indices, _)])}"
                slices.append((slice_name, qimage))
                
                self.add_slice_to_list(slice_name)
    
                # Update progress
                progress_value = int((idx + 1) / total_slices * 100)
                progress.setValue(progress_value)
                QApplication.processEvents()  # Allow GUI to update
    
        progress.setValue(100)  # Ensure progress reaches 100%
    
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
    
        print(f"Created {len(slices)} slices for {base_name}")
        return slices



    def add_slice_to_list(self, slice_name):
        item = QListWidgetItem(slice_name)
        
        if self.dark_mode:
            # Dark mode
            item.setBackground(QColor(40, 40, 40))  # Very dark gray background for all items
            if slice_name in self.all_annotations:
                item.setForeground(QColor(60, 60, 60))  # Dark gray text
                item.setBackground(QColor(173, 216, 230))  # Light blue background
            else:
                item.setForeground(QColor(200, 200, 200))  # Light gray text
        else:
            # Light mode
            item.setBackground(QColor(240, 240, 240))  # Very light gray background for all items
            if slice_name in self.all_annotations:
                item.setForeground(QColor(255, 255, 255))  # White text
                item.setBackground(QColor(70, 130, 180))  # Medium-dark blue background
            else:
                item.setForeground(QColor(0, 0, 0))  # Black text
        
        self.slice_list.addItem(item)




    
    def normalize_array(self, array):
       # print(f"Normalizing array. Shape: {array.shape}, dtype: {array.dtype}")
       # print(f"Array min: {array.min()}, max: {array.max()}, mean: {array.mean()}")
        
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
        self.current_slice = slice_name
        self.image_file_name = slice_name
        self.load_image_annotations()
        self.update_annotation_list()
        
        for name, qimage in self.slices:
            if name == slice_name:
                self.current_image = qimage
                self.display_image()
                break
        
        self.image_label.update()
        
        items = self.slice_list.findItems(slice_name, Qt.MatchExactly)
        if items:
            self.slice_list.setCurrentItem(items[0])

    
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
        for slice_name, _ in self.slices:
            item = QListWidgetItem(slice_name)
            if slice_name in self.all_annotations:
                item.setForeground(QColor(Qt.green))
            else:
                item.setForeground(QColor(Qt.black) if not self.dark_mode else QColor(Qt.white))
            self.slice_list.addItem(item)
        
        # Select the current slice
        if self.current_slice:
            items = self.slice_list.findItems(self.current_slice, Qt.MatchExactly)
            if items:
                self.slice_list.setCurrentItem(items[0])
                
    def clear_slice_list(self):
        self.slice_list.clear()
        self.slices = []
        self.current_slice = None
    
    def reset_tool_buttons(self):
        for button in self.tool_group.buttons():
            button.setChecked(False)                      

    def keyPressEvent(self, event):
        # Check if the current focus is on a text editing widget
        focused_widget = QApplication.focusWidget()
        if isinstance(focused_widget, (QLineEdit, QTextEdit)):
            super().keyPressEvent(event)
            return

        if event.key() == Qt.Key_F2:
            # Launch the secret Snake game
            self.launch_snake_game()
        elif event.key() == Qt.Key_Delete:
            # Handle deletions
            if self.class_list.hasFocus() and self.class_list.currentItem():
                self.delete_class()
            elif self.annotation_list.hasFocus() and self.annotation_list.selectedItems():
                self.delete_selected_annotations()
            elif self.image_list.hasFocus() and self.image_list.currentItem():
                self.delete_selected_image()
        elif event.key() == Qt.Key_Up or event.key() == Qt.Key_Down:
            # Handle slice navigation
            if self.slice_list.hasFocus():
                current_row = self.slice_list.currentRow()
                if event.key() == Qt.Key_Up and current_row > 0:
                    self.slice_list.setCurrentRow(current_row - 1)
                elif event.key() == Qt.Key_Down and current_row < self.slice_list.count() - 1:
                    self.slice_list.setCurrentRow(current_row + 1)
                self.switch_slice(self.slice_list.currentItem())
            else:
                # Pass the event to the parent for default handling
                super().keyPressEvent(event)
        else:
            # Pass any other key events to the parent for default handling
            super().keyPressEvent(event)
     
    def launch_snake_game(self):
        self.snake_game = SnakeGame()
        self.snake_game.show()
        
    def import_annotations(self):
        if not self.image_label.check_unsaved_changes():    
            return
        print("Starting import_annotations")
        import_format = self.import_format_selector.currentText()
        print(f"Import format: {import_format}")
    
        if import_format == "COCO JSON":
            file_name, _ = QFileDialog.getOpenFileName(self, "Import COCO JSON Annotations", "", "JSON Files (*.json)")
            if not file_name:
                print("No file selected, returning")
                return
            
            print(f"Selected file: {file_name}")
            json_dir = os.path.dirname(file_name)
            images_dir = os.path.join(json_dir, 'images')
            imported_annotations, image_info = import_coco_json(file_name, self.class_mapping)
        
        elif import_format == "YOLO v8":
            directory = QFileDialog.getExistingDirectory(self, "Select YOLO v8 Dataset Directory")
            if not directory:
                print("No directory selected, returning")
                return
            
            print(f"Selected directory: {directory}")
            images_dir = os.path.join(directory, 'images')
            try:
                imported_annotations, image_info = process_import_format(import_format, directory, self.class_mapping)
            except ValueError as e:
                QMessageBox.warning(self, "Import Error", str(e))
                return
        
        else:
            QMessageBox.warning(self, "Unsupported Format", f"The selected format '{import_format}' is not implemented for import.")
            return
    
        print(f"JSON/YOLO directory: {json_dir if import_format == 'COCO JSON' else directory}")
        print(f"Images directory: {images_dir}")
        print(f"Imported annotations count: {len(imported_annotations)}")
        print(f"Image info count: {len(image_info)}")
        
        images_loaded = 0
        images_not_found = []
    
        for info in image_info.values():
            print(f"Processing image: {info['file_name']}")
            image_path = os.path.join(images_dir, info['file_name'])
    
            if os.path.exists(image_path):
                print(f"Image found at: {image_path}")
                self.image_paths[info['file_name']] = image_path
                self.all_images.append({
                    "file_name": info['file_name'],
                    "height": info['height'],
                    "width": info['width'],
                    "id": info['id'],
                    "is_multi_slice": False
                })
                images_loaded += 1
            else:
                print(f"Image not found at: {image_path}")
                images_not_found.append(info['file_name'])
    
        print(f"Images loaded: {images_loaded}")
        print(f"Images not found: {len(images_not_found)}")
    
        if images_not_found:
            message = f"The following {len(images_not_found)} images were not found in the 'images' directory:\n\n"
            message += "\n".join(images_not_found[:10])
            if len(images_not_found) > 10:
                message += f"\n... and {len(images_not_found) - 10} more."
            message += "\n\nDo you want to proceed and ignore annotations for these missing images?"
            reply = QMessageBox.question(self, "Missing Images", message, 
                                          QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            
            if reply == QMessageBox.No:
                print("Import cancelled due to missing images")
                QMessageBox.information(self, "Import Cancelled", 
                                        "Import cancelled. Please ensure all images are in the 'images' directory and try again.")
                return
    
        # Update annotations (only for found images)
        for image_name, annotations in imported_annotations.items():
            if image_name not in self.image_paths:
                continue
            self.all_annotations[image_name] = {}
            for category_name, category_annotations in annotations.items():
                self.all_annotations[image_name][category_name] = []
                for i, ann in enumerate(category_annotations, start=1):
                    new_ann = {
                        "segmentation": ann["segmentation"],
                        "category_id": ann["category_id"],
                        "category_name": category_name,
                        "number": i,
                        "type": "polygon"  # Assuming all imported annotations are polygons
                    }
                    self.all_annotations[image_name][category_name].append(new_ann)
    
        # Update class mapping and colors
        for annotations in self.all_annotations.values():
            for category_name in annotations.keys():
                if category_name not in self.class_mapping:
                    new_id = len(self.class_mapping) + 1
                    self.class_mapping[category_name] = new_id
                    self.image_label.class_colors[category_name] = QColor(Qt.GlobalColor(new_id % 16 + 7))
    
        print("Updating UI")
        # Update UI
        self.update_class_list()
        self.update_image_list()
        self.update_annotation_list()
    
        # Highlight and display the first image
        if self.image_list.count() > 0:
            self.image_list.setCurrentRow(0)
            self.switch_image(self.image_list.item(0))
    
        self.image_label.update()
    
        message = f"Annotations have been imported successfully from {file_name if import_format == 'COCO JSON' else directory}.\n"
        message += f"{images_loaded} images were loaded from the 'images' directory.\n"
        if images_not_found:
            message += f"Annotations for {len(images_not_found)} missing images were ignored."
    
        print("Import complete, showing message")
        QMessageBox.information(self, "Import Complete", message)
        self.auto_save()  # Auto-save after importing annotations
    
    
    
    def export_annotations(self):
        if not self.image_label.check_unsaved_changes():    
            return
        export_format = self.export_format_selector.currentText()
        
        supported_formats = [
            "COCO JSON", "YOLO v8", "Labeled Images", 
            "Semantic Labels", "Pascal VOC (BBox)", "Pascal VOC (BBox + Segmentation)"
        ]
        
        if export_format not in supported_formats:
            QMessageBox.warning(self, "Unsupported Format", f"The selected format '{export_format}' is not implemented.")
            return
    
        if export_format == "COCO JSON":
            file_name, _ = QFileDialog.getSaveFileName(self, "Export COCO JSON Annotations", "", "JSON Files (*.json)")
        else:
            file_name = QFileDialog.getExistingDirectory(self, f"Select Output Directory for {export_format} Export")
    
        if not file_name:
            return
    
        self.save_current_annotations()
    
        if export_format == "COCO JSON":
            output_dir = os.path.dirname(file_name)
            json_filename = os.path.basename(file_name)
            json_file, images_dir = export_coco_json(self.all_annotations, self.class_mapping, 
                                                     self.image_paths, self.slices, self.image_slices, 
                                                     output_dir, json_filename)
            message = f"Annotations have been exported successfully in COCO JSON format.\n"
            message += f"JSON file: {json_file}\nImages directory: {images_dir}"
        
        elif export_format == "YOLO v8":
            labels_dir, yaml_path = export_yolo_v8(self.all_annotations, self.class_mapping, self.image_paths, self.slices, self.image_slices, file_name)
            message = f"Annotations have been exported successfully in YOLO v8 format.\nLabels: {labels_dir}\nYAML: {yaml_path}"
        
        elif export_format == "Labeled Images":
            labeled_images_dir = export_labeled_images(self.all_annotations, self.class_mapping, self.image_paths, self.slices, self.image_slices, file_name)
            message = f"Labeled images have been exported successfully.\nLabeled Images: {labeled_images_dir}\n"
            message += f"A class summary has been saved in: {os.path.join(labeled_images_dir, 'class_summary.txt')}"
        
        elif export_format == "Semantic Labels":
            semantic_labels_dir = export_semantic_labels(self.all_annotations, self.class_mapping, self.image_paths, self.slices, self.image_slices, file_name)
            message = f"Semantic labels have been exported successfully.\nSemantic Labels: {semantic_labels_dir}\n"
            message += f"A class-pixel mapping has been saved in: {os.path.join(semantic_labels_dir, 'class_pixel_mapping.txt')}"
        
        elif export_format == "Pascal VOC (BBox)":
            voc_dir = export_pascal_voc_bbox(self.all_annotations, self.class_mapping, self.image_paths, self.slices, self.image_slices, file_name)
            message = f"Annotations have been exported successfully in Pascal VOC format (BBox only).\nPascal VOC Annotations: {voc_dir}"
        
        elif export_format == "Pascal VOC (BBox + Segmentation)":
            voc_dir = export_pascal_voc_both(self.all_annotations, self.class_mapping, self.image_paths, self.slices, self.image_slices, file_name)
            message = f"Annotations have been exported successfully in Pascal VOC format (BBox + Segmentation).\nPascal VOC Annotations: {voc_dir}"
    
        QMessageBox.information(self, "Export Complete", message)
    

    def save_slices(self, directory):
        slices_saved = False
        for image_file, image_slices in self.image_slices.items():
            for slice_name, qimage in image_slices:
                if slice_name in self.all_annotations and self.all_annotations[slice_name]:
                    file_path = os.path.join(directory, f"{slice_name}.png")
                    qimage.save(file_path, "PNG")
                    slices_saved = True
        
        return slices_saved


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
            for annotation in class_annotations:
                number = annotation.get('number', 0)
                item_text = f"{class_name} - {number}"
                item = QListWidgetItem(item_text)
                item.setData(Qt.UserRole, annotation)
                item.setForeground(color)
                self.annotation_list.addItem(item)
        
        # Force the annotation list to repaint
        self.annotation_list.repaint()
        
        #print(f"Annotation list updated for {current_name}. Current annotations: {annotations}")
                
    
    
    def update_slice_list_colors(self):
        # Set the background color of the entire list widget
        if self.dark_mode:
            self.slice_list.setStyleSheet("QListWidget { background-color: rgb(40, 40, 40); }")
        else:
            self.slice_list.setStyleSheet("QListWidget { background-color: rgb(240, 240, 240); }")
    
        for i in range(self.slice_list.count()):
            item = self.slice_list.item(i)
            slice_name = item.text()
            
            if self.dark_mode:
                # Dark mode
                if slice_name in self.all_annotations and any(self.all_annotations[slice_name].values()):
                    item.setForeground(QColor(60, 60, 60))  # Dark gray text
                    item.setBackground(QColor(173, 216, 230))  # Light blue background
                else:
                    item.setForeground(QColor(200, 200, 200))  # Light gray text
                    item.setBackground(QColor(40, 40, 40))  # Very dark gray background
            else:
                # Light mode
                if slice_name in self.all_annotations and any(self.all_annotations[slice_name].values()):
                    item.setForeground(QColor(255, 255, 255))  # White text
                    item.setBackground(QColor(70, 130, 180))  # Medium-dark blue background
                else:
                    item.setForeground(QColor(0, 0, 0))  # Black text
                    item.setBackground(QColor(240, 240, 240))  # Very light gray background
    
        # Force the list to repaint
        self.slice_list.repaint()
                

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
        #print(f"Current name for annotations: {current_name}")
        #print(f"All annotations keys: {list(self.all_annotations.keys())}")
        if current_name in self.all_annotations:
            self.image_label.annotations = copy.deepcopy(self.all_annotations[current_name])
            #print(f"Loaded annotations: {self.image_label.annotations}")
        else:
            print(f"No annotations found for {current_name}")
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



    def setup_tool_buttons(self):
        """Set up the tool buttons with grouped manual and automated tools."""
        self.tool_group = QButtonGroup(self)
        self.tool_group.setExclusive(False)
    
        # Create a widget for manual tools
        manual_tools_widget = QWidget()
        manual_layout = QVBoxLayout(manual_tools_widget)
        manual_layout.setSpacing(5)
    
        manual_label = QLabel("Manual Tools")
        manual_label.setAlignment(Qt.AlignCenter)
        manual_layout.addWidget(manual_label)
    
        manual_buttons_layout = QHBoxLayout()
        self.polygon_button = QPushButton("Polygon")
        self.polygon_button.setCheckable(True)
        self.rectangle_button = QPushButton("Rectangle")
        self.rectangle_button.setCheckable(True)
        manual_buttons_layout.addWidget(self.polygon_button)
        manual_buttons_layout.addWidget(self.rectangle_button)
        manual_layout.addLayout(manual_buttons_layout)
    
        self.tool_group.addButton(self.polygon_button)
        self.tool_group.addButton(self.rectangle_button)
        self.polygon_button.clicked.connect(self.toggle_tool)
        self.rectangle_button.clicked.connect(self.toggle_tool)
    
        # Create a widget for automated tools
        automated_tools_widget = QWidget()
        automated_layout = QVBoxLayout(automated_tools_widget)
        automated_layout.setSpacing(5)
    
        automated_label = QLabel("Automated Tools")
        automated_label.setAlignment(Qt.AlignCenter)
        automated_layout.addWidget(automated_label)
    
        automated_buttons_layout = QHBoxLayout()
        self.sam_magic_wand_button = QPushButton("Magic Wand")
        self.sam_magic_wand_button.setCheckable(True)
        automated_buttons_layout.addWidget(self.sam_magic_wand_button)
        automated_layout.addLayout(automated_buttons_layout)
    
        self.tool_group.addButton(self.sam_magic_wand_button)
        self.sam_magic_wand_button.clicked.connect(self.toggle_tool)
    
        # Add the grouped tools to the sidebar layout
        self.sidebar_layout.addWidget(manual_tools_widget)
        self.sidebar_layout.addWidget(automated_tools_widget)
    

    
        # Set a fixed size for all buttons to make them smaller
        for button in [self.polygon_button, self.rectangle_button, self.load_sam2_button, self.sam_magic_wand_button]:
            button.setFixedSize(100, 30)  

    def setup_annotation_list(self):
        """Set up the annotation list widget."""
        self.annotation_list = QListWidget()
        self.annotation_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.annotation_list.itemSelectionChanged.connect(self.update_highlighted_annotations)
        
        
            
    
    def create_menu_bar(self):
        menu_bar = self.menuBar()
    
        # Project Menu
        project_menu = menu_bar.addMenu("&Project")
        
        new_project_action = QAction("&New Project", self)
        new_project_action.setShortcut(QKeySequence.New)
        new_project_action.triggered.connect(self.new_project)
        project_menu.addAction(new_project_action)
    
        open_project_action = QAction("&Open Project", self)
        open_project_action.setShortcut(QKeySequence.Open)
        open_project_action.triggered.connect(self.open_project)
        project_menu.addAction(open_project_action)
    
        save_project_action = QAction("&Save Project", self)
        save_project_action.setShortcut(QKeySequence.Save)
        save_project_action.triggered.connect(self.save_project)
        project_menu.addAction(save_project_action)
        
        save_project_as_action = QAction("Save Project &As...", self)
        save_project_as_action.setShortcut(QKeySequence("Ctrl+Shift+S"))
        save_project_as_action.triggered.connect(self.save_project_as)
        project_menu.addAction(save_project_as_action)
    
        close_project_action = QAction("&Close Project", self)
        close_project_action.setShortcut(QKeySequence("Ctrl+W"))
        close_project_action.triggered.connect(self.close_project)
        project_menu.addAction(close_project_action)
    
        # Settings Menu
        settings_menu = menu_bar.addMenu("&Settings")
        
        font_size_menu = settings_menu.addMenu("&Font Size")
        for size in ["Small", "Medium", "Large", "XL", "XXL"]:
            action = QAction(size, self)
            action.triggered.connect(lambda checked, s=size: self.change_font_size(s))
            font_size_menu.addAction(action)
    
        toggle_dark_mode_action = QAction("Toggle &Dark Mode", self)
        toggle_dark_mode_action.setShortcut(QKeySequence("Ctrl+D"))
        toggle_dark_mode_action.triggered.connect(self.toggle_dark_mode)
        settings_menu.addAction(toggle_dark_mode_action)
        
        # Tools Menu
        tools_menu = menu_bar.addMenu("&Tools")
        
        annotation_stats_action = QAction("Annotation Statistics", self)
        annotation_stats_action.triggered.connect(self.show_annotation_statistics)
        annotation_stats_action.setShortcut(QKeySequence("Ctrl+Alt+S"))
        tools_menu.addAction(annotation_stats_action)      
        
        coco_json_combiner_action = QAction("COCO JSON Combiner", self)
        coco_json_combiner_action.triggered.connect(self.show_coco_json_combiner)
        tools_menu.addAction(coco_json_combiner_action)
    
        dataset_splitter_action = QAction("Dataset Splitter", self)
        dataset_splitter_action.triggered.connect(self.open_dataset_splitter)
        tools_menu.addAction(dataset_splitter_action)     
        
        stack_to_slices_action = QAction("Stack to Slices", self)
        stack_to_slices_action.triggered.connect(self.show_stack_to_slices)
        tools_menu.addAction(stack_to_slices_action)
        
        image_patcher_action = QAction("Image Patcher", self)
        image_patcher_action.triggered.connect(self.show_image_patcher)
        tools_menu.addAction(image_patcher_action)
        
        image_augmenter_action = QAction("Image Augmenter", self)
        image_augmenter_action.triggered.connect(self.show_image_augmenter)
        tools_menu.addAction(image_augmenter_action)
    
        # Help Menu
        help_menu = menu_bar.addMenu("&Help")
    
        help_action = QAction("&Show Help", self)
        help_action.setShortcut(QKeySequence.HelpContents)
        help_action.triggered.connect(self.show_help)
        help_menu.addAction(help_action)
        
        
    

    def change_font_size(self, size):
        self.current_font_size = size
        self.apply_theme_and_font()

    def setup_sidebar(self):
        self.sidebar = QWidget()
        self.sidebar_layout = QVBoxLayout(self.sidebar)
        self.layout.addWidget(self.sidebar, 1)
        
        # Helper function to create section headers
        def create_section_header(text):
            label = QLabel(text)
            label.setProperty("class", "section-header")
            label.setAlignment(Qt.AlignLeft)
            return label
    
        # New code for import functionality
        self.import_button = QPushButton("Import Annotations with Images")
        self.import_button.clicked.connect(self.import_annotations)
        self.sidebar_layout.addWidget(self.import_button)
        
        self.import_format_selector = QComboBox()
        self.import_format_selector.addItem("COCO JSON")
        self.import_format_selector.addItem("YOLO v8")
        self.sidebar_layout.addWidget(self.import_format_selector)
        
        # Add spacing
        self.sidebar_layout.addSpacing(20) 
    
        self.add_images_button = QPushButton("Add New Images")
        self.add_images_button.clicked.connect(self.add_images)
        self.sidebar_layout.addWidget(self.add_images_button)
        
        self.add_class_button = QPushButton("Add Classes")
        self.add_class_button.clicked.connect(lambda: self.add_class())
        self.sidebar_layout.addWidget(self.add_class_button)
        
        # Class list (without the "Classes" header)
        self.class_list = QListWidget()
        self.class_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.class_list.customContextMenuRequested.connect(self.show_class_context_menu)
        self.class_list.itemClicked.connect(self.on_class_selected)
        self.sidebar_layout.addWidget(self.class_list)
        
        # Annotation section
        self.sidebar_layout.addWidget(create_section_header("Annotation"))
        annotation_widget = QWidget()
        annotation_layout = QVBoxLayout(annotation_widget)
    
        # Manual tools subsection
        manual_widget = QWidget()
        manual_layout = QVBoxLayout(manual_widget)
    
        button_layout_top = QHBoxLayout()
        self.polygon_button = QPushButton("Polygon")
        self.polygon_button.setCheckable(True)
        self.rectangle_button = QPushButton("Rectangle")
        self.rectangle_button.setCheckable(True)
        button_layout_top.addWidget(self.polygon_button)
        button_layout_top.addWidget(self.rectangle_button)
    
        button_layout_bottom = QHBoxLayout()
        self.paint_brush_button = QPushButton("Paint Brush")
        self.paint_brush_button.setCheckable(True)
        self.eraser_button = QPushButton("Eraser")
        self.eraser_button.setCheckable(True)
        button_layout_bottom.addWidget(self.paint_brush_button)
        button_layout_bottom.addWidget(self.eraser_button)
    
        manual_layout.addLayout(button_layout_top)
        manual_layout.addLayout(button_layout_bottom)
    
        annotation_layout.addWidget(manual_widget)

        # SAM-Assisted tools subsection
        sam_widget = QWidget()
        sam_layout = QVBoxLayout(sam_widget)
    
        # SAM-Assisted button on top
        self.sam_magic_wand_button = QPushButton("SAM-Assisted")
        self.sam_magic_wand_button.setCheckable(True)
        self.sam_magic_wand_button.clicked.connect(self.toggle_sam_assisted)
        sam_layout.addWidget(self.sam_magic_wand_button)
    
        # Add SAM model selector
        self.sam_model_selector = QComboBox()
        self.sam_model_selector.addItem("Pick a SAM Model")
        self.sam_model_selector.addItems(list(self.sam_utils.sam_models.keys()))
        self.sam_model_selector.currentTextChanged.connect(self.change_sam_model)
        sam_layout.addWidget(self.sam_model_selector)
    
        annotation_layout.addWidget(sam_widget)
    
        # Setup tool group
        self.tool_group = QButtonGroup(self)
        self.tool_group.setExclusive(False)
        self.tool_group.addButton(self.polygon_button)
        self.tool_group.addButton(self.rectangle_button)
        self.tool_group.addButton(self.paint_brush_button)
        self.tool_group.addButton(self.eraser_button)
        self.tool_group.addButton(self.sam_magic_wand_button)
    
        self.polygon_button.clicked.connect(self.toggle_tool)
        self.rectangle_button.clicked.connect(self.toggle_tool)
        self.paint_brush_button.clicked.connect(self.toggle_tool)
        self.eraser_button.clicked.connect(self.toggle_tool)
        self.sam_magic_wand_button.clicked.connect(self.toggle_tool)
    
        # Annotations list subsection
        annotation_layout.addWidget(QLabel("Annotations"))
        self.annotation_list = QListWidget()
        self.annotation_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.annotation_list.itemSelectionChanged.connect(self.update_highlighted_annotations)
        annotation_layout.addWidget(self.annotation_list)
        
        #Delete and Merge annotation buttons
        self.delete_button = QPushButton("Delete")
        self.delete_button.clicked.connect(self.delete_selected_annotations)
        self.merge_button = QPushButton("Merge")
        self.merge_button.clicked.connect(self.merge_annotations)
    
        # Create a horizontal layout for the buttons
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.delete_button)
        button_layout.addWidget(self.merge_button)
    
        # Add the button layout to the annotation layout
        annotation_layout.addLayout(button_layout)
    
        # Add export format selector 
        self.export_format_selector = QComboBox()
        self.export_format_selector.addItem("COCO JSON")
        self.export_format_selector.addItem("YOLO v8")
        self.export_format_selector.addItem("Labeled Images")
        self.export_format_selector.addItem("Semantic Labels")
        self.export_format_selector.addItem("Pascal VOC (BBox)")
        self.export_format_selector.addItem("Pascal VOC (BBox + Segmentation)")
        
        annotation_layout.addWidget(QLabel("Export Format:"))
        annotation_layout.addWidget(self.export_format_selector)
    
        self.export_button = QPushButton("Export Annotations")
        self.export_button.clicked.connect(self.export_annotations)
        annotation_layout.addWidget(self.export_button)
    
        # Add the annotation widget to the sidebar
        self.sidebar_layout.addWidget(annotation_widget)
        
    def change_sam_model(self, model_name):
        self.sam_utils.change_sam_model(model_name)
        self.current_sam_model = self.sam_utils.current_sam_model
        
        if model_name != "Pick a SAM Model":
            # Enable the SAM Magic Wand button
            self.sam_magic_wand_button.setEnabled(True)
            
            # Activate the SAM Magic Wand tool
            self.sam_magic_wand_button.setChecked(True)
            self.activate_sam_magic_wand()
            
            print(f"Changed SAM model to: {model_name}")
        else:
            # Disable and deactivate the SAM Magic Wand button
            self.sam_magic_wand_button.setEnabled(False)
            self.sam_magic_wand_button.setChecked(False)
            self.deactivate_sam_magic_wand()
            print("SAM model unset")
        
        
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
        self.update()

        
    def toggle_dark_mode(self):
        self.dark_mode = not self.dark_mode
        self.apply_theme_and_font()
        
        # Update slice list colors
        self.update_slice_list_colors()
        
        # Update other UI elements if necessary
        self.update_class_list()
        self.update_annotation_list()
        
        # Force a repaint of the main window
        self.repaint()
        
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

##########    ### Tools  ########## I love useful image processing tools :)
    def open_dataset_splitter(self):
        self.dataset_splitter = DatasetSplitterTool(self)
        self.dataset_splitter.setWindowModality(Qt.ApplicationModal)
        self.dataset_splitter.show_centered(self)
        
    def show_annotation_statistics(self):
        if not self.all_annotations:
            QMessageBox.warning(self, "No Annotations", "There are no annotations to analyze.")
            return
        try:
            self.annotation_stats_dialog = show_annotation_statistics(self, self.all_annotations)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred while showing annotation statistics: {str(e)}")

            
    def show_coco_json_combiner(self):
        self.coco_json_combiner_dialog = show_coco_json_combiner(self)
        
    def show_stack_to_slices(self):
        self.stack_to_slices_dialog = show_stack_to_slices(self)
        

    def show_image_patcher(self):
        self.image_patcher_dialog = show_image_patcher(self)    
        
    def show_image_augmenter(self):
        self.image_augmenter_dialog = show_image_augmenter(self)

        

###################################################################

    # update the show_help method:
    def show_help(self):
        self.help_window = HelpWindow(dark_mode=self.dark_mode, font_size=self.font_sizes[self.current_font_size])
        self.help_window.show_centered(self)

            
    def add_images(self):
        if not self.image_label.check_unsaved_changes():    
            return
        file_names, _ = QFileDialog.getOpenFileNames(self, "Add Images", "", "Image Files (*.png *.jpg *.bmp *.tif *.tiff *.czi)")
        if file_names:
            self.add_images_to_list(file_names)
            
                
    def clear_all(self, new_project=False, show_messages=True):
        if not new_project and show_messages:
            reply = self.show_question('Clear All',
                                       "Are you sure you want to clear all images and annotations? This action cannot be undone.")
            if reply != QMessageBox.Yes:
                return
    
        # Clear images
        self.image_list.clear()
        self.image_paths.clear()
        self.all_images.clear()
        self.current_image = None
        self.image_file_name = ""
        
        # Clear the image display
        self.image_label.clear()
        self.image_label.setPixmap(QPixmap())  # Set an empty pixmap
        self.image_label.original_pixmap = None
        self.image_label.scaled_pixmap = None
    
        # Clear annotations
        self.all_annotations.clear()
        self.annotation_list.clear()
        self.image_label.annotations.clear()
        self.image_label.highlighted_annotations.clear()
    
        # Clear current class
        self.current_class = None
    
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
        
        # Reset zoom
        self.image_label.zoom_factor = 1.0
        self.zoom_slider.setValue(100)
        
        # Reset tools
        self.image_label.current_tool = None
        self.polygon_button.setChecked(False)
        self.rectangle_button.setChecked(False)
        self.sam_magic_wand_button.setChecked(False)
        self.sam_magic_wand_button.setEnabled(False)  # Disable the SAM-Assisted button
        self.image_label.sam_magic_wand_active = False  # Deactivate SAM magic wand
    
        # Reset SAM-related attributes
        self.image_label.sam_bbox = None
        self.image_label.drawing_sam_bbox = False
        self.image_label.temp_sam_prediction = None
    
        self.image_label.setCursor(Qt.ArrowCursor)  # Reset cursor to default
        self.sam_model_selector.setCurrentIndex(0)  # Reset to "Pick a SAM Model"
        self.current_sam_model = None  # Reset the current SAM model
        
        # Reset project-related attributes
        if not new_project:
            if hasattr(self, 'current_project_file'):
                del self.current_project_file
            if hasattr(self, 'current_project_dir'):
                del self.current_project_dir
        
        # Update UI
        self.image_label.update()
        self.update_image_info()

       
        
        # Force a repaint of the main window
        self.repaint()
        self.update_window_title()
                    
            
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
        redefine_dimensions_action = None
    
        current_item = self.image_list.itemAt(position)
        if current_item:
            file_name = current_item.text()
            file_path = self.image_paths.get(file_name)
            if file_path and file_path.lower().endswith(('.tif', '.tiff', '.czi')):
                redefine_dimensions_action = menu.addAction("Redefine Dimensions")
    
        action = menu.exec_(self.image_list.mapToGlobal(position))
        
        if action == delete_action:
            self.remove_image()
        elif action == redefine_dimensions_action:
            self.redefine_dimensions(file_name)
            
    def redefine_dimensions(self, file_name):
        file_path = self.image_paths.get(file_name)
        if not file_path or not file_path.lower().endswith(('.tif', '.tiff', '.czi')):
            return  # Exit the method if it's not a TIFF or CZI file
    
        reply = QMessageBox.warning(self, "Redefine Dimensions",
                                    "Redefining dimensions will cause all associated annotations to be lost. "
                                    "Do you want to continue?",
                                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            # Remove existing annotations for this file
            base_name = os.path.splitext(file_name)[0]
            
            print(f"Removing annotations for image: {base_name}")
            #print(f"Current annotations: {list(self.all_annotations.keys())}")
            
            # Create a list of keys to remove, using a more specific matching condition
            keys_to_remove = [key for key in self.all_annotations.keys() 
                              if key == base_name or (key.startswith(f"{base_name}_") and not key.startswith(f"{base_name}_8bit"))]
            
            print(f"Keys to remove: {keys_to_remove}")
            
            # Remove the annotations
            for key in keys_to_remove:
                del self.all_annotations[key]
            
            #print(f"Annotations after removal: {list(self.all_annotations.keys())}")
            
            # Remove existing slices
            if base_name in self.image_slices:
                del self.image_slices[base_name]
            
            # Clear current image if it's the one being redefined
            if self.image_file_name == file_name:
                self.current_image = None
                self.image_label.clear()
            
            # Reload the image with new dimension dialog
            if file_path.lower().endswith(('.tif', '.tiff')):
                self.load_tiff(file_path, force_dimension_dialog=True)
            elif file_path.lower().endswith('.czi'):
                self.load_czi(file_path, force_dimension_dialog=True)
            
            # Update UI
            self.update_slice_list()
            self.update_annotation_list()
            self.image_label.update()
            
            #print(f"Final annotations: {list(self.all_annotations.keys())}")
            
            QMessageBox.information(self, "Dimensions Redefined", 
                                    "The dimensions have been redefined and the image reloaded. "
                                    "All previous annotations for this image have been removed.")
    
    def remove_image(self):
        current_item = self.image_list.currentItem()
        if current_item:
            file_name = current_item.text()
            
            # Remove from all data structures
            self.image_list.takeItem(self.image_list.row(current_item))
            self.image_paths.pop(file_name, None)
            self.all_images = [img for img in self.all_images if img["file_name"] != file_name]
            
            # Remove annotations
            self.all_annotations.pop(file_name, None)
            
            # Handle multi-dimensional images
            base_name = os.path.splitext(file_name)[0]
            if base_name in self.image_slices:
                # Remove slices
                for slice_name, _ in self.image_slices[base_name]:
                    self.all_annotations.pop(slice_name, None)
                del self.image_slices[base_name]
                
                # Clear slice list
                self.slice_list.clear()
            
            # Clear current image and slice if it was the removed image
            if self.image_file_name == file_name:
                self.current_image = None
                self.image_file_name = ""
                self.current_slice = None
                self.image_label.clear()
                self.annotation_list.clear()
            
            # Switch to another image if available
            if self.image_list.count() > 0:
                next_item = self.image_list.item(0)
                self.image_list.setCurrentItem(next_item)
                self.switch_image(next_item)
            else:
                # No images left
                self.current_image = None
                self.image_file_name = ""
                self.current_slice = None
                self.image_label.clear()
                self.annotation_list.clear()
                self.slice_list.clear()
            
            # Update UI
            self.update_ui()  
            self.auto_save()  # Auto-save after removing an image


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
                            ann["bbox"] = annotation["bbox"]
                            ann["type"] = "bbox"
                        
                        # Add number field if it's missing
                        if "number" not in ann:
                            ann["number"] = len(self.all_annotations[file_name][category_name]) + 1
                        
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
                
            self.image_label.highlighted_annotations = []  # Clear existing highlights
            self.update_annotation_list()  # This will repopulate the annotation list
            self.image_label.update()  # Force a redraw of the image label




    def clear_highlighted_annotation(self):
        self.image_label.highlighted_annotation = None
        self.image_label.update()
        
    def update_highlighted_annotations(self):
        if not self.image_label.check_unsaved_changes():
            return
        selected_items = self.annotation_list.selectedItems()
        self.image_label.highlighted_annotations = [item.data(Qt.UserRole) for item in selected_items]
        self.image_label.update()  # Force a redraw of the image label
        
        # Enable/disable merge button based on selection
        self.merge_button.setEnabled(len(selected_items) >= 2)

    def renumber_annotations(self):
        current_name = self.current_slice or self.image_file_name
        if current_name in self.all_annotations:
            for class_name, annotations in self.all_annotations[current_name].items():
                for i, ann in enumerate(annotations, start=1):
                    ann['number'] = i
        self.update_annotation_list()

    def delete_selected_annotations(self):
        selected_items = self.annotation_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "No Selection", "Please select an annotation to delete.")
            return
        
        reply = QMessageBox.question(self, 'Delete Annotations',
                                     f"Are you sure you want to delete {len(selected_items)} annotation(s)?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            # Create a list of annotations to remove
            annotations_to_remove = []
            for item in selected_items:
                annotation = item.data(Qt.UserRole)
                annotations_to_remove.append((annotation['category_name'], annotation))
            
            # Remove annotations from image_label.annotations
            for category_name, annotation in annotations_to_remove:
                if category_name in self.image_label.annotations:
                    if annotation in self.image_label.annotations[category_name]:
                        self.image_label.annotations[category_name].remove(annotation)
            
            # Renumber remaining annotations
            self.renumber_annotations()
            
            # Update all_annotations
            current_name = self.current_slice or self.image_file_name
            self.all_annotations[current_name] = self.image_label.annotations
            
            # Clear and repopulate the annotation list
            self.update_annotation_list()
            
            self.image_label.highlighted_annotations.clear()
            self.image_label.update()
            
            # Update slice list colors
            self.update_slice_list_colors()
    
            QMessageBox.information(self, "Annotations Deleted", f"{len(selected_items)} annotation(s) have been deleted.")  
            self.auto_save()  # Auto-save after deleting annotations


    def merge_annotations(self):
        if self.image_label.editing_polygon is not None:
            QMessageBox.warning(self, "Edit Mode Active", 
                                "Please exit the annotation edit mode before merging annotations.")
            return
    
        selected_items = self.annotation_list.selectedItems()
        if len(selected_items) < 2:
            QMessageBox.warning(self, "Not Enough Annotations", "Please select at least two annotations to merge.")
            return
    
        class_name = selected_items[0].data(Qt.UserRole)['category_name']
        if not all(item.data(Qt.UserRole)['category_name'] == class_name for item in selected_items):
            QMessageBox.warning(self, "Mixed Classes", "All selected annotations must be from the same class.")
            return
    
        polygons = []
        original_annotations = []
        for item in selected_items:
            annotation = item.data(Qt.UserRole)
            original_annotations.append(annotation)
            if 'segmentation' in annotation:
                points = zip(annotation['segmentation'][0::2], annotation['segmentation'][1::2])
                polygon = Polygon(points)
                if not polygon.is_valid:
                    polygon = polygon.buffer(0)
                polygons.append(polygon)
    
        def are_all_polygons_connected(polygons):
            if len(polygons) < 2:
                return True
            
            connected = set([0])  # Start with the first polygon
            to_check = set(range(1, len(polygons)))
            
            while to_check:
                newly_connected = set()
                for i in connected:
                    for j in to_check:
                        if polygons[i].intersects(polygons[j]) or polygons[i].touches(polygons[j]):
                            newly_connected.add(j)
                
                if not newly_connected:
                    return False  # If no new connections found, they're not all connected
                
                connected.update(newly_connected)
                to_check -= newly_connected
            
            return True  # All polygons are connected
    
        if not are_all_polygons_connected(polygons):
            QMessageBox.warning(self, "Disconnected Polygons", "Not all selected annotations are connected. Please select only connected annotations to merge.")
            return
    
        try:
            merged_polygon = unary_union(polygons)
        except Exception as e:
            QMessageBox.warning(self, "Merge Error", f"Unable to merge the selected annotations due to an error: {str(e)}")
            return
    
        new_annotation = {
            "segmentation": [],
            "category_id": self.class_mapping[class_name],
            "category_name": class_name,
        }
    
        if isinstance(merged_polygon, Polygon):
            new_annotation["segmentation"] = [coord for point in merged_polygon.exterior.coords for coord in point]
        elif isinstance(merged_polygon, MultiPolygon):
            largest_polygon = max(merged_polygon.geoms, key=lambda p: p.area)
            new_annotation["segmentation"] = [coord for point in largest_polygon.exterior.coords for coord in point]
    
        # Ask user about keeping original annotations
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Merge Annotations")
        msg_box.setText("Do you want to keep the original annotations?")
        msg_box.setIcon(QMessageBox.Question)
        
        keep_button = msg_box.addButton("Keep", QMessageBox.YesRole)
        delete_button = msg_box.addButton("Delete", QMessageBox.NoRole)
        cancel_button = msg_box.addButton("Cancel", QMessageBox.RejectRole)
        
        msg_box.setDefaultButton(cancel_button)
        msg_box.setEscapeButton(cancel_button)
    
        msg_box.exec_()
    
        if msg_box.clickedButton() == cancel_button:
            return
    
        if msg_box.clickedButton() == delete_button:
            for annotation in original_annotations:
                if annotation in self.image_label.annotations[class_name]:
                    self.image_label.annotations[class_name].remove(annotation)
    
        self.image_label.annotations.setdefault(class_name, []).append(new_annotation)
    
        current_name = self.current_slice or self.image_file_name
        self.all_annotations[current_name] = self.image_label.annotations
    
        self.renumber_annotations()
        self.update_annotation_list()
        self.save_current_annotations()
        self.update_slice_list_colors()
        self.image_label.update()
    
        QMessageBox.information(self, "Merge Complete", "Annotations have been merged successfully.")
        self.auto_save()  # Auto-save after merging annotations
    
    
        
    def delete_selected_image(self):
        current_item = self.image_list.currentItem()
        if current_item:
            file_name = current_item.text()
            reply = QMessageBox.question(self, 'Delete Image',
                                         f"Are you sure you want to delete the image '{file_name}'?\n\n"
                                         "This will remove the image and all its associated annotations.",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            
            if reply == QMessageBox.Yes:
                # Remove from all data structures
                self.image_list.takeItem(self.image_list.row(current_item))
                self.image_paths.pop(file_name, None)
                self.all_images = [img for img in self.all_images if img["file_name"] != file_name]
                
                # Remove annotations
                self.all_annotations.pop(file_name, None)
                
                # Handle multi-dimensional images
                base_name = os.path.splitext(file_name)[0]
                if base_name in self.image_slices:
                    # Remove slices
                    for slice_name, _ in self.image_slices[base_name]:
                        self.all_annotations.pop(slice_name, None)
                    del self.image_slices[base_name]
                    
                    # Clear slice list
                    self.slice_list.clear()
                
                # Clear current image and slice if it was the removed image
                if self.image_file_name == file_name:
                    self.current_image = None
                    self.image_file_name = ""
                    self.current_slice = None
                    self.image_label.clear()
                    self.annotation_list.clear()
                
                # Switch to another image if available
                if self.image_list.count() > 0:
                    next_item = self.image_list.item(0)
                    self.image_list.setCurrentItem(next_item)
                    self.switch_image(next_item)
                else:
                    # No images left
                    self.current_image = None
                    self.image_file_name = ""
                    self.current_slice = None
                    self.image_label.clear()
                    self.annotation_list.clear()
                    self.slice_list.clear()
                
                # Update UI
                self.update_ui()
                
                QMessageBox.information(self, "Image Deleted", f"The image '{file_name}' has been deleted.")
            

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
            
    def update_ui(self):
        self.update_image_list()
        self.update_slice_list()
        self.update_class_list()
        self.update_annotation_list()
        self.image_label.update()
        self.update_image_info()
    

    def add_class(self, class_name=None, color=None):
        if not self.image_label.check_unsaved_changes():    
            return
        if class_name is None:
            while True:
                class_name, ok = QInputDialog.getText(self, "Add Class", "Enter class name:")
                if not ok:
                    print("Class addition cancelled")
                    return
                if not class_name.strip():
                    QMessageBox.warning(self, "Invalid Input", "Please enter a class name or press Cancel.")
                    continue
                if class_name in self.class_mapping:
                    QMessageBox.warning(self, "Duplicate Class", f"The class '{class_name}' already exists. Please choose a different name.")
                    continue
                break
    
        if not isinstance(class_name, str):
            print(f"Warning: class_name is not a string. Converting {class_name} to string.")
            class_name = str(class_name)
    
        if color is None:
            color = QColor(Qt.GlobalColor(len(self.image_label.class_colors) % 16 + 7))
        elif isinstance(color, str):
            color = QColor(color)
    
        print(f"Adding class: {class_name}, color: {color.name()}")
    
        self.image_label.class_colors[class_name] = color
        self.class_mapping[class_name] = len(self.class_mapping) + 1
    
        try:
            item = QListWidgetItem(class_name)
            self.update_class_item_color(item, color)
            self.class_list.addItem(item)
    
            self.class_list.setCurrentItem(item)
            self.current_class = class_name
            print(f"Class added successfully: {class_name}")
            self.auto_save()  # Auto-save after adding a class
        except Exception as e:
            print(f"Error adding class: {e}")
            import traceback
            traceback.print_exc()



    
    def update_class_item_color(self, item, color):
        pixmap = QPixmap(16, 16)
        pixmap.fill(color)
        item.setIcon(QIcon(pixmap))
        
    def update_class_list(self):
        self.class_list.clear()
        for class_name, color in self.image_label.class_colors.items():
            item = QListWidgetItem(class_name)
            self.update_class_item_color(item, color)
            self.class_list.addItem(item)
        
        # Re-select the current class if it exists
        if self.current_class:
            items = self.class_list.findItems(self.current_class, Qt.MatchExactly)
            if items:
                self.class_list.setCurrentItem(items[0])
        elif self.class_list.count() > 0:
            # If no class is selected, select the last one
            self.select_class(self.class_list.count() - 1)
    
        print(f"Updated class list with {self.class_list.count()} items")

    def toggle_tool(self):
        if not self.image_label.check_unsaved_changes():
            return
        
        sender = self.sender()
        if sender is None:
            sender = self.sam_magic_wand_button
    
        other_buttons = [btn for btn in self.tool_group.buttons() if btn != sender]
    
        # Deactivate SAM if we're switching to a different tool
        if sender != self.sam_magic_wand_button and self.image_label.sam_magic_wand_active:
            self.deactivate_sam_magic_wand()
    
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
                self.activate_sam_magic_wand()
            elif sender == self.paint_brush_button:
                self.image_label.current_tool = "paint_brush"
                self.image_label.setFocus()  # Set focus on the image label
            elif sender == self.eraser_button:
                self.image_label.current_tool = "eraser"
                self.image_label.setFocus()  # Set focus on the image label
            
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
            if sender == self.sam_magic_wand_button:
                self.deactivate_sam_magic_wand()
    
        # Update UI based on the current tool
        self.update_ui_for_current_tool()

    def wheelEvent(self, event):
        if event.modifiers() == Qt.ControlModifier:
            delta = event.angleDelta().y()
            if self.image_label.current_tool == "paint_brush":
                self.paint_brush_size = max(1, self.paint_brush_size + delta // 120)
                print(f"Paint brush size: {self.paint_brush_size}")
            elif self.image_label.current_tool == "eraser":
                self.eraser_size = max(1, self.eraser_size + delta // 120)
                print(f"Eraser size: {self.eraser_size}")
        else:
            super().wheelEvent(event)



    def update_ui_for_current_tool(self):
        # Disable finish_polygon_button if it still exists in your code
        if hasattr(self, 'finish_polygon_button'):
            self.finish_polygon_button.setEnabled(self.image_label.current_tool in ["polygon", "rectangle"])
    
        # Update button states
        self.polygon_button.setChecked(self.image_label.current_tool == "polygon")
        self.rectangle_button.setChecked(self.image_label.current_tool == "rectangle")
        self.sam_magic_wand_button.setChecked(self.image_label.current_tool == "sam_magic_wand")        

    def on_class_selected(self):
        if not self.image_label.check_unsaved_changes():
            return
        selected_item = self.class_list.currentItem()
        if selected_item:
            self.current_class = selected_item.text()
            print(f"Class selected: {self.current_class}")

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
                self.auto_save()  # Auto-save after changing class color
                

    

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
                self.auto_save()  # Auto-save after renaming a class
    
                #print(f"Class renamed from '{old_name}' to '{new_name}'")

    def delete_class(self):
        current_item = self.class_list.currentItem()
        if current_item:
            class_name = current_item.text()
            
            # Show confirmation dialog
            reply = QMessageBox.question(self, 'Delete Class',
                                         f"Are you sure you want to delete the class '{class_name}'?\n\n"
                                         "This will remove all annotations associated with this class.",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            
            if reply == QMessageBox.Yes:
                # Proceed with deletion
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
                
                # Inform the user
                QMessageBox.information(self, "Class Deleted", f"The class '{class_name}' has been deleted.")
                self.auto_save()  # Auto-save after deleting a class
            else:
                # User cancelled the operation
                QMessageBox.information(self, "Deletion Cancelled", "The class deletion was cancelled.")

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
            self.image_label.drawing_polygon = False  # Reset the drawing_polygon flag
            self.image_label.reset_annotation_state()
            self.image_label.update()
            
            # Save the current annotations
            self.save_current_annotations()
            
            # Update the slice list colors
            self.update_slice_list_colors()
            self.auto_save()  # Auto-save after adding a polygon annotation


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
        number = max([ann.get('number', 0) for ann in annotations] + [0]) + 1
        annotation['number'] = number
        item_text = f"{class_name} - {number}"
        item = QListWidgetItem(item_text)
        item.setData(Qt.UserRole, annotation)
        item.setForeground(color)
        self.annotation_list.addItem(item)
        
        # Clear the current selection
        self.annotation_list.clearSelection()
        self.image_label.highlighted_annotations.clear()
        self.image_label.update()
            
    


    def zoom_in(self):
        new_zoom = min(self.image_label.zoom_factor + 0.1, 5.0)
        self.set_zoom(new_zoom)

    def zoom_out(self):
        new_zoom = max(self.image_label.zoom_factor - 0.1, 0.1)
        self.set_zoom(new_zoom)

    def set_zoom(self, zoom_factor):
        self.image_label.set_zoom(zoom_factor)
        self.zoom_slider.setValue(int(zoom_factor * 100))
        self.image_label.update()  

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
            self.auto_save()

    def enter_edit_mode(self, annotation):
        self.editing_mode = True
        self.disable_tools()

        QMessageBox.information(self, "Edit Mode", "You are now in edit mode. Click and drag points to move them, Shift+Click to delete points, or click on edges to add new points.")

    def exit_edit_mode(self):
        self.editing_mode = False
        self.enable_tools()

        self.image_label.editing_polygon = None
        self.image_label.editing_point_index = None
        self.image_label.hover_point_index = None
        self.update_annotation_list()
        self.image_label.update()

