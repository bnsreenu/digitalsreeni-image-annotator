import os
from ultralytics import YOLO
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLineEdit, QLabel, QFileDialog, QDialogButtonBox)
import yaml
import numpy as np
from pathlib import Path
from .export_formats import export_yolo_v5plus


from collections import deque


from PyQt5.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QPushButton
from PyQt5.QtCore import Qt, pyqtSignal, QObject

class TrainingInfoDialog(QDialog):
    stop_signal = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Training Progress")
        self.setModal(False)
        self.layout = QVBoxLayout(self)

        self.info_text = QTextEdit(self)
        self.info_text.setReadOnly(True)
        self.layout.addWidget(self.info_text)

        self.stop_button = QPushButton("Stop Training", self)
        self.stop_button.clicked.connect(self.stop_training)
        self.layout.addWidget(self.stop_button)

        self.close_button = QPushButton("Close", self)
        self.close_button.clicked.connect(self.hide)
        self.layout.addWidget(self.close_button)

        self.setMinimumSize(400, 300)

    def update_info(self, text):
        self.info_text.append(text)
        self.info_text.verticalScrollBar().setValue(self.info_text.verticalScrollBar().maximum())

    def stop_training(self):
        self.stop_signal.emit()
        self.stop_button.setEnabled(False)
        self.stop_button.setText("Stopping...")

    def closeEvent(self, event):
        event.ignore()
        self.hide()

class LoadPredictionModelDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Load Prediction Model and YAML")
        self.model_path = ""
        self.yaml_path = ""

        layout = QVBoxLayout(self)

        # Model file selection
        model_layout = QHBoxLayout()
        self.model_edit = QLineEdit()
        model_button = QPushButton("Browse")
        model_button.clicked.connect(self.browse_model)
        model_layout.addWidget(QLabel("Model File:"))
        model_layout.addWidget(self.model_edit)
        model_layout.addWidget(model_button)
        layout.addLayout(model_layout)

        # YAML file selection
        yaml_layout = QHBoxLayout()
        self.yaml_edit = QLineEdit()
        yaml_button = QPushButton("Browse")
        yaml_button.clicked.connect(self.browse_yaml)
        yaml_layout.addWidget(QLabel("YAML File:"))
        yaml_layout.addWidget(self.yaml_edit)
        yaml_layout.addWidget(yaml_button)
        layout.addLayout(yaml_layout)

        # OK and Cancel buttons
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

    def browse_model(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select YOLO Model", "", "YOLO Model (*.pt)")
        if file_name:
            self.model_path = file_name
            self.model_edit.setText(file_name)

    def browse_yaml(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select YAML File", "", "YAML Files (*.yaml *.yml)")
        if file_name:
            self.yaml_path = file_name
            self.yaml_edit.setText(file_name)
        
class YOLOTrainer(QObject):
    progress_signal = pyqtSignal(str)

    def __init__(self, project_dir, main_window):
        super().__init__()
        self.project_dir = project_dir
        self.main_window = main_window
        self.model = None
        self.dataset_path = os.path.join(project_dir, "yolo_dataset")
        self.model_path = os.path.join(project_dir, "yolo_model")
        self.yaml_path = None
        self.yaml_data = None
        self.epoch_info = deque(maxlen=10)
        self.progress_callback = None
        self.total_epochs = None
        self.conf_threshold = 0.25
        self.stop_training = False
        self.class_names = None

    def load_model(self, model_path=None):
        if model_path is None:
            model_path, _ = QFileDialog.getOpenFileName(self.main_window, "Select YOLO Model", "", "YOLO Model (*.pt)")
        if model_path:
            try:
                self.model = YOLO(model_path)
                return True
            except Exception as e:
                QMessageBox.critical(self.main_window, "Error Loading Model", f"Could not load the model. Error: {str(e)}")
        return False

    def prepare_dataset(self):
        output_dir, yaml_path = export_yolo_v5plus(
            self.main_window.all_annotations,
            self.main_window.class_mapping,
            self.main_window.image_paths,
            self.main_window.slices,
            self.main_window.image_slices,
            self.dataset_path
        )
        
        yaml_path = Path(yaml_path)
        with yaml_path.open('r') as f:
            yaml_content = yaml.safe_load(f)
        
        # Update paths for new YOLO v5+ structure
        yaml_content['train'] = 'images/train'  # Changed from train/images
        yaml_content['val'] = 'images/val'      # Changed from train/images
        yaml_content['test'] = '../test/images'
        
        with yaml_path.open('w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)
        
        self.yaml_path = str(yaml_path)
        return self.yaml_path

    def load_yaml(self, yaml_path=None):
        if yaml_path is None:
            yaml_path, _ = QFileDialog.getOpenFileName(self.main_window, "Select YOLO Dataset YAML", "", "YAML Files (*.yaml *.yml)")
        if yaml_path and os.path.exists(yaml_path):
            with open(yaml_path, 'r') as f:
                try:
                    yaml_data = yaml.safe_load(f)
                    print(f"Loaded YAML contents: {yaml_data}")
    
                    # Ensure paths are relative
                    for key in ['train', 'val', 'test']:
                        if key in yaml_data and os.path.isabs(yaml_data[key]):
                            yaml_data[key] = os.path.relpath(yaml_data[key], start=os.path.dirname(yaml_path))
    
                    print(f"Updated YAML contents: {yaml_data}")
    
                    # Save the updated YAML data
                    self.yaml_data = yaml_data
                    self.yaml_path = yaml_path
    
                    # Write the updated YAML back to the file
                    with open(yaml_path, 'w') as f:
                        yaml.dump(yaml_data, f, default_flow_style=False)
    
                    return True
                except yaml.YAMLError as e:
                    QMessageBox.critical(self.main_window, "Error Loading YAML", f"Invalid YAML file. Error: {str(e)}")
        return False

    def on_train_epoch_end(self, trainer):
        epoch = trainer.epoch + 1  # Add 1 to start from 1 instead of 0
        total_epochs = trainer.epochs
        loss = trainer.loss.item()
        progress_text = f"Epoch {epoch}/{total_epochs}, Loss: {loss:.4f}"
        
        # Only emit the signal, don't call the callback directly
        self.progress_signal.emit(progress_text)
        
        if self.stop_training:
            trainer.model.stop = True
            self.stop_training = False
            return False
        return True
    
    def train_model(self, epochs=100, imgsz=640):
        if self.model is None:
            raise ValueError("No model loaded. Please load a model first.")
        if self.yaml_path is None or not Path(self.yaml_path).exists():
            raise FileNotFoundError("Dataset YAML not found. Please prepare or load a dataset first.")
    
        self.stop_training = False
        self.total_epochs = epochs
        self.epoch_info.clear()
        
        # Add the callback
        self.model.add_callback("on_train_epoch_end", self.on_train_epoch_end)
        
        try:
            yaml_path = Path(self.yaml_path)
            yaml_dir = yaml_path.parent
            
            print(f"Training with YAML: {yaml_path}")
            print(f"YAML directory: {yaml_dir}")
            
            with yaml_path.open('r') as f:
                yaml_content = yaml.safe_load(f)
            print(f"YAML content: {yaml_content}")
            
            # For now, use train as val since we don't have separate validation set
            train_dir = str(yaml_dir / 'images' / 'train')
            
            # Update YAML content with correct paths
            yaml_content['train'] = train_dir
            yaml_content['val'] = train_dir  # Use same directory for validation
            
            # Create the val directory structure if it doesn't exist
            val_img_dir = yaml_dir / 'images' / 'val'
            val_label_dir = yaml_dir / 'labels' / 'val'
            val_img_dir.mkdir(parents=True, exist_ok=True)
            val_label_dir.mkdir(parents=True, exist_ok=True)
            
            # Write updated YAML with adjusted paths
            temp_yaml_path = yaml_dir / 'temp_train.yaml'
            with temp_yaml_path.open('w') as f:
                yaml.dump(yaml_content, f, default_flow_style=False)
            
            print(f"Training with updated YAML: {temp_yaml_path}")
            print(f"Updated YAML content: {yaml_content}")
            
            results = self.model.train(data=str(temp_yaml_path), epochs=epochs, imgsz=imgsz)
            return results
        finally:
            # Clear the callback
            self.model.callbacks["on_train_epoch_end"] = []
            # Remove temporary YAML file
            if 'temp_yaml_path' in locals():
                temp_yaml_path.unlink(missing_ok=True)
           
    def verify_dataset_structure(self):
        yaml_path = Path(self.yaml_path)
        yaml_dir = yaml_path.parent
        
        with yaml_path.open('r') as f:
            yaml_content = yaml.safe_load(f)
        
        # Use paths from YAML content
        train_images_dir = yaml_dir / yaml_content.get('train', 'images/train')
        val_images_dir = yaml_dir / yaml_content.get('val', 'images/val')
        train_labels_dir = yaml_dir / 'labels' / 'train'  # Labels directory corresponds to images
        val_labels_dir = yaml_dir / 'labels' / 'val'      # Labels directory corresponds to images
        
        # Check both train and val directories
        missing_dirs = []
        if not train_images_dir.exists():
            missing_dirs.append(f"Training images directory: {train_images_dir}")
        if not train_labels_dir.exists():
            missing_dirs.append(f"Training labels directory: {train_labels_dir}")
        if not val_images_dir.exists():
            missing_dirs.append(f"Validation images directory: {val_images_dir}")
        if not val_labels_dir.exists():
            missing_dirs.append(f"Validation labels directory: {val_labels_dir}")
        
        if missing_dirs:
            raise FileNotFoundError(f"The following directories were not found:\n" + "\n".join(missing_dirs))
        
        print(f"Dataset structure verified:")
        print(f"Train images: {train_images_dir}")
        print(f"Train labels: {train_labels_dir}")
        print(f"Val images: {val_images_dir}")
        print(f"Val labels: {val_labels_dir}")

    def check_ultralytics_settings(self):
        settings_path = Path.home() / ".config" / "Ultralytics" / "settings.yaml"
        if settings_path.exists():
            with settings_path.open('r') as f:
                settings = yaml.safe_load(f)
            print(f"Ultralytics settings: {settings}")
        else:
            print("Ultralytics settings file not found.")
            
    def stop_training_signal(self):
        self.stop_training = True
        self.progress_signal.emit("Stopping training...")

    def set_progress_callback(self, callback):
        self.progress_callback = callback

    
    def stop_training_callback(self, trainer):
        if getattr(self, 'stop_training', False):
            trainer.model.stop = True
            self.stop_training = False
            

            
    def on_epoch_end(self, trainer):
        # Get current epoch
        epoch = trainer.epoch if hasattr(trainer, 'epoch') else trainer.current_epoch

        # Get total epochs
        total_epochs = self.total_epochs  # Use the value we set in train_model

        # Get loss
        if hasattr(trainer, 'metrics') and 'train/box_loss' in trainer.metrics:
            loss = trainer.metrics['train/box_loss']
        elif hasattr(trainer, 'loss'):
            loss = trainer.loss
        else:
            loss = 0  # Default value if loss can't be found

        # Ensure loss is a number
        loss = float(loss)

        info = f"Epoch {epoch}/{total_epochs}, Loss: {loss:.4f}"
        self.epoch_info.append(info)
        
        display_text = f"Current Progress:\n" + "\n".join(self.epoch_info)
        if self.progress_callback:
            self.progress_callback(display_text)


    def save_model(self):
        if self.model is None:
            raise ValueError("No model to save. Please train a model first.")
        save_path, _ = QFileDialog.getSaveFileName(self.main_window, "Save YOLO Model", "", "YOLO Model (*.pt)")
        if save_path:
            self.model.export(save_path)
            return True
        return False

    def load_prediction_model(self, model_path, yaml_path):
        try:
            self.model = YOLO(model_path)
            with open(yaml_path, 'r') as f:
                self.prediction_yaml = yaml.safe_load(f)
            
            if 'names' not in self.prediction_yaml:
                raise ValueError("The YAML file does not contain a 'names' section for class names.")
            
            self.class_names = self.prediction_yaml['names']
            print(f"Loaded class names: {self.class_names}")
            
            # Verify that the number of classes in the YAML matches the model
            if len(self.class_names) != len(self.model.names):
                mismatch_message = (f"Warning: Number of classes in YAML ({len(self.class_names)}) "
                                    f"does not match the model ({len(self.model.names)}). "
                                    "This may cause issues during prediction.")
                print(mismatch_message)
                return True, mismatch_message
            
            return True, None
        except Exception as e:
            error_message = f"Error loading model or YAML: {str(e)}"
            print(error_message)
            return False, error_message
    
    def predict(self, input_data):
        if self.model is None:
            raise ValueError("No model loaded. Please load a model first.")
        if isinstance(input_data, str):
            # It's a file path
            results = self.model(input_data, task='segment', conf=self.conf_threshold, save=False, show=False)
        elif isinstance(input_data, np.ndarray):
            # It's a numpy array
            results = self.model(input_data, task='segment', conf=self.conf_threshold, save=False, show=False)
        else:
            raise ValueError("Invalid input type. Expected file path or numpy array.")
        
        # Get the input size used for prediction and the original image size
        input_size = results[0].orig_shape
        original_size = results[0].orig_img.shape[:2]
        return results, input_size, original_size

    def set_conf_threshold(self, conf):
        self.conf_threshold = conf