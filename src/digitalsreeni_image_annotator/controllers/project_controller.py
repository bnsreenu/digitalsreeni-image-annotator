"""Project lifecycle controller.

Extracted from `ImageAnnotator` to give project I/O a single home:
creating, opening, saving, auto-saving, and handling missing images for
`.iap` project files.

State (`is_loading_project`, `backup_project_path`, `current_project_file`,
`current_project_dir`, `project_notes`, etc.) currently still lives on
the main window and is read here via `self.mw`. A future phase may
migrate ownership of those attributes to the controller — for now this
extraction is purely method relocation.
"""

import json
import os
import shutil
from datetime import datetime

from PyQt6.QtCore import QObject
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QFileDialog, QInputDialog, QMessageBox

from ..core import image_utils
from ..core.keypoint_schema import sanitize_schema as _sanitize_keypoint_schema


class ProjectController(QObject):
    def __init__(self, main_window):
        super().__init__(main_window)
        self.mw = main_window

    def update_window_title(self):
        base_title = "Image Annotator"
        if hasattr(self.mw, "current_project_file"):
            project_name = os.path.basename(self.mw.current_project_file)
            project_name = os.path.splitext(project_name)[0]
            self.mw.setWindowTitle(f"{base_title} - {project_name}")
        else:
            self.mw.setWindowTitle(base_title)

    def new_project(self):
        self.mw.remove_all_temp_annotations()
        project_file, _ = QFileDialog.getSaveFileName(
            self.mw, "Create New Project", "", "Image Annotator Project (*.iap)"
        )
        if project_file:
            if not project_file.lower().endswith(".iap"):
                project_file += ".iap"

            self.mw.current_project_file = project_file
            self.mw.current_project_dir = os.path.dirname(project_file)

            images_dir = os.path.join(self.mw.current_project_dir, "images")
            os.makedirs(images_dir, exist_ok=True)

            self.mw.clear_all(new_project=True, show_messages=False)

            notes, ok = QInputDialog.getMultiLineText(
                self.mw, "Project Notes", "Enter initial project notes:"
            )
            self.mw.project_notes = notes if ok else ""
            self.mw.project_creation_date = datetime.now().isoformat()

            self.save_project(show_message=False)

            self.mw.show_info(
                "New Project", f"New project created at {self.mw.current_project_file}"
            )
            self.mw.initialize_yolo_trainer()
            self.update_window_title()

    def open_project(self):
        print("open_project method called")
        self.mw.remove_all_temp_annotations()
        project_file, _ = QFileDialog.getOpenFileName(
            self.mw, "Open Project", "", "Image Annotator Project (*.iap)"
        )
        print(f"Selected project file: {project_file}")
        if project_file:
            try:
                self.backup_project_before_open(project_file)
                self.open_specific_project(project_file)
            except Exception as e:
                self.restore_project_from_backup()
                QMessageBox.critical(
                    self.mw,
                    "Error",
                    f"An error occurred while opening the project: {str(e)}\n"
                    f"The project file has been restored from backup.",
                )
        else:
            print("No project file selected")

    def backup_project_before_open(self, project_file):
        """Create a backup of the project file before opening it."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = os.path.join(os.path.dirname(project_file), ".project_backups")
        os.makedirs(backup_dir, exist_ok=True)

        self.mw.backup_project_path = os.path.join(
            backup_dir, f"{os.path.basename(project_file)}.{timestamp}.backup"
        )
        shutil.copy2(project_file, self.mw.backup_project_path)

    def restore_project_from_backup(self):
        """Restore the project file from its backup if available."""
        if self.mw.backup_project_path and os.path.exists(self.mw.backup_project_path):
            try:
                shutil.copy2(self.mw.backup_project_path, self.mw.current_project_file)
                print(f"Project restored from backup: {self.mw.backup_project_path}")
            except Exception as e:
                print(f"Failed to restore from backup: {str(e)}")

    def open_specific_project(self, project_file):
        print(f"Opening specific project: {project_file}")
        if os.path.exists(project_file):
            try:
                self.mw.is_loading_project = True

                with open(project_file, "r", encoding='utf-8') as f:
                    project_data = json.load(f)

                self.mw.clear_all(show_messages=False)
                self.mw.current_project_file = project_file
                self.mw.current_project_dir = os.path.dirname(project_file)

                self.mw.project_notes = project_data.get("notes", "")
                self.mw.project_creation_date = project_data.get("creation_date", "")
                self.mw.last_modified = project_data.get("last_modified", "")

                if self.mw.project_creation_date:
                    self.mw.project_creation_date = datetime.fromisoformat(
                        self.mw.project_creation_date
                    ).strftime("%Y-%m-%d %H:%M:%S")
                if self.mw.last_modified:
                    self.mw.last_modified = datetime.fromisoformat(
                        self.mw.last_modified
                    ).strftime("%Y-%m-%d %H:%M:%S")

                self.load_project_data(project_data)

                self.mw.is_loading_project = False
                if self.mw.dino_class_table.rowCount() > 0:
                    self.mw.dino_class_table.selectRow(0)
                self.save_project(show_message=False)

                self.mw.initialize_yolo_trainer()
                self.update_window_title()

                # No success dialog — the loaded canvas + updated window title
                # already make a successful open obvious; a modal just adds a
                # click. Errors below still surface as dialogs.
                print(f"Project opened successfully: {project_file}")

            except Exception as e:
                self.mw.is_loading_project = False
                raise e
        else:
            print(f"Project file not found: {project_file}")
            QMessageBox.critical(
                self.mw, "Error", f"Project file not found: {project_file}"
            )

    def load_project_data(self, project_data):
        """Load project data without triggering auto-saves."""
        self.mw.class_mapping.clear()
        self.mw.image_label.class_colors.clear()
        self.mw.keypoint_schemas.clear()
        for class_info in project_data.get("classes", []):
            self.mw.add_class(class_info["name"], QColor(class_info["color"]))
            # Restore the keypoint schema for pose classes (issue #35). Malformed
            # schemas are dropped with a warning rather than crashing the load,
            # mirroring the DINO-config validate-on-load pattern below.
            schema = _sanitize_keypoint_schema(class_info.get("keypoint_schema"))
            if schema is not None:
                self.mw.keypoint_schemas[class_info["name"]] = schema
            elif class_info.get("keypoint_schema") is not None:
                print(f"  Skipped malformed keypoint schema for class "
                      f"'{class_info['name']}'.")

        self.mw.all_images = project_data.get("images", [])
        self.mw.image_paths = project_data.get("image_paths", {})

        self.mw.all_annotations.clear()
        for image_info in project_data["images"]:
            if image_info.get("is_multi_slice", False):
                for slice_info in image_info.get("slices", []):
                    self.mw.all_annotations[slice_info["name"]] = slice_info["annotations"]
            else:
                self.mw.all_annotations[image_info["file_name"]] = image_info.get(
                    "annotations", {}
                )

        missing_images = []
        for image_info in project_data["images"]:
            image_path = os.path.join(
                self.mw.current_project_dir, "images", image_info["file_name"]
            )

            if not os.path.exists(image_path):
                missing_images.append(image_info["file_name"])
                continue

            self.mw.image_paths[image_info["file_name"]] = image_path

            if image_info.get("is_multi_slice", False):
                dimensions = image_info.get("dimensions", [])
                shape = image_info.get("shape", [])
                self.mw.load_multi_slice_image(image_path, dimensions, shape)
            else:
                self.mw.add_images_to_list([image_path])

        dino_cfg = project_data.get("dino_config", {})
        valid_classes = set(self.mw.class_mapping.keys())

        phrases = dino_cfg.get("phrases", {})
        if phrases:
            kept = {k: v for k, v in phrases.items() if k in valid_classes}
            for orphan in phrases.keys() - kept.keys():
                print(f"  Skipped saved DINO phrases for unknown class "
                      f"'{orphan}' — class is not in the current project.")
            self.mw.dino_phrase_panel.set_phrases(kept)

        for cls_name, thr in dino_cfg.get("thresholds", {}).items():
            ok = self.mw.dino_class_table.set_thresholds(
                cls_name,
                thr.get("box", 0.25),
                thr.get("txt", 0.25),
                thr.get("nms", 0.50),
            )
            if not ok:
                print(f"  Skipped saved DINO thresholds for unknown class "
                      f"'{cls_name}' — class is not in the current project.")

        self.mw.update_ui()

        if missing_images:
            self.handle_missing_images(missing_images)

        if self.mw.image_list.count() > 0:
            self.mw.image_list.setCurrentRow(0)
            first_item = self.mw.image_list.item(0)
            if first_item:
                self.mw.switch_image(first_item)

        if self.mw.class_list.count() > 0:
            self.mw.class_list.setCurrentRow(0)
            self.mw.on_class_selected()

    def handle_missing_images(self, missing_images):
        message = "The following images have annotations but were not found in the project directory:\n\n"
        message += "\n".join(missing_images[:10])
        if len(missing_images) > 10:
            message += f"\n... and {len(missing_images) - 10} more."
        message += "\n\nWould you like to locate these images now?"

        reply = QMessageBox.question(
            self.mw,
            "Missing Images",
            message,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.load_missing_images(missing_images)
        else:
            self.remove_missing_images(missing_images)

    def remove_missing_images(self, missing_images):
        for image_name in missing_images:
            self.mw.all_images = [
                img for img in self.mw.all_images if img["file_name"] != image_name
            ]
            self.mw.image_paths.pop(image_name, None)
            self.mw.all_annotations.pop(image_name, None)

            base_name = os.path.splitext(image_name)[0]
            if base_name in self.mw.image_slices:
                for slice_name, _ in self.mw.image_slices[base_name]:
                    self.mw.all_annotations.pop(slice_name, None)
                del self.mw.image_slices[base_name]

        self.mw.update_ui()
        QMessageBox.information(
            self.mw,
            "Images Removed",
            f"{len(missing_images)} missing images and their annotations have been removed from the project.",
        )

    def prompt_load_missing_images(self, missing_images):
        message = "The following images have annotations but were not found in the project directory:\n\n"
        message += "\n".join(missing_images[:10])
        if len(missing_images) > 10:
            message += f"\n... and {len(missing_images) - 10} more."
        message += "\n\nWould you like to locate these images now?"

        reply = QMessageBox.question(
            self.mw,
            "Load Missing Images",
            message,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.load_missing_images(missing_images)

    def load_missing_images(self, missing_images):
        files, _ = QFileDialog.getOpenFileNames(
            self.mw,
            "Select Missing Images",
            "",
            "Image Files (*.png *.jpg *.bmp *.tif *.tiff *.czi)",
        )
        if files:
            images_loaded = 0
            for file_path in files:
                file_name = os.path.basename(file_path)
                if file_name in missing_images:
                    dst_path = os.path.join(
                        self.mw.current_project_dir, "images", file_name
                    )
                    shutil.copy2(file_path, dst_path)
                    self.mw.image_paths[file_name] = dst_path

                    if not any(
                        img["file_name"] == file_name for img in self.mw.all_images
                    ):
                        self.mw.all_images.append(
                            {
                                "file_name": file_name,
                                "height": 0,
                                "width": 0,
                                "id": len(self.mw.all_images) + 1,
                                "is_multi_slice": False,
                            }
                        )
                    images_loaded += 1
                    missing_images.remove(file_name)

            self.mw.update_image_list()
            if images_loaded > 0:
                self.mw.image_list.setCurrentRow(0)
                self.mw.switch_image(self.mw.image_list.item(0))
            QMessageBox.information(
                self.mw,
                "Images Loaded",
                f"Successfully copied and loaded {images_loaded} out of {len(files)} selected images.",
            )

            if missing_images:
                self.prompt_load_missing_images(missing_images)

    def check_missing_images(self):
        missing_images = [
            img["file_name"]
            for img in self.mw.all_images
            if img["file_name"] not in self.mw.image_paths
            or not os.path.exists(self.mw.image_paths[img["file_name"]])
        ]
        if missing_images:
            self.prompt_load_missing_images(missing_images)

    def close_project(self):
        if hasattr(self.mw, "current_project_file"):
            reply = QMessageBox.question(
                self.mw,
                "Close Project",
                "Do you want to save the current project before closing?",
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.No
                | QMessageBox.StandardButton.Cancel,
            )

            if reply == QMessageBox.StandardButton.Yes:
                self.mw.remove_all_temp_annotations()
                self.save_project(show_message=False)
            elif reply == QMessageBox.StandardButton.Cancel:
                return

        self.mw.clear_all(new_project=True, show_messages=False)

        if hasattr(self.mw, "current_project_file"):
            del self.mw.current_project_file
        if hasattr(self.mw, "current_project_dir"):
            del self.mw.current_project_dir

        self.update_window_title()

    def save_project(self, show_message=True):
        if not hasattr(self.mw, "current_project_file") or not self.mw.current_project_file:
            self.mw.current_project_file, _ = QFileDialog.getSaveFileName(
                self.mw, "Save Project", "", "Image Annotator Project (*.iap)"
            )
            if not self.mw.current_project_file:
                return

        self.mw.current_project_dir = os.path.dirname(self.mw.current_project_file)

        images_dir = os.path.join(self.mw.current_project_dir, "images")
        os.makedirs(images_dir, exist_ok=True)

        images_to_copy = []
        for file_name, src_path in self.mw.image_paths.items():
            dst_path = os.path.join(images_dir, file_name)
            if os.path.abspath(src_path) != os.path.abspath(dst_path):
                if not os.path.exists(dst_path):
                    images_to_copy.append((file_name, src_path, dst_path))

        if images_to_copy:
            reply = QMessageBox.question(
                self.mw,
                "Image Directory Structure",
                f"The project structure requires all images to be in an 'images' subdirectory. "
                f"{len(images_to_copy)} images need to be copied to the correct location. "
                f"Do you want to copy these images?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )

            if reply == QMessageBox.StandardButton.Yes:
                for file_name, src_path, dst_path in images_to_copy:
                    try:
                        shutil.copy2(src_path, dst_path)
                        self.mw.image_paths[file_name] = dst_path
                    except Exception as e:
                        QMessageBox.warning(
                            self.mw, "Copy Failed", f"Failed to copy {file_name}: {str(e)}"
                        )
                        return
            else:
                QMessageBox.warning(
                    self.mw,
                    "Save Cancelled",
                    "Project cannot be saved without the correct directory structure.",
                )
                return

        images_data = []
        for image_info in self.mw.all_images:
            file_name = image_info["file_name"]
            image_data = {
                "file_name": file_name,
                "width": image_info["width"],
                "height": image_info["height"],
                "is_multi_slice": image_info["is_multi_slice"],
            }

            if image_data["is_multi_slice"]:
                base_name_without_ext = os.path.splitext(file_name)[0]
                image_data["slices"] = []
                for slice_name, _ in self.mw.image_slices.get(base_name_without_ext, []):
                    slice_data = {
                        "name": slice_name,
                        "annotations": image_utils.convert_to_serializable(
                            self.mw.all_annotations.get(slice_name, {})
                        ),
                    }
                    image_data["slices"].append(slice_data)

                image_data["dimensions"] = image_utils.convert_to_serializable(
                    self.mw.image_dimensions.get(base_name_without_ext, [])
                )
                image_data["shape"] = image_utils.convert_to_serializable(
                    self.mw.image_shapes.get(base_name_without_ext, [])
                )
            else:
                image_data["annotations"] = {}
                for class_name, annotations in self.mw.all_annotations.get(
                    file_name, {}
                ).items():
                    image_data["annotations"][class_name] = [
                        ann.copy() for ann in annotations
                    ]

            images_data.append(image_data)

        project_data = {
            "classes": [
                {
                    "name": name,
                    "color": color.name(),
                    # Pose classes carry their keypoint schema inline (issue #35);
                    # normal classes add nothing, so old projects are unchanged.
                    **(
                        {"keypoint_schema": self.mw.keypoint_schemas[name]}
                        if name in self.mw.keypoint_schemas
                        else {}
                    ),
                }
                for name, color in self.mw.image_label.class_colors.items()
            ],
            "images": images_data,
            "image_paths": {
                k: v for k, v in self.mw.image_paths.items() if os.path.exists(v)
            },
            "notes": getattr(self.mw, "project_notes", ""),
            "creation_date": getattr(
                self.mw, "project_creation_date", datetime.now().isoformat()
            ),
            "last_modified": datetime.now().isoformat(),
        }

        dino_cfg = {
            "phrases": self.mw.dino_phrase_panel.get_all_phrases(),
            "thresholds": self.mw.dino_class_table.get_thresholds_dict(),
        }
        if dino_cfg["phrases"] or dino_cfg["thresholds"]:
            project_data["dino_config"] = dino_cfg

        with open(self.mw.current_project_file, "w", encoding='utf-8') as f:
            json.dump(image_utils.convert_to_serializable(project_data), f, indent=2)

        if show_message:
            self.mw.show_info(
                "Project Saved", f"Project saved to {self.mw.current_project_file}"
            )

        self.update_window_title()

        for file_name in self.mw.image_paths.keys():
            self.mw.image_paths[file_name] = os.path.join(images_dir, file_name)

    def save_project_as(self):
        new_project_file, _ = QFileDialog.getSaveFileName(
            self.mw, "Save Project As", "", "Image Annotator Project (*.iap)"
        )
        if new_project_file:
            if not new_project_file.lower().endswith(".iap"):
                new_project_file += ".iap"

            original_project_file = getattr(self.mw, "current_project_file", None)

            self.mw.current_project_file = new_project_file
            self.mw.current_project_dir = os.path.dirname(new_project_file)

            self.save_project(show_message=False)
            self.update_window_title()

            QMessageBox.information(
                self.mw, "Project Saved As", f"Project saved as:\n{new_project_file}"
            )

            if original_project_file is None:
                self.mw.current_project_file = new_project_file

    def auto_save(self):
        if self.mw.is_loading_project:
            return

        if not hasattr(self.mw, "current_project_file"):
            reply = QMessageBox.question(
                self.mw,
                "No Project",
                "You need to save the project before auto-saving. Would you like to save now?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.save_project()
            else:
                return

        if hasattr(self.mw, "current_project_file"):
            self.save_project(show_message=False)
            print("Project auto-saved.")
