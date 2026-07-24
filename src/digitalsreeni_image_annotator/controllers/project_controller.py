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
from pathlib import PurePath

from PyQt6.QtCore import QObject
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QFileDialog, QInputDialog, QMessageBox

from ..core import image_utils, recovery
from ..core.keypoint_schema import sanitize_schema as _sanitize_keypoint_schema
from ..core.project_schema import validate_project_data
from ..core.slice_cache import release_slices, slice_names

from ..core.logging_config import get_logger

logger = get_logger(__name__)


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
        logger.debug("open_project method called")
        self.mw.remove_all_temp_annotations()
        project_file, _ = QFileDialog.getOpenFileName(
            self.mw, "Open Project", "", "Image Annotator Project (*.iap)"
        )
        logger.debug(f"Selected project file: {project_file}")
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
            logger.debug("No project file selected")

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
                logger.info(f"Project restored from backup: {self.mw.backup_project_path}")
            except Exception:
                logger.exception("Failed to restore from backup")

    def open_specific_project(self, project_file):
        logger.debug(f"Opening specific project: {project_file}")
        if os.path.exists(project_file):
            try:
                self.mw.is_loading_project = True

                with open(project_file, "r", encoding='utf-8') as f:
                    project_data = json.load(f)

                problems = validate_project_data(project_data)
                if problems:
                    raise ValueError(
                        "This project file is not valid:\n- " + "\n- ".join(problems)
                    )

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
                logger.info(f"Project opened successfully: {project_file}")

            except Exception as e:
                self.mw.is_loading_project = False
                raise e
        else:
            logger.warning(f"Project file not found: {project_file}")
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
                logger.warning(f"Skipped malformed keypoint schema for class "
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
            image_path = self.resolve_image_path(image_info["file_name"], project_data)

            if image_path is None:
                missing_images.append(image_info["file_name"])
                continue

            self.mw.image_paths[image_info["file_name"]] = image_path

            if image_info.get("is_multi_slice", False):
                if image_info.get("is_video"):
                    # A missing video already flowed through resolve_image_path
                    # → missing_images above, so image_path exists here (#47).
                    self.mw.image_controller.load_video(image_path)
                else:
                    dimensions = image_info.get("dimensions", [])
                    shape = image_info.get("shape", [])
                    self.mw.load_multi_slice_image(image_path, dimensions, shape)
            else:
                self.mw.add_images_to_list([image_path])

        # Per-image group tags (issue #43) need no restoration step: line ~193
        # aliases self.mw.all_images to project_data["images"], and the load
        # loop above does not rebuild it (add_images_to_list no-ops because
        # image_paths[file_name] is set first, and load_multi_slice_image only
        # loads slices), so the "group" keys parsed from JSON survive as-is.

        dino_cfg = project_data.get("dino_config", {})
        valid_classes = set(self.mw.class_mapping.keys())

        phrases = dino_cfg.get("phrases", {})
        if phrases:
            kept = {k: v for k, v in phrases.items() if k in valid_classes}
            for orphan in phrases.keys() - kept.keys():
                logger.warning(f"Skipped saved DINO phrases for unknown class "
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
                logger.warning(f"Skipped saved DINO thresholds for unknown class "
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

    def resolve_image_path(self, file_name, project_data):
        """Resolve a stored image reference to an existing absolute path (#42).

        Tries, in order, the first that exists on disk:
          1. the project-relative path (portable across machines/OSes);
          2. the stored absolute path (covers images referenced outside the
             project's ``images/`` dir — previously dead data on load);
          3. the historical ``<project_dir>/images/<file_name>`` convention
             (so v1 projects with neither key resolve exactly as before).
        Returns None when none exist — the caller reports it as missing.
        """
        project_dir = getattr(self.mw, "current_project_dir", None)

        rel = (project_data.get("image_paths_rel") or {}).get(file_name)
        if rel and project_dir:
            candidate = os.path.normpath(os.path.join(project_dir, rel))
            if os.path.exists(candidate):
                return candidate

        abs_path = (project_data.get("image_paths") or {}).get(file_name)
        if abs_path and os.path.exists(abs_path):
            return abs_path

        if project_dir:
            candidate = os.path.join(project_dir, "images", file_name)
            if os.path.exists(candidate):
                return candidate

        return None

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
                stack_slices = self.mw.image_slices[base_name]
                for slice_name in slice_names(stack_slices):
                    self.mw.all_annotations.pop(slice_name, None)
                release_slices(stack_slices)  # evict cached QImages (issue #45)
                del self.mw.image_slices[base_name]
                # Drop live refs if this was the active stack/video, else the
                # update_ui() below resurrects the orphaned slices (video-removal
                # bug; mirrors remove_image / delete_selected_image).
                if self.mw.slices is stack_slices:
                    self.mw.slices = []

            # Release the video's cv2 capture if this base is a video (#47).
            if base_name in self.mw.video_handlers:
                self.mw.video_handlers[base_name].release()
                del self.mw.video_handlers[base_name]

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
        from ..core.video_handler import file_dialog_filter

        files, _ = QFileDialog.getOpenFileNames(
            self.mw, "Select Missing Images or Videos", "", file_dialog_filter()
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

    def build_project_data(self):
        """Assemble the full project dict for serialization.

        Pure data-building only: no dialogs, no file I/O, no image copying — so
        it can be reused by both save_project() and the silent unsaved-project
        recovery writer (issue #41). For a saved project the output matches the
        previous inline block, plus the portable ``image_paths_rel`` key (#42).
        """
        images_data = []
        for image_info in self.mw.all_images:
            file_name = image_info["file_name"]
            image_data = {
                "file_name": file_name,
                "width": image_info["width"],
                "height": image_info["height"],
                "is_multi_slice": image_info["is_multi_slice"],
            }
            # Persist the optional group tag (issue #43); absent for
            # ungrouped images so old projects are unchanged.
            if image_info.get("group"):
                image_data["group"] = image_info["group"]

            if image_data["is_multi_slice"]:
                base_name_without_ext = os.path.splitext(file_name)[0]
                image_data["slices"] = []
                # Name-only (slice_names) — saving a project must not decode a
                # single slice's pixels (issue #45); it also accepts a plain
                # list or an absent/None entry.
                for slice_name in slice_names(
                    self.mw.image_slices.get(base_name_without_ext)
                ):
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
                # Videos carry no dimensions/shape but need their metadata so
                # a reload knows to re-open them via load_video (issue #47).
                # Slice entries (name + annotations) serialize unchanged — the
                # LazySliceList persists names only, no pixels (issue #45).
                if image_info.get("is_video"):
                    image_data["is_video"] = True
                    image_data["video_metadata"] = image_info.get("video_metadata")
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

        # Portable, project-relative image paths alongside the absolutes (#42).
        # Only when a project dir exists — recovery snapshots for an unsaved
        # project have none and rely on the absolute paths (restore is
        # same-machine). The absolute entries stay so older app versions still
        # open the file; resolve_image_path() prefers the relative ones on load.
        project_dir = getattr(self.mw, "current_project_dir", None)
        if project_dir:
            rel = {}
            for file_name, abs_path in project_data["image_paths"].items():
                try:
                    rel[file_name] = PurePath(
                        os.path.relpath(abs_path, project_dir)
                    ).as_posix()
                except ValueError:
                    # Different drive (Windows): no relative path exists; the
                    # absolute entry remains the fallback.
                    pass
            project_data["image_paths_rel"] = rel

        dino_cfg = {
            "phrases": self.mw.dino_phrase_panel.get_all_phrases(),
            "thresholds": self.mw.dino_class_table.get_thresholds_dict(),
        }
        if dino_cfg["phrases"] or dino_cfg["thresholds"]:
            project_data["dino_config"] = dino_cfg

        return project_data

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

        project_data = self.build_project_data()

        with open(self.mw.current_project_file, "w", encoding='utf-8') as f:
            json.dump(image_utils.convert_to_serializable(project_data), f, indent=2)

        if show_message:
            self.mw.show_info(
                "Project Saved", f"Project saved to {self.mw.current_project_file}"
            )

        # The project now has a real `.iap`; drop any unsaved-project recovery
        # snapshot so a stale one is never offered on next launch (issue #41).
        # Covers new_project() too, which saves immediately after creation.
        recovery.clear_recovery()

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

        if not getattr(self.mw, "current_project_file", None):
            # No project file yet. NEVER pop a dialog here — auto_save fires from
            # deep inside mutation handlers, and a modal there re-enters the event
            # loop mid-edit. Instead write a silent recovery snapshot the app
            # offers to restore on next launch (issue #41). Skip a trivially empty
            # session so a fresh launch never creates a recovery file.
            if self._project_is_trivially_empty():
                return
            try:
                recovery.write_recovery(self.build_project_data())
            except Exception:
                logger.exception("Failed to write unsaved-project recovery snapshot.")
            return

        self.save_project(show_message=False)
        logger.info("Project auto-saved.")

    def _project_is_trivially_empty(self):
        """True when nothing worth recovering has been done yet (#41)."""
        return (
            not self.mw.all_images
            and not self.mw.image_label.class_colors
            and not any(self.mw.all_annotations.values())
            and not self.mw.dino_phrase_panel.get_all_phrases()
            and not self.mw.dino_class_table.get_thresholds_dict()
        )

    def offer_recovery(self, settings=None):
        """On launch, offer to restore an unsaved-project recovery snapshot (#41).

        Fired from ``main()`` after the window is shown — never from the
        constructor, so tests that build ``ImageAnnotator()`` don't trigger it.
        On accept, the snapshot loads through the normal ``load_project_data``
        path but ``current_project_file`` is left unset, so the user still does a
        first real save (and continued edits keep writing fresh snapshots).
        """
        path = recovery.pending_recovery(settings)
        if not path:
            return
        try:
            mtime = datetime.fromtimestamp(os.path.getmtime(path)).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        except OSError:
            mtime = "an earlier session"
        reply = QMessageBox.question(
            self.mw,
            "Restore Unsaved Work",
            f"An unsaved project from a previous session was found "
            f"(last modified {mtime}). Restore it?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if reply != QMessageBox.StandardButton.Yes:
            recovery.clear_recovery(settings)
            return

        try:
            self.mw.is_loading_project = True
            with open(path, "r", encoding="utf-8") as f:
                project_data = json.load(f)
            problems = validate_project_data(project_data)
            if problems:
                raise ValueError("\n".join(problems))
            self.mw.clear_all(show_messages=False)
            self.load_project_data(project_data)
        except Exception:
            # A decode/validation failure (or a partial load) means the snapshot
            # is unusable — drop it so it isn't re-offered on every launch, and
            # reset to a clean empty state rather than a half-loaded one.
            logger.exception("Failed to restore recovery snapshot.")
            self.mw.clear_all(show_messages=False)
            recovery.clear_recovery(settings)
            QMessageBox.warning(
                self.mw,
                "Restore Failed",
                "The unsaved-project recovery file could not be restored.",
            )
        finally:
            self.mw.is_loading_project = False

        # On success the snapshot is deliberately KEPT: the restored project is
        # still unsaved (current_project_file is unset), so the first real save
        # retires it (save_project -> clear_recovery). Keeping it until then means
        # a re-crash before that save can still re-offer the recovered work.
