"""Class management controller (add / delete / rename / colour /
visibility) plus the slice-list colouring driven by per-slice
annotations.

Extracted from `ImageAnnotator`. Owns the class list widget plumbing,
context menu, programmatic and interactive class addition (with DINO
phrase-panel + threshold-table sync), and the slice-list colouring
that highlights annotated slices.

State stays on the main window (consistent with prior phases):
- `class_mapping` (dict[name, id])
- `image_label.class_colors`, `image_label.class_visibility`
- `current_class`
- `class_list`, `slice_list` widgets
- DINO widgets (`dino_class_table`, `dino_phrase_panel`)
"""

from PyQt6.QtCore import Qt, QObject
from PyQt6.QtGui import QColor, QIcon, QPixmap
from PyQt6.QtWidgets import (
    QColorDialog,
    QInputDialog,
    QListWidgetItem,
    QMenu,
    QMessageBox,
)

from ..core.constants import default_class_color

from ..core.logging_config import get_logger

logger = get_logger(__name__)


class ClassController(QObject):
    def __init__(self, main_window):
        super().__init__(main_window)
        self.mw = main_window

    def select_class(self, index):
        if 0 <= index < self.mw.class_list.count():
            item = self.mw.class_list.item(index)
            self.mw.class_list.setCurrentItem(item)
            self.mw.current_class = item.text()
            logger.debug(f"Selected class: {self.mw.current_class}")
        else:
            logger.warning("Invalid class index")

    def delete_selected_class(self):
        selected_items = self.mw.class_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(
                self.mw, "No Selection", "Please select a class to delete."
            )
            return

        class_name = selected_items[0].text()
        reply = QMessageBox.question(
            self.mw,
            "Delete Class",
            f"Are you sure you want to delete the class '{class_name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.delete_class(class_name)

    def update_slice_list_colors(self):
        if self.mw.dark_mode:
            self.mw.slice_list.setStyleSheet(
                "QListWidget { background-color: rgb(40, 40, 40); }"
            )
        else:
            self.mw.slice_list.setStyleSheet(
                "QListWidget { background-color: rgb(240, 240, 240); }"
            )

        for i in range(self.mw.slice_list.count()):
            item = self.mw.slice_list.item(i)
            slice_name = item.text()

            if self.mw.dark_mode:
                if slice_name in self.mw.all_annotations and any(
                    self.mw.all_annotations[slice_name].values()
                ):
                    item.setForeground(QColor(235, 235, 235))
                    item.setBackground(QColor(58, 95, 140))
                else:
                    item.setForeground(QColor(200, 200, 200))
                    item.setBackground(QColor(40, 40, 40))
            else:
                if slice_name in self.mw.all_annotations and any(
                    self.mw.all_annotations[slice_name].values()
                ):
                    item.setForeground(QColor(255, 255, 255))
                    item.setBackground(QColor(70, 130, 180))
                else:
                    item.setForeground(QColor(0, 0, 0))
                    item.setBackground(QColor(240, 240, 240))

        self.mw.slice_list.repaint()

        # Re-apply hook for the image-list annotation filter. Contract:
        # every annotation-mutation site either calls this method directly
        # or emits annotationsBatchSaved, whose handler
        # (_on_annotations_batch_saved) calls it. New mutation paths must
        # keep one of those two routes.
        self.mw.image_controller.apply_image_filter()

        # Repaint the video timeline's annotated-frame marks (issue #48). This
        # is THE mark-refresh choke point (runs on every annotation mutation AND
        # at the end of switch_image / switch_slice), so hooking here keeps the
        # timeline in sync with a single call rather than duplicating it in each
        # frame-switch path. No-op unless the active image is a video.
        self.mw.image_controller.update_video_timeline()

    def add_class(self, class_name=None, color=None):
        if not self.mw.image_label.check_unsaved_changes():
            return

        if class_name is None:
            while True:
                class_name, ok = QInputDialog.getText(
                    self.mw, "Add Class", "Enter class name:"
                )
                if not ok:
                    logger.debug("Class addition cancelled")
                    return
                if not class_name.strip():
                    QMessageBox.warning(
                        self.mw,
                        "Invalid Input",
                        "Please enter a class name or press Cancel.",
                    )
                    continue
                if class_name in self.mw.class_mapping:
                    QMessageBox.warning(
                        self.mw,
                        "Duplicate Class",
                        f"The class '{class_name}' already exists. Please choose a different name.",
                    )
                    continue
                break
        else:
            if class_name in self.mw.class_mapping:
                logger.warning(f"Class '{class_name}' already exists. Skipping addition.")
                return

        if not isinstance(class_name, str):
            logger.warning(
                f"class_name is not a string. Converting {class_name} to string."
            )
            class_name = str(class_name)

        if color is None:
            color = QColor(
                default_class_color(len(self.mw.image_label.class_colors))
            )
        elif isinstance(color, str):
            color = QColor(color)

        logger.debug(f"Adding class: {class_name}, color: {color.name()}")

        self.mw.image_label.class_colors[class_name] = color
        self.mw.class_mapping[class_name] = len(self.mw.class_mapping) + 1

        try:
            item = QListWidgetItem(class_name)

            pixmap = QPixmap(16, 16)
            pixmap.fill(color)
            item.setIcon(QIcon(pixmap))

            item.setData(Qt.ItemDataRole.UserRole, True)

            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)

            self.mw.class_list.addItem(item)

            self.mw.class_list.setCurrentItem(item)
            self.mw.current_class = class_name
            logger.info(f"Class added successfully: {class_name}")

            # DINO phrase/threshold sync. Skip the row-select during
            # project load (classes are added in a loop and we don't
            # want N row-selection signals firing during bulk restoration).
            row_added = self.mw.dino_class_table.add_class(class_name)
            self.mw.dino_phrase_panel.on_class_added(class_name)
            if row_added and not self.mw.is_loading_project:
                self.mw.dino_class_table.selectRow(
                    self.mw.dino_class_table.rowCount() - 1
                )

            if not self.mw.is_loading_project:
                self.mw.auto_save()
        except Exception:
            logger.exception("Error adding class")

    def update_class_item_color(self, item, color):
        pixmap = QPixmap(16, 16)
        pixmap.fill(color)
        item.setIcon(QIcon(pixmap))

    def update_class_list(self):
        self.mw.class_list.clear()
        for class_name, color in self.mw.image_label.class_colors.items():
            item = QListWidgetItem(class_name)

            pixmap = QPixmap(16, 16)
            pixmap.fill(color)
            item.setIcon(QIcon(pixmap))

            item.setData(
                Qt.ItemDataRole.UserRole,
                self.mw.image_label.class_visibility.get(class_name, True),
            )

            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked
                if item.data(Qt.ItemDataRole.UserRole)
                else Qt.CheckState.Unchecked
            )

            self.mw.class_list.addItem(item)

        if self.mw.current_class:
            items = self.mw.class_list.findItems(
                self.mw.current_class, Qt.MatchFlag.MatchExactly
            )
            if items:
                self.mw.class_list.setCurrentItem(items[0])
        elif self.mw.class_list.count() > 0:
            self.mw.class_list.setCurrentItem(self.mw.class_list.item(0))

        logger.debug(f"Updated class list with {self.mw.class_list.count()} items")

    def update_class_selection(self):
        for i in range(self.mw.class_list.count()):
            item = self.mw.class_list.item(i)
            if item.text() == self.mw.current_class:
                item.setSelected(True)
            else:
                item.setSelected(False)

    def toggle_class_visibility(self, item):
        class_name = item.text()
        is_visible = item.checkState() == Qt.CheckState.Checked
        self.mw.image_label.set_class_visibility(class_name, is_visible)
        item.setData(Qt.ItemDataRole.UserRole, is_visible)
        self.mw.image_label.update()

    def on_class_selected(self, current=None, previous=None):
        if not self.mw.image_label.check_unsaved_changes():
            return

        if current is None:
            current = self.mw.class_list.currentItem()

        if current:
            self.mw.current_class = current.text()
            logger.debug(f"Class selected: {self.mw.current_class}")

            # Keep the DINO phrase/threshold panel in sync with the top
            # class list -- the top list is the single source of truth.
            # Selecting the matching threshold-table row cascades to the
            # phrase panel via the table's itemSelectionChanged ->
            # on_dino_class_row_changed. A no-op for Temp-* classes (not in
            # the table). Skipped during project load to avoid signal churn
            # while classes are restored in bulk. (#63)
            if not self.mw.is_loading_project:
                self.mw.dino_class_table.select_class_by_name(self.mw.current_class)

            if self.mw.current_class.startswith("Temp-"):
                self.mw.disable_annotation_tools()
            else:
                self.mw.enable_annotation_tools()

            # The keypoint tool needs a pose schema on the current class; if
            # the newly-selected class has none, deactivate rather than
            # leaving the tool active-but-silently-inert (clicks would no-op
            # with no feedback). check_unsaved_changes() above already
            # committed/discarded any in-progress placement. (#35)
            if (
                self.mw.image_label.current_tool == "keypoint"
                and self.mw.current_class not in self.mw.keypoint_schemas
            ):
                self.mw.activate_tool(None)
            # Conversely, a pose class admits only the keypoint tool: if a
            # shape/SAM tool is active when a pose class is selected, deactivate
            # it so a tool can't stay active-but-invalid on a pose class. (#44)
            elif (
                self.mw.current_class in self.mw.keypoint_schemas
                and self.mw.image_label.current_tool is not None
                and self.mw.image_label.current_tool != "keypoint"
            ):
                self.mw.activate_tool(None)
        else:
            # Deliberately one-directional: an empty top-list selection leaves
            # the DINO table/panel showing the last class rather than blanking
            # the phrase editor. Sync follows *selection*, not deselection.
            self.mw.current_class = None
            self.mw.disable_annotation_tools()

    def show_class_context_menu(self, position):
        menu = QMenu()
        rename_action = menu.addAction("Rename Class")
        change_color_action = menu.addAction("Change Color")
        keypoint_action = menu.addAction("Define Keypoint Schema...")
        delete_action = menu.addAction("Delete Class")

        item = self.mw.class_list.itemAt(position)
        if item:
            action = menu.exec(self.mw.class_list.mapToGlobal(position))

            if action == rename_action:
                self.rename_class(item)
            elif action == change_color_action:
                self.change_class_color(item)
            elif action == keypoint_action:
                self.define_keypoint_schema(item)
            elif action == delete_action:
                self.delete_class(item)
        else:
            QMessageBox.warning(
                self.mw,
                "No Selection",
                "Please select a class to perform actions.",
            )

    def change_class_color(self, item):
        class_name = item.text()
        current_color = self.mw.image_label.class_colors.get(
            class_name, QColor(Qt.GlobalColor.white)
        )
        color = QColorDialog.getColor(
            current_color, self.mw, f"Select Color for {class_name}"
        )

        if color.isValid():
            self.mw.image_label.class_colors[class_name] = color

            pixmap = QPixmap(16, 16)
            pixmap.fill(color)
            item.setIcon(QIcon(pixmap))

            self.mw.update_annotation_list_colors(class_name, color)
            self.mw.image_label.update()
            self.mw.auto_save()

    def define_keypoint_schema(self, item):
        """Open the keypoint-schema editor for a class, making it a pose class
        (issue #35). The keypoint count is locked once instances exist."""
        from ..dialogs.keypoint_schema_dialog import KeypointSchemaDialog

        class_name = item.text()
        # A class is pose OR regular, not both (ADR-029). Refuse to turn a class
        # that already holds plain annotations into a pose class; editing the
        # schema of an existing pose class (even a legacy-mixed one) stays
        # allowed so names/skeleton can still be fixed. (#44)
        is_new_pose_class = class_name not in self.mw.keypoint_schemas
        if is_new_pose_class and self._class_has_plain_annotations(class_name):
            QMessageBox.warning(
                self.mw,
                "Cannot Define Keypoint Schema",
                f"'{class_name}' already has regular (polygon/box/paint) "
                "annotations. A class can be a pose class or a regular class, "
                "not both. Move or delete those annotations first, or create a "
                "new class for poses.",
            )
            return
        has_instances = self._class_has_keypoint_instances(class_name)
        dialog = KeypointSchemaDialog(
            self.mw,
            class_name=class_name,
            schema=self.mw.keypoint_schemas.get(class_name),
            lock_k=has_instances,
        )
        if dialog.exec() != dialog.DialogCode.Accepted:
            return
        schema = dialog.get_schema()
        if schema is None:
            return
        # Defensive: the dialog locks add/remove when instances exist, but make
        # the K-stability invariant explicit so a future dialog change can't
        # silently corrupt existing instances.
        if has_instances:
            old_k = len((self.mw.keypoint_schemas.get(class_name) or {}).get("names", []))
            if old_k and len(schema["names"]) != old_k:
                QMessageBox.warning(
                    self.mw,
                    "Keypoint Schema",
                    "Cannot change the number of keypoints while instances exist.",
                )
                return
        self.mw.keypoint_schemas[class_name] = schema
        self.mw.image_label.update()
        if not self.mw.is_loading_project:
            self.mw.auto_save()

    def _class_has_plain_annotations(self, class_name):
        """True if the class holds any non-keypoint (polygon/box/paint)
        annotation. A class is pose OR regular, not both (ADR-029 / #44)."""
        for image_annotations in self.mw.all_annotations.values():
            for ann in image_annotations.get(class_name, []):
                if "keypoints" not in ann:
                    return True
        return False

    def _class_has_keypoint_instances(self, class_name):
        for image_annotations in self.mw.all_annotations.values():
            for ann in image_annotations.get(class_name, []):
                if "keypoints" in ann:
                    return True
        return False

    def rename_class(self, item):
        old_name = item.text()
        new_name, ok = QInputDialog.getText(
            self.mw, "Rename Class", "Enter new class name:", text=old_name
        )
        # Strip before every guard: "Drone " would otherwise clear both the
        # collision check and the Temp- prefix check, creating a class
        # distinguishable from "Drone" only by trailing whitespace -- which
        # then keys all seven registries. The phrase editor already sanitises
        # its input the same way (_add_phrase / _rename_phrase).
        if new_name:
            new_name = new_name.strip()
        if ok and new_name and new_name != old_name:
            # --- Pre-flight ------------------------------------------------
            # Validate every precondition BEFORE mutating anything. A rename
            # touches seven name-keyed registries (docs/08 "Class Name Is a
            # Primary Key"); bailing out partway leaves the class renamed in
            # some and not others. Check first, then commit.

            # A pending Temp-* review class isn't in class_mapping, so it would
            # otherwise fall into the bailout below and fail *silently* -- the
            # one rejection a user is most likely to hit by accident. Every
            # other path in this block tells them why.
            if old_name.startswith("Temp-"):
                QMessageBox.warning(
                    self.mw,
                    "Pending Review Class",
                    "Pending detection-review classes can't be renamed. "
                    "Accept or reject them first.",
                )
                return

            if old_name not in self.mw.class_mapping:
                logger.warning(f"Class '{old_name}' not found in class_mapping")
                return
            if old_name not in self.mw.image_label.class_colors:
                logger.warning(f"Class '{old_name}' not found in class_colors")
                return

            # Collision check against class_colors, NOT class_mapping:
            # Temp-* review classes are registered in class_colors + class_list
            # only (add_temp_classes never writes class_mapping), so a
            # class_mapping check lets a rename onto a *pending* temp class
            # through -- the real class's annotation bucket then merges into
            # the temp one, and the next Reject sweeps Temp-* and deletes it.
            # Without any check the rename half-clobbers every registry: the
            # old class's id and colour overwrite the target's, buckets merge,
            # and the DINO table ends up with two rows of one name (duplicate
            # detection passes + one row's thresholds silently dropped).
            if new_name in self.mw.image_label.class_colors:
                QMessageBox.warning(
                    self.mw,
                    "Duplicate Class",
                    f"A class named '{new_name}' already exists. "
                    "Please choose a different name.",
                )
                return

            # "Temp-" is a reserved prefix with destructive semantics -- the
            # DINO review sweeps Temp-* wholesale on accept/reject -- so a
            # class renamed into that namespace would be silently consumed by
            # the next review.
            if new_name.startswith("Temp-"):
                QMessageBox.warning(
                    self.mw,
                    "Reserved Name",
                    "Names starting with 'Temp-' are reserved for pending "
                    "detection-review classes. Please choose a different name.",
                )
                return

            # --- Commit ----------------------------------------------------
            self.mw.class_mapping[new_name] = self.mw.class_mapping.pop(old_name)

            self.mw.image_label.class_colors[new_name] = (
                self.mw.image_label.class_colors.pop(old_name)
            )

            # Visibility is name-keyed too (read via .get(name, True)), so a
            # rename that skips it silently un-hides a hidden class.
            if old_name in self.mw.image_label.class_visibility:
                self.mw.image_label.class_visibility[new_name] = (
                    self.mw.image_label.class_visibility.pop(old_name)
                )

            # Keypoint schema follows the class name (issue #35).
            if old_name in self.mw.keypoint_schemas:
                self.mw.keypoint_schemas[new_name] = (
                    self.mw.keypoint_schemas.pop(old_name)
                )

            # The DINO threshold row and phrase list are keyed by class name
            # too. Without this they stay under the dead name: detection runs
            # against a class that no longer exists, and the next project load
            # silently discards the renamed class's phrases *and* thresholds
            # (both are filtered against the live class list). Same
            # registry-drift family as #63.
            self.mw.dino_class_table.rename_class(old_name, new_name)
            self.mw.dino_phrase_panel.on_class_renamed(old_name, new_name)

            for image_name, image_annotations in self.mw.all_annotations.items():
                if old_name in image_annotations:
                    image_annotations[new_name] = image_annotations.pop(old_name)
                    for annotation in image_annotations[new_name]:
                        annotation["category_name"] = new_name

            if old_name in self.mw.image_label.annotations:
                self.mw.image_label.annotations[new_name] = (
                    self.mw.image_label.annotations.pop(old_name)
                )
                for annotation in self.mw.image_label.annotations[new_name]:
                    annotation["category_name"] = new_name

            if self.mw.current_class == old_name:
                self.mw.current_class = new_name

            self.mw.update_all_annotation_lists()

            # Block signals: setText emits itemChanged -> toggle_class_visibility,
            # which re-derives class_visibility from the checkbox and would be a
            # hidden co-author of the re-key above (masking a broken re-key).
            # A rename changes the name, never the visibility.
            self.mw.class_list.blockSignals(True)
            try:
                item.setText(new_name)
            finally:
                self.mw.class_list.blockSignals(False)

            self.mw.image_label.update()
            self.mw.auto_save()

            logger.info(f"Class renamed from '{old_name}' to '{new_name}'")

    def delete_class(self, item=None):
        if item is None:
            item = self.mw.class_list.currentItem()

        if item is None:
            QMessageBox.warning(
                self.mw, "No Selection", "Please select a class to delete."
            )
            return

        # delete_selected_class calls self.delete_class(class_name) with a
        # string instead of a QListWidgetItem — handle both. The
        # show_class_context_menu / Delete key path passes a QListWidgetItem,
        # while delete_selected_class passes the class name string.
        if isinstance(item, str):
            class_name = item
            row_items = self.mw.class_list.findItems(class_name, Qt.MatchFlag.MatchExactly)
            list_item = row_items[0] if row_items else None
        else:
            class_name = item.text()
            list_item = item

        reply = QMessageBox.question(
            self.mw,
            "Delete Class",
            f"Are you sure you want to delete the class '{class_name}'?\n\n"
            "This will remove all annotations associated with this class.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.mw.image_label.class_colors.pop(class_name, None)
            self.mw.class_mapping.pop(class_name, None)
            self.mw.keypoint_schemas.pop(class_name, None)  # issue #35

            for image_annotations in self.mw.all_annotations.values():
                image_annotations.pop(class_name, None)

            self.mw.image_label.annotations.pop(class_name, None)

            self.mw.dino_class_table.remove_class(class_name)
            self.mw.dino_phrase_panel.on_class_removed(class_name)

            self.mw.update_annotation_list()

            if list_item is not None:
                row = self.mw.class_list.row(list_item)
                self.mw.class_list.takeItem(row)

            if self.mw.current_class == class_name:
                self.mw.current_class = None
                if self.mw.class_list.count() > 0:
                    self.mw.class_list.setCurrentRow(0)
                    self.on_class_selected(self.mw.class_list.item(0))
                else:
                    self.mw.disable_annotation_tools()

            self.mw.image_label.update()

            QMessageBox.information(
                self.mw,
                "Class Deleted",
                f"The class '{class_name}' has been deleted.",
            )
            self.mw.auto_save()
        else:
            QMessageBox.information(
                self.mw,
                "Deletion Cancelled",
                "The class deletion was cancelled.",
            )

    def is_class_visible(self, class_name):
        items = self.mw.class_list.findItems(class_name, Qt.MatchFlag.MatchExactly)
        if items:
            return items[0].checkState() == Qt.CheckState.Checked
        return False
