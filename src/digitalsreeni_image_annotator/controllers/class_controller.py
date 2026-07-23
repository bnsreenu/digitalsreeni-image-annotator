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

import traceback

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


class ClassController(QObject):
    def __init__(self, main_window):
        super().__init__(main_window)
        self.mw = main_window

    def select_class(self, index):
        if 0 <= index < self.mw.class_list.count():
            item = self.mw.class_list.item(index)
            self.mw.class_list.setCurrentItem(item)
            self.mw.current_class = item.text()
            print(f"Selected class: {self.mw.current_class}")
        else:
            print("Invalid class index")

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

    def add_class(self, class_name=None, color=None):
        if not self.mw.image_label.check_unsaved_changes():
            return

        if class_name is None:
            while True:
                class_name, ok = QInputDialog.getText(
                    self.mw, "Add Class", "Enter class name:"
                )
                if not ok:
                    print("Class addition cancelled")
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
                print(f"Class '{class_name}' already exists. Skipping addition.")
                return

        if not isinstance(class_name, str):
            print(
                f"Warning: class_name is not a string. Converting {class_name} to string."
            )
            class_name = str(class_name)

        if color is None:
            color = QColor(
                default_class_color(len(self.mw.image_label.class_colors))
            )
        elif isinstance(color, str):
            color = QColor(color)

        print(f"Adding class: {class_name}, color: {color.name()}")

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
            print(f"Class added successfully: {class_name}")

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
        except Exception as e:
            print(f"Error adding class: {e}")
            traceback.print_exc()

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

        print(f"Updated class list with {self.mw.class_list.count()} items")

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
            print(f"Class selected: {self.mw.current_class}")

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
        else:
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
        if ok and new_name and new_name != old_name:
            if old_name in self.mw.class_mapping:
                old_id = self.mw.class_mapping[old_name]
                self.mw.class_mapping[new_name] = old_id
                del self.mw.class_mapping[old_name]
            else:
                print(f"Warning: Class '{old_name}' not found in class_mapping")
                return

            if old_name in self.mw.image_label.class_colors:
                self.mw.image_label.class_colors[new_name] = (
                    self.mw.image_label.class_colors.pop(old_name)
                )
            else:
                print(f"Warning: Class '{old_name}' not found in class_colors")
                return

            # Keypoint schema follows the class name (issue #35).
            if old_name in self.mw.keypoint_schemas:
                self.mw.keypoint_schemas[new_name] = (
                    self.mw.keypoint_schemas.pop(old_name)
                )

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

            item.setText(new_name)

            self.mw.image_label.update()
            self.mw.auto_save()

            print(f"Class renamed from '{old_name}' to '{new_name}'")

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
