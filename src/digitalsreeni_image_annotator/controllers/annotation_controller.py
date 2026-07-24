"""Annotation CRUD + UI list management controller.

Extracted from `ImageAnnotator`. Owns the annotation list widget
plumbing, the per-image annotation cache sync (`load_image_annotations`
/ `save_current_annotations`), polygon and rectangle commit paths,
merge/delete/change-class workflows, sort & renumber, the COCO-load
path, and the edit-mode lifecycle.

This is the cluster `ImageLabel` mutates most directly (via
`main_window.add_annotation_to_list(...)` etc.). Phase 5 keeps the
delegation pattern on `ImageAnnotator`; Phase 6 will replace
`ImageLabel`'s `main_window.*` calls with Qt signals targeting these
controller methods.

State stays on the main window:
- `all_annotations` (dict[image_name, dict[class_name, list[ann]]])
- `image_label.annotations` (per-image working copy)
- `editing_mode`, `loaded_json`, `current_sort_method`
- All Qt widgets (`annotation_list`, `merge_button`, `change_class_button`)
"""

import copy
import json

from PyQt6.QtCore import Qt, QObject
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QListWidgetItem,
    QMessageBox,
    QSpinBox,
    QTableWidgetItem,
    QTableWidgetSelectionRange,
    QVBoxLayout,
)
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

from ..core.constants import (
    ANNOT_COL_AREA,
    ANNOT_COL_CLASS,
    ANNOT_COL_DETAIL,
    ANNOT_COL_ID,
    default_class_color,
)
from ..io.export_formats import create_coco_annotation as _export_create_coco_annotation
from ..utils import (
    calculate_area,
    calculate_bbox,
    clamp_bbox,
    clamp_keypoints,
    keypoint_instance_bbox,
    simplify_polygon,
)
from .annotation_history import AnnotationHistory

from ..core.logging_config import get_logger

logger = get_logger(__name__)


class AnnotationController(QObject):
    def __init__(self, main_window):
        super().__init__(main_window)
        self.mw = main_window
        # Snapshot-based undo/redo of annotation edits (ADR-026). Per-image
        # stacks keyed by current_slice or image_file_name.
        self.history = AnnotationHistory()
        # Detail-% drags fire valueChanged per step; coalesce a run on one
        # annotation into a single history entry (token = id of that ann).
        self._detail_coalesce_key = None
        # Baseline captured at the *start* of a deferred gesture (bbox drag,
        # paint stroke) whose commit notifies us only after mutating in place.
        self._pending_baseline = None

    # --- COCO conversion helper ---

    def create_coco_annotation(self, ann, image_id, annotation_id):
        """Delegate to the live io.export_formats implementation (issue #35
        PR-2 review) — this method used to be a near-duplicate that had
        drifted keypoint-blind. Reconstructs a single-entry class_mapping
        from the ann's own category fields since callers here don't have
        one handy."""
        class_name = ann.get("category_name")
        class_mapping = {class_name: ann["category_id"]}
        return _export_create_coco_annotation(ann, image_id, annotation_id, class_name, class_mapping)

    # --- List widget updates ---

    def update_all_annotation_lists(self):
        for image_name in self.mw.all_annotations.keys():
            self.update_annotation_list(image_name)
        self.update_annotation_list()

    # --- Annotations table (issue #24): ID | Class | Area | Detail % ---
    # The table replaces the old QListWidget. Column 0 (ID) carries the
    # annotation dict in its UserRole — the value-equality marker the canvas ↔
    # list selection bridge reads (ADR-022). The Detail % spinbox per row drives
    # reversible polygon simplification (ADR-025).

    def _make_detail_spin(self, annotation):
        """A per-row 1..100 Detail % spinbox (100 = raw). Disabled for
        annotations with no polygon to simplify (bbox-only imports)."""
        sp = QSpinBox()
        sp.setRange(1, 100)
        sp.setSuffix(" %")
        has_seg = bool(annotation.get("segmentation"))
        sp.setValue(int(annotation.get("detail_pct", 100)) if has_seg else 100)
        sp.setEnabled(has_seg)
        sp.setFrame(True)
        return sp

    def _insert_annotation_row(self, annotation, color):
        """Append one annotation as a table row. valueChanged is connected
        *after* the initial setValue so building the table never fires the
        simplification handler."""
        tbl = self.mw.annotation_list
        row = tbl.rowCount()
        tbl.insertRow(row)

        id_item = QTableWidgetItem(str(annotation.get("number", 0)))
        id_item.setData(Qt.ItemDataRole.UserRole, annotation)
        id_item.setForeground(color)
        tbl.setItem(row, ANNOT_COL_ID, id_item)

        class_item = QTableWidgetItem(annotation["category_name"])
        class_item.setForeground(color)
        tbl.setItem(row, ANNOT_COL_CLASS, class_item)

        area_item = QTableWidgetItem(f"{calculate_area(annotation):.2f}")
        area_item.setForeground(color)
        tbl.setItem(row, ANNOT_COL_AREA, area_item)

        spin = self._make_detail_spin(annotation)
        spin.valueChanged.connect(lambda val, r=row: self.on_detail_pct_changed(r, val))
        tbl.setCellWidget(row, ANNOT_COL_DETAIL, spin)

    def _selected_row_items(self):
        """Col-0 items of the selected rows, deduped. The table selects whole
        rows, but selectedItems()/selectedIndexes() yield a cell per column."""
        tbl = self.mw.annotation_list
        rows = sorted({idx.row() for idx in tbl.selectedIndexes()})
        items = [tbl.item(r, ANNOT_COL_ID) for r in rows]
        return [it for it in items if it is not None]

    def update_annotation_list(self, image_name=None):
        self.mw.annotation_list.setRowCount(0)
        current_name = image_name or self.mw.current_slice or self.mw.image_file_name
        annotations = self.mw.all_annotations.get(current_name, {})
        for class_name, class_annotations in annotations.items():
            if not class_name.startswith("Temp-"):
                color = self.mw.image_label.class_colors.get(
                    class_name, QColor(Qt.GlobalColor.white)
                )
                for annotation in class_annotations:
                    self._insert_annotation_row(annotation, color)

    def update_annotation_list_colors(self, class_name=None, color=None):
        tbl = self.mw.annotation_list
        for i in range(tbl.rowCount()):
            id_item = tbl.item(i, ANNOT_COL_ID)
            if id_item is None:
                continue
            annotation = id_item.data(Qt.ItemDataRole.UserRole)
            if class_name is None or annotation["category_name"] == class_name:
                item_color = (
                    color
                    if class_name
                    else self.mw.image_label.class_colors.get(
                        annotation["category_name"], QColor(Qt.GlobalColor.white)
                    )
                )
                for c in (ANNOT_COL_ID, ANNOT_COL_CLASS, ANNOT_COL_AREA):
                    cell = tbl.item(i, c)
                    if cell is not None:
                        cell.setForeground(item_color)

    def update_annotation_list_with_sorted(self, sorted_annotations):
        self.mw.annotation_list.setRowCount(0)
        for annotation in sorted_annotations:
            class_name = annotation["category_name"]
            if not class_name.startswith("Temp-"):
                color = self.mw.image_label.class_colors.get(
                    class_name, QColor(Qt.GlobalColor.white)
                )
                self._insert_annotation_row(annotation, color)

        self.mw.image_label.update()

    def on_detail_pct_changed(self, row, pct):
        """A row's Detail % spinbox changed → re-simplify that annotation's
        polygon from its raw (full-precision) copy. Reversible: 100 % restores
        raw exactly; the raw is lazy-captured the first time a mask is thinned.
        (issue #24)"""
        tbl = self.mw.annotation_list
        id_item = tbl.item(row, ANNOT_COL_ID)
        if id_item is None:
            return
        captured = id_item.data(Qt.ItemDataRole.UserRole)
        # Resolve to the live drawn object so the in-place edit is what gets
        # rendered + saved (a list/table copy would be lost — see #40).
        live = self.mw.image_label._live_annotation(captured)
        if live is None or not live.get("segmentation"):
            return

        # Coalesce a whole spinbox drag on one annotation into a single undo
        # entry: record the pre-drag state once, then suppress until the run
        # moves to a different annotation (or any other edit clears the token).
        token = (self._history_key(), live.get("number"), live.get("category_name"))
        if token != self._detail_coalesce_key:
            self.record_history()  # resets _detail_coalesce_key to None
            self._detail_coalesce_key = token

        if pct >= 100:
            raw = live.get("segmentation_raw")
            if raw:
                live["segmentation"] = list(raw)
            live["detail_pct"] = 100
        else:
            if not live.get("segmentation_raw"):
                live["segmentation_raw"] = list(live["segmentation"])
            live["segmentation"] = simplify_polygon(live["segmentation_raw"], pct)
            live["detail_pct"] = pct
        if live.get("bbox") is not None:
            live["bbox"] = calculate_bbox(live["segmentation"])

        # Refresh this row in place (no rebuild — keeps the spinbox stable): the
        # UserRole + Area must track the new value so repeated edits resolve and
        # the selection bridge keeps matching.
        id_item.setData(Qt.ItemDataRole.UserRole, dict(live))
        area_item = tbl.item(row, ANNOT_COL_AREA)
        if area_item is not None:
            area_item.setText(f"{calculate_area(live):.2f}")

        # Re-point the canvas selection at the mutated live object. Otherwise
        # highlighted_annotations still holds the pre-simplify value, so the
        # selection overlay draws stale geometry and a subsequent #40 handle
        # drag (_live_annotation) can't re-match the row → edit lost.
        hl = self.mw.image_label.highlighted_annotations
        for i, a in enumerate(hl):
            if a == captured:
                hl[i] = live

        self.mw.image_label.update()
        self.save_current_annotations()
        self.mw.auto_save()

    # --- Per-image annotation cache sync ---

    def load_image_annotations(self):
        self.mw.image_label.annotations.clear()
        current_name = self.mw.current_slice or self.mw.image_file_name
        if current_name in self.mw.all_annotations:
            self.mw.image_label.annotations = copy.deepcopy(
                self.mw.all_annotations[current_name]
            )
        else:
            logger.debug(f"No annotations found for {current_name}")
        self.mw.image_label.update()

    def save_current_annotations(self):
        if self.mw.current_slice:
            current_name = self.mw.current_slice
        elif self.mw.image_file_name:
            current_name = self.mw.image_file_name
        else:
            return

        if self.mw.image_label.annotations:
            self.mw.all_annotations[current_name] = (
                self.mw.image_label.annotations.copy()
            )
        elif current_name in self.mw.all_annotations:
            del self.mw.all_annotations[current_name]

        self.mw.update_slice_list_colors()

    def replace_annotations(self, image_key: str, annotations: dict) -> None:
        """Replace the full per-class annotation dict for one image.
        Used by the eraser path which has already cut polygons in
        ImageLabel.annotations. Triggers list refresh, save, and slice
        colour update atomically."""
        # Eraser already mutated ImageLabel.annotations in place, so the
        # pre-cut state lives in all_annotations until this overwrite. Record
        # it for undo before replacing.
        self.record_history(image_key)
        self.mw.all_annotations[image_key] = annotations
        self.update_annotation_list()
        self.save_current_annotations()
        self.mw.class_controller.update_slice_list_colors()

    # --- Undo / redo (ADR-026) ---

    def _history_key(self):
        return self.mw.current_slice or self.mw.image_file_name

    def record_history(self, key=None):
        """Snapshot the pre-mutation state of one image for undo.

        Call *before* a synchronous mutation. ``key`` defaults to the current
        image; pass an explicit key for off-screen writes (e.g. DINO batch
        commits to an image other than the one on screen). Skipped during
        project load so restoring a project never seeds bogus history.
        """
        if self.mw.is_loading_project:
            return
        key = key or self._history_key()
        if not key:
            return
        snapshot = copy.deepcopy(self.mw.all_annotations.get(key, {}))
        self.history.record(key, snapshot)
        # Any explicit edit ends a Detail-% coalescing run and drops any stale
        # deferred-gesture baseline (e.g. a discarded paint stroke).
        self._detail_coalesce_key = None
        self._pending_baseline = None

    def capture_edit_baseline(self):
        """Remember the pre-gesture state at the *start* of a bbox drag or
        paint stroke. The commit notification arrives only after ImageLabel
        has mutated in place, so we cannot snapshot the 'before' there."""
        if self.mw.is_loading_project:
            return
        key = self._history_key()
        if not key:
            return
        self._pending_baseline = (key, copy.deepcopy(self.mw.all_annotations.get(key, {})))

    def commit_edit_baseline(self):
        """Push the baseline captured by capture_edit_baseline onto the undo
        stack. Called when a deferred gesture actually commits. The history
        dedup drops it if nothing changed (aborted drag / empty stroke)."""
        if self._pending_baseline is None:
            return
        key, snapshot = self._pending_baseline
        self._pending_baseline = None
        if self.mw.is_loading_project:
            return
        # Only commit a baseline for the image still on screen; a stale baseline
        # from a different image (e.g. after a switch) must not be pushed.
        if key != self._history_key():
            return
        self.history.record(key, snapshot)
        self._detail_coalesce_key = None

    def reset_coalesce(self):
        """Drop any in-progress Detail-% coalescing token (on image switch)."""
        self._detail_coalesce_key = None

    def clear_history(self):
        self.history.clear()
        self._detail_coalesce_key = None
        self._pending_baseline = None

    def _undo_blocked(self):
        """Undo/redo are no-ops while a project loads, a modal is open, a text
        field has focus, or an edit/draw gesture is in flight."""
        if self.mw.is_loading_project:
            return True
        from PyQt6.QtWidgets import QApplication, QLineEdit, QTextEdit

        if QApplication.activeModalWidget() is not None:
            return True
        focus = QApplication.focusWidget()
        if isinstance(focus, (QLineEdit, QTextEdit)):
            return True
        il = self.mw.image_label
        # Any in-flight draw/edit gesture blocks undo/redo. Paint/eraser are
        # included because a mid-stroke undo would otherwise restore a snapshot
        # while a deferred baseline is still pending, corrupting the next push.
        if (
            il.editing_polygon
            or getattr(il, "drawing_polygon", False)
            or getattr(il, "current_annotation", None)
            or getattr(il, "current_rectangle", None)
            or getattr(il, "temp_sam_prediction", None)
            or getattr(il, "temp_annotations", None)
            or getattr(il, "bbox_edit", None) is not None
            or getattr(il, "temp_paint_mask", None) is not None
            or getattr(il, "is_painting", False)
            or getattr(il, "temp_eraser_mask", None) is not None
            or getattr(il, "is_erasing", False)
            or getattr(il, "drawing_sam_bbox", False)
            or getattr(il, "drawing_keypoints", False)
            or getattr(il, "editing_keypoint", None) is not None
        ):
            return True
        return False

    def undo(self):
        if self._undo_blocked():
            return
        key = self._history_key()
        if not key or not self.history.can_undo(key):
            return
        current = copy.deepcopy(self.mw.all_annotations.get(key, {}))
        snapshot = self.history.undo(key, current)
        if snapshot is not None:
            self._restore_snapshot(key, snapshot)

    def redo(self):
        if self._undo_blocked():
            return
        key = self._history_key()
        if not key or not self.history.can_redo(key):
            return
        current = copy.deepcopy(self.mw.all_annotations.get(key, {}))
        snapshot = self.history.redo(key, current)
        if snapshot is not None:
            self._restore_snapshot(key, snapshot)

    def _restore_snapshot(self, key, snapshot):
        """Apply a whole-image snapshot back onto the live model and refresh
        the canvas + list. Independent deep copies break the shallow-copy
        aliasing between all_annotations and image_label.annotations.

        The snapshot is restored verbatim — no renumbering. It already holds a
        previously-consistent numbering, and renumbering only one of the two
        copies would skew the table's UserRole numbers against the persisted
        model (breaking value-equality selection matching). See ADR-026.
        """
        self.mw.all_annotations[key] = copy.deepcopy(snapshot)
        self.mw.image_label.annotations = copy.deepcopy(snapshot)
        self.mw.image_label.highlighted_annotations.clear()
        self._sync_selection_buttons(0)
        self._detail_coalesce_key = None
        # Drop any deferred-gesture baseline so a mid-gesture undo can't leak a
        # stale "before" into the next commit.
        self._pending_baseline = None
        self.update_annotation_list()  # rebuild table from the restored dict
        self.mw.image_label.update()
        # save_current_annotations reconciles all_annotations (deleting the key
        # if the restored state is empty) and refreshes slice colours.
        self.save_current_annotations()
        self.mw.update_slice_list_colors()
        self.mw.auto_save()

    # --- Sorting ---

    def sort_annotations_by_class(self):
        current_name = self.mw.current_slice or self.mw.image_file_name
        if current_name not in self.mw.all_annotations:
            QMessageBox.information(
                self.mw,
                "No Annotations",
                "There are no annotations to sort for this image.",
            )
            return

        annotations = self.mw.all_annotations[current_name]
        sorted_annotations = []
        for class_name in sorted(annotations.keys()):
            if not class_name.startswith("Temp-"):
                class_annotations = sorted(
                    annotations[class_name], key=lambda x: x.get("number", 0)
                )
                sorted_annotations.extend(class_annotations)

        self.update_annotation_list_with_sorted(sorted_annotations)

    def sort_annotations_by_area(self):
        current_name = self.mw.current_slice or self.mw.image_file_name
        if current_name not in self.mw.all_annotations:
            QMessageBox.information(
                self.mw,
                "No Annotations",
                "There are no annotations to sort for this image.",
            )
            return

        annotations = self.mw.all_annotations[current_name]
        sorted_annotations = []
        for class_name in annotations.keys():
            if not class_name.startswith("Temp-"):
                class_annotations = sorted(
                    annotations[class_name],
                    key=lambda x: calculate_area(x),
                    reverse=True,
                )
                sorted_annotations.extend(class_annotations)

        self.update_annotation_list_with_sorted(sorted_annotations)

    # --- COCO JSON load (independent of project save/load) ---

    def load_annotations(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self.mw, "Load Annotations", "", "JSON Files (*.json)"
        )
        if not file_name:
            return

        with open(file_name, "r", encoding='utf-8') as f:
            self.mw.loaded_json = json.load(f)

        self.mw.class_list.clear()
        self.mw.image_label.class_colors.clear()
        self.mw.class_mapping.clear()
        for category in self.mw.loaded_json["categories"]:
            class_name = category["name"]
            self.mw.class_mapping[class_name] = category["id"]

            if class_name not in self.mw.image_label.class_colors:
                color = QColor(
                    default_class_color(len(self.mw.image_label.class_colors))
                )
                self.mw.image_label.class_colors[class_name] = color

            item = QListWidgetItem(class_name)
            self.mw.update_class_item_color(
                item, self.mw.image_label.class_colors[class_name]
            )
            self.mw.class_list.addItem(item)

        image_id_to_filename = {
            img["id"]: img["file_name"] for img in self.mw.loaded_json["images"]
        }

        json_images = {img["file_name"]: img for img in self.mw.loaded_json["images"]}

        updated_all_images = []
        for i in range(self.mw.image_list.count()):
            item = self.mw.image_list.item(i)
            file_name = item.text()
            if file_name in json_images:
                updated_image = self.mw.all_images[i].copy()
                updated_image.update(json_images[file_name])
                updated_all_images.append(updated_image)
                del json_images[file_name]
            else:
                updated_all_images.append(self.mw.all_images[i])

        for img in json_images.values():
            updated_all_images.append(img)

        self.mw.all_images = updated_all_images
        # Rebuild the list in sorted order (issue #60). The reconciliation
        # loop above already consumed the pre-sort row/index alignment.
        self.mw.update_image_list()

        self.mw.all_annotations.clear()
        for annotation in self.mw.loaded_json["annotations"]:
            image_id = annotation["image_id"]
            file_name = image_id_to_filename.get(image_id)
            if file_name:
                if file_name not in self.mw.all_annotations:
                    self.mw.all_annotations[file_name] = {}

                category = next(
                    (
                        cat
                        for cat in self.mw.loaded_json["categories"]
                        if cat["id"] == annotation["category_id"]
                    ),
                    None,
                )
                if category:
                    category_name = category["name"]
                    if category_name not in self.mw.all_annotations[file_name]:
                        self.mw.all_annotations[file_name][category_name] = []

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

                    if "number" not in ann:
                        ann["number"] = (
                            len(self.mw.all_annotations[file_name][category_name]) + 1
                        )

                    self.mw.all_annotations[file_name][category_name].append(ann)

        missing_images = [
            img["file_name"]
            for img in self.mw.loaded_json["images"]
            if img["file_name"] not in self.mw.image_paths
        ]
        if missing_images:
            self.mw.show_warning(
                "Missing Images",
                "The following images are missing:\n" + "\n".join(missing_images),
            )

        if self.mw.image_file_name and self.mw.image_file_name in self.mw.all_annotations:
            self.mw.switch_image(
                self.mw.image_list.findItems(
                    self.mw.image_file_name, Qt.MatchFlag.MatchExactly
                )[0]
            )
        elif self.mw.all_images:
            self.mw.switch_image(self.mw.image_list.item(0))

        self.mw.image_label.highlighted_annotations = []
        self.update_annotation_list()
        self.mw.image_label.update()

    # --- Highlighting / selection ---

    def clear_highlighted_annotation(self):
        self.mw.image_label.highlighted_annotations.clear()
        # Selection is gone — Merge/Change Class must follow, or they linger
        # enabled against an empty list selection after an image/slice switch.
        self._sync_selection_buttons(0)
        self.mw.image_label.update()

    def update_highlighted_annotations(self):
        selected_items = self._selected_row_items()
        self.mw.image_label.highlighted_annotations = [
            item.data(Qt.ItemDataRole.UserRole) for item in selected_items
        ]
        self.mw.image_label.update()
        self._sync_selection_buttons(len(selected_items))

    def _sync_selection_buttons(self, count):
        """Merge needs ≥2 annotations; Change Class needs ≥1. Shared by the
        list-driven and canvas-driven (issue #75) selection paths."""
        self.mw.merge_button.setEnabled(count >= 2)
        self.mw.change_class_button.setEnabled(count > 0)

    def apply_canvas_selection(self, annotations, mode):
        """Apply a selection change that originated on the canvas (issue #75)
        and mirror it onto the annotation list so Delete / Merge / Change
        Class operate on the same set. Matching uses dict value-equality,
        consistent with the rest of the selection code.

        ``mode`` is one of ``"replace"``, ``"add"``, ``"toggle"``.
        """
        current = list(self.mw.image_label.highlighted_annotations)

        def contains(seq, ann):
            return any(a == ann for a in seq)

        if mode == "replace":
            new = list(annotations)
        elif mode == "add":
            new = current + [a for a in annotations if not contains(current, a)]
        elif mode == "toggle":
            new = list(current)
            for a in annotations:
                match = next((x for x in new if x == a), None)
                if match is not None:
                    new.remove(match)
                else:
                    new.append(a)
        else:
            return

        self.mw.image_label.highlighted_annotations = new

        # Mirror onto the table. Block signals so the programmatic selection
        # doesn't retrigger itemSelectionChanged → update_highlighted_annotations,
        # which would overwrite `new` with the table items' own (all_annotations)
        # object identities. Match by value-equality on col 0's UserRole.
        tbl = self.mw.annotation_list
        tbl.blockSignals(True)
        tbl.clearSelection()
        last_col = tbl.columnCount() - 1
        for i in range(tbl.rowCount()):
            item = tbl.item(i, ANNOT_COL_ID)
            if item is not None and contains(new, item.data(Qt.ItemDataRole.UserRole)):
                # setRangeSelected is additive; selectRow() would *replace* the
                # selection in ExtendedSelection mode, dropping all but the last.
                tbl.setRangeSelected(
                    QTableWidgetSelectionRange(i, 0, i, last_col), True
                )
        tbl.blockSignals(False)

        self._sync_selection_buttons(len(new))
        self.mw.image_label.update()

    def commit_bbox_edit(self):
        """Persist a bbox resize/move performed directly on the canvas
        (issue #40). ImageLabel already mutated + clamped the bbox in place, so
        we save (pushing the new coords into all_annotations), rebuild the list
        so the displayed area refreshes, then re-mirror the selection so the
        edited box stays selected and list/canvas stay in sync."""
        selected = list(self.mw.image_label.highlighted_annotations)
        # The drag mutated ImageLabel.annotations in place; push the baseline
        # captured at gesture start so the move/resize is undoable (ADR-026).
        self.commit_edit_baseline()
        self.save_current_annotations()
        self.update_annotation_list()
        self.apply_canvas_selection(selected, "replace")
        self.mw.auto_save()

    def commit_polygon_edit(self):
        """Persist a committed vertex edit (double-click polygon edit, Enter).

        The edit mutated ImageLabel.annotations in place but — unlike every
        other edit path — its commit historically did NOT sync all_annotations.
        We sync here so the edit persists reliably, then push the baseline
        captured at edit-mode entry so Ctrl+Z reverts it (ADR-026)."""
        self.commit_edit_baseline()
        self.save_current_annotations()
        self.update_annotation_list()
        self.mw.update_slice_list_colors()
        self.mw.auto_save()

    def commit_keypoint_edit(self):
        """Persist a single-keypoint drag or visibility toggle on a committed
        pose instance (#35). ImageLabel mutated the keypoints in place; recompute
        num_keypoints, refresh the list, push the gesture's undo baseline
        (ADR-026), then re-mirror the selection so the instance stays selected.

        The recompute is a no-op for both current gestures (a drag only moves an
        already-labelled point; a toggle only flips between v=1/v=2) — kept as a
        defensive safety net so num_keypoints can't silently drift out of sync
        with a future edit path that does change which points are labelled."""
        selected = list(self.mw.image_label.highlighted_annotations)
        for ann in selected:
            kps = ann.get("keypoints")
            if kps is not None:
                ann["num_keypoints"] = sum(
                    1 for i in range(2, len(kps), 3) if kps[i] > 0
                )
        self.commit_edit_baseline()
        self.save_current_annotations()
        self.update_annotation_list()
        self.apply_canvas_selection(selected, "replace")
        self.mw.update_slice_list_colors()
        self.mw.auto_save()

    def highlight_annotation_in_list(self, annotation):
        tbl = self.mw.annotation_list
        for i in range(tbl.rowCount()):
            item = tbl.item(i, ANNOT_COL_ID)
            if item is not None and item.data(Qt.ItemDataRole.UserRole) == annotation:
                tbl.selectRow(i)
                break

    def select_annotation_in_list(self, annotation):
        tbl = self.mw.annotation_list
        for i in range(tbl.rowCount()):
            item = tbl.item(i, ANNOT_COL_ID)
            if item is not None and item.data(Qt.ItemDataRole.UserRole) == annotation:
                tbl.selectRow(i)
                break

    # --- Annotation numbering ---

    def renumber_annotations(self):
        current_name = self.mw.current_slice or self.mw.image_file_name
        if current_name in self.mw.all_annotations:
            for class_name, annotations in self.mw.all_annotations[
                current_name
            ].items():
                for i, ann in enumerate(annotations, start=1):
                    ann["number"] = i
        self.update_annotation_list()

    # --- Delete / merge / change-class ---

    def delete_selected_annotations(self):
        selected_items = self._selected_row_items()
        if not selected_items:
            QMessageBox.warning(
                self.mw, "No Selection", "Please select an annotation to delete."
            )
            return

        # No confirmation / success dialogs — delete is instant and reversible
        # via Ctrl+Z (ADR-026). Snapshot the pre-delete state first.
        self.record_history()

        annotations_to_remove = []
        for item in selected_items:
            annotation = item.data(Qt.ItemDataRole.UserRole)
            annotations_to_remove.append((annotation["category_name"], annotation))

        for category_name, annotation in annotations_to_remove:
            if category_name in self.mw.image_label.annotations:
                if annotation in self.mw.image_label.annotations[category_name]:
                    self.mw.image_label.annotations[category_name].remove(annotation)

        current_name = self.mw.current_slice or self.mw.image_file_name
        self.mw.all_annotations[current_name] = self.mw.image_label.annotations

        if self.mw.current_sort_method == "area":
            self.sort_annotations_by_area()
        else:
            self.sort_annotations_by_class()

        self.mw.image_label.highlighted_annotations.clear()
        # Selection is now empty — Merge/Change Class must follow.
        self._sync_selection_buttons(0)
        self.mw.image_label.update()

        self.mw.update_slice_list_colors()
        self.mw.auto_save()

    def merge_annotations(self):
        if self.mw.image_label.editing_polygon is not None:
            QMessageBox.warning(
                self.mw,
                "Edit Mode Active",
                "Please exit the annotation edit mode before merging annotations.",
            )
            return

        selected_items = self._selected_row_items()
        if len(selected_items) < 2:
            QMessageBox.warning(
                self.mw,
                "Not Enough Annotations",
                "Please select at least two annotations to merge.",
            )
            return

        # Keypoint instances have no mergeable geometry; merging would silently
        # delete them (they're skipped from the polygon union but still removed
        # from the class list). Reject up front. (#35)
        if any(
            "keypoints" in item.data(Qt.ItemDataRole.UserRole)
            for item in selected_items
        ):
            QMessageBox.warning(
                self.mw,
                "Cannot Merge",
                "Keypoint (pose) instances can't be merged.",
            )
            return

        class_name = selected_items[0].data(Qt.ItemDataRole.UserRole)["category_name"]
        if not all(
            item.data(Qt.ItemDataRole.UserRole)["category_name"] == class_name
            for item in selected_items
        ):
            QMessageBox.warning(
                self.mw,
                "Mixed Classes",
                "All selected annotations must be from the same class.",
            )
            return

        polygons = []
        original_annotations = []
        for item in selected_items:
            annotation = item.data(Qt.ItemDataRole.UserRole)
            original_annotations.append(annotation)
            if "segmentation" in annotation:
                points = zip(
                    annotation["segmentation"][0::2], annotation["segmentation"][1::2]
                )
                polygon = Polygon(points)
                if not polygon.is_valid:
                    polygon = polygon.buffer(0)
                polygons.append(polygon)

        def are_all_polygons_connected(polygons):
            if len(polygons) < 2:
                return True

            connected = set([0])
            to_check = set(range(1, len(polygons)))

            while to_check:
                newly_connected = set()
                for i in connected:
                    for j in to_check:
                        if polygons[i].intersects(polygons[j]) or polygons[i].touches(
                            polygons[j]
                        ):
                            newly_connected.add(j)

                if not newly_connected:
                    return False

                connected.update(newly_connected)
                to_check -= newly_connected

            return True

        if not are_all_polygons_connected(polygons):
            QMessageBox.warning(
                self.mw,
                "Disconnected Polygons",
                "Not all selected annotations are connected. Please select only connected annotations to merge.",
            )
            return

        try:
            merged_polygon = unary_union(polygons)
        except Exception as e:
            QMessageBox.warning(
                self.mw,
                "Merge Error",
                f"Unable to merge the selected annotations due to an error: {str(e)}",
            )
            return

        new_annotation = {
            "segmentation": [],
            "category_id": self.mw.class_mapping[class_name],
            "category_name": class_name,
        }

        if isinstance(merged_polygon, Polygon):
            new_annotation["segmentation"] = [
                coord for point in merged_polygon.exterior.coords for coord in point
            ]
        elif isinstance(merged_polygon, MultiPolygon):
            largest_polygon = max(merged_polygon.geoms, key=lambda p: p.area)
            new_annotation["segmentation"] = [
                coord for point in largest_polygon.exterior.coords for coord in point
            ]

        # No keep/delete prompt and no success dialog — merging always
        # replaces the originals with the union, instant and reversible via
        # Ctrl+Z (ADR-026). Snapshot the pre-merge state first.
        self.record_history()

        for annotation in original_annotations:
            if annotation in self.mw.image_label.annotations[class_name]:
                self.mw.image_label.annotations[class_name].remove(annotation)

        self.mw.image_label.annotations.setdefault(class_name, []).append(new_annotation)

        current_name = self.mw.current_slice or self.mw.image_file_name
        self.mw.all_annotations[current_name] = self.mw.image_label.annotations

        self.renumber_annotations()
        self.update_annotation_list()
        self.save_current_annotations()
        self.mw.update_slice_list_colors()
        self.mw.image_label.update()
        self.mw.auto_save()

    def change_annotation_class(self):
        selected_items = self._selected_row_items()
        if not selected_items:
            QMessageBox.warning(
                self.mw,
                "No Selection",
                "Please select one or more annotations to change class.",
            )
            return

        class_dialog = QDialog(self.mw)
        class_dialog.setWindowTitle("Change Class")
        layout = QVBoxLayout(class_dialog)

        class_combo = QComboBox()
        for class_name in self.mw.class_mapping.keys():
            class_combo.addItem(class_name)
        layout.addWidget(class_combo)

        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(class_dialog.accept)
        button_box.rejected.connect(class_dialog.reject)
        layout.addWidget(button_box)

        if class_dialog.exec() == QDialog.DialogCode.Accepted:
            new_class = class_combo.currentText()
            current_name = self.mw.current_slice or self.mw.image_file_name

            # Geometry types must stay compatible (#35): a keypoint instance may
            # only move to a pose class whose schema has an identical `names`
            # list (name/order is what the flat [x,y,v] payload is keyed on;
            # skeleton/flip differences are harmless). A normal annotation may
            # not move into a pose class, nor a keypoint instance out to one.
            new_schema = self.mw.keypoint_schemas.get(new_class)
            for item in selected_items:
                ann = item.data(Qt.ItemDataRole.UserRole)
                if "keypoints" in ann:
                    old_schema = self.mw.keypoint_schemas.get(ann["category_name"])
                    if (
                        new_schema is None
                        or old_schema is None
                        or new_schema.get("names") != old_schema.get("names")
                    ):
                        QMessageBox.warning(
                            self.mw,
                            "Incompatible Class",
                            "A keypoint instance can only be moved to a pose class "
                            "with the same keypoint schema.",
                        )
                        return
                elif new_schema is not None:
                    QMessageBox.warning(
                        self.mw,
                        "Incompatible Class",
                        "Only keypoint instances can be moved into a pose class.",
                    )
                    return

            self.record_history()

            max_number = max(
                [
                    ann.get("number", 0)
                    for ann in self.mw.image_label.annotations.get(new_class, [])
                ]
                + [0]
            )

            for item in selected_items:
                annotation = item.data(Qt.ItemDataRole.UserRole)
                old_class = annotation["category_name"]

                self.mw.image_label.annotations[old_class].remove(annotation)
                if not self.mw.image_label.annotations[old_class]:
                    del self.mw.image_label.annotations[old_class]

                annotation["category_name"] = new_class
                annotation["category_id"] = self.mw.class_mapping[new_class]
                max_number += 1
                annotation["number"] = max_number
                if new_class not in self.mw.image_label.annotations:
                    self.mw.image_label.annotations[new_class] = []
                self.mw.image_label.annotations[new_class].append(annotation)

            self.mw.all_annotations[current_name] = self.mw.image_label.annotations

            self.renumber_annotations()

            self.update_annotation_list()
            self.mw.image_label.update()
            self.save_current_annotations()
            self.mw.update_slice_list_colors()
            self.mw.auto_save()

            QMessageBox.information(
                self.mw,
                "Class Changed",
                f"Selected annotations have been changed to class '{new_class}'.",
            )

    # --- Commit paths for the drawing tools ---

    def finish_polygon(self):
        if (
            self.mw.image_label.current_tool == "polygon"
            and len(self.mw.image_label.current_annotation) > 2
        ):
            if self.mw.current_class is None:
                QMessageBox.warning(
                    self.mw,
                    "No Class Selected",
                    "Please select a class before finishing the annotation.",
                )
                return

            polygon = Polygon(self.mw.image_label.current_annotation)

            image_boundary = Polygon(
                [
                    (0, 0),
                    (self.mw.current_image.width(), 0),
                    (self.mw.current_image.width(), self.mw.current_image.height()),
                    (0, self.mw.current_image.height()),
                ]
            )

            clipped_polygon = polygon.intersection(image_boundary)

            if clipped_polygon.is_empty:
                QMessageBox.warning(
                    self.mw,
                    "Invalid Annotation",
                    "The annotation is completely outside the image boundaries.",
                )
                self.mw.image_label.clear_current_annotation()
                self.mw.image_label.update()
                return

            if isinstance(clipped_polygon, Polygon):
                segmentation = [
                    coord
                    for point in clipped_polygon.exterior.coords
                    for coord in point
                ]
            elif isinstance(clipped_polygon, MultiPolygon):
                largest_polygon = max(clipped_polygon.geoms, key=lambda p: p.area)
                segmentation = [
                    coord
                    for point in largest_polygon.exterior.coords
                    for coord in point
                ]
            else:
                QMessageBox.warning(
                    self.mw,
                    "Invalid Annotation",
                    "The annotation could not be processed.",
                )
                return

            self.record_history()
            new_annotation = {
                "segmentation": segmentation,
                "category_id": self.mw.class_mapping[self.mw.current_class],
                "category_name": self.mw.current_class,
            }
            self.mw.image_label.annotations.setdefault(
                self.mw.current_class, []
            ).append(new_annotation)
            self.add_annotation_to_list(new_annotation)
            self.mw.image_label.clear_current_annotation()
            self.mw.image_label.drawing_polygon = False
            self.mw.image_label.reset_annotation_state()
            self.mw.image_label.update()

            self.save_current_annotations()

            self.mw.update_slice_list_colors()
            self.mw.auto_save()

    def finish_rectangle(self):
        if self.mw.image_label.current_rectangle:
            x1, y1, x2, y2 = self.mw.image_label.current_rectangle

            rectangle = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])

            image_boundary = Polygon(
                [
                    (0, 0),
                    (self.mw.current_image.width(), 0),
                    (self.mw.current_image.width(), self.mw.current_image.height()),
                    (0, self.mw.current_image.height()),
                ]
            )

            clipped_rectangle = rectangle.intersection(image_boundary)

            if clipped_rectangle.is_empty:
                QMessageBox.warning(
                    self.mw,
                    "Invalid Annotation",
                    "The annotation is completely outside the image boundaries.",
                )
                self.mw.image_label.current_rectangle = None
                self.mw.image_label.update()
                return

            if isinstance(clipped_rectangle, Polygon):
                segmentation = [
                    coord
                    for point in clipped_rectangle.exterior.coords
                    for coord in point
                ]
            elif isinstance(clipped_rectangle, MultiPolygon):
                largest_polygon = max(clipped_rectangle.geoms, key=lambda p: p.area)
                segmentation = [
                    coord
                    for point in largest_polygon.exterior.coords
                    for coord in point
                ]
            else:
                QMessageBox.warning(
                    self.mw,
                    "Invalid Annotation",
                    "The annotation could not be processed.",
                )
                return

            self.record_history()
            new_annotation = {
                "segmentation": segmentation,
                "category_id": self.mw.class_mapping[self.mw.current_class],
                "category_name": self.mw.current_class,
            }
            self.mw.image_label.annotations.setdefault(
                self.mw.current_class, []
            ).append(new_annotation)
            self.add_annotation_to_list(new_annotation)
            self.mw.image_label.start_point = None
            self.mw.image_label.end_point = None
            self.mw.image_label.current_rectangle = None
            self.mw.image_label.update()

            self.save_current_annotations()

            self.mw.update_slice_list_colors()
            self.mw.auto_save()

    def finish_keypoint(self):
        """Commit the in-progress pose instance (#35). Pads any not-yet-placed
        points to K with v=0 (not labelled), clamps into the image, derives the
        instance bbox from the labelled points, then stores + lists it. Mirrors
        finish_rectangle (record_history before mutate, then add + persist)."""
        il = self.mw.image_label
        placed = list(il.current_keypoints)
        class_name = self.mw.current_class
        schema = self.mw.keypoint_schemas.get(class_name)
        if not placed or not schema:
            il.reset_annotation_state()
            il.update()
            return

        k = len(schema["names"])
        points = placed[:k] + [(0, 0, 0)] * max(0, k - len(placed))
        flat = [float(c) for p in points for c in p]

        img = self.mw.current_image
        if img is not None:
            flat = clamp_keypoints(flat, img.width(), img.height())
        bbox = self._keypoint_instance_bbox(
            flat,
            img.width() if img is not None else None,
            img.height() if img is not None else None,
        )
        num_keypoints = sum(1 for i in range(2, len(flat), 3) if flat[i] > 0)

        self.record_history()
        new_annotation = {
            "keypoints": flat,
            "num_keypoints": num_keypoints,
            "bbox": bbox,
            "category_id": self.mw.class_mapping[class_name],
            "category_name": class_name,
        }
        il.annotations.setdefault(class_name, []).append(new_annotation)
        self.add_annotation_to_list(new_annotation)
        il.reset_annotation_state()
        il.update()

        self.save_current_annotations()
        self.mw.update_slice_list_colors()
        self.mw.auto_save()

    @staticmethod
    def _keypoint_instance_bbox(flat, width, height, margin=6):
        """Bounding box [x, y, w, h] around a pose instance's labelled (v>0)
        points. Delegates to utils.keypoint_instance_bbox, which COCO import
        also uses (issue #35 PR-2) — single source of truth."""
        return keypoint_instance_bbox(flat, width, height, margin)

    def add_annotation_to_list(self, annotation):
        class_name = annotation["category_name"]
        color = self.mw.image_label.class_colors.get(
            class_name, QColor(Qt.GlobalColor.white)
        )
        annotations = self.mw.image_label.annotations.get(class_name, [])
        number = max([ann.get("number", 0) for ann in annotations] + [0]) + 1
        annotation["number"] = number

        self._insert_annotation_row(annotation, color)

        self.mw.annotation_list.clearSelection()
        self.mw.image_label.highlighted_annotations.clear()
        self.mw.image_label.update()

    # --- Edit mode ---

    def enter_edit_mode(self, annotation):
        self.mw.editing_mode = True
        self.mw.disable_tools()

        QMessageBox.information(
            self.mw,
            "Edit Mode",
            "You are now in edit mode. Click and drag points to move them, Shift+Click to delete points, or click on edges to add new points.",
        )

    def exit_edit_mode(self):
        self.mw.editing_mode = False
        self.mw.enable_tools()

        self.mw.image_label.editing_polygon = None
        self.mw.image_label.editing_point_index = None
        self.mw.image_label.hover_point_index = None
        self.update_annotation_list()
        self.mw.image_label.update()
