"""Keypoint-schema editor dialog (issue #35).

Defines a pose class's ordered keypoints, their horizontal-flip partners, and
the skeleton edges between them. List-based (not a graphical skeleton editor) —
the lowest-risk first cut, matching the app's other table-driven config dialogs.

Flip partners and skeleton endpoints are chosen by *index* (point number), not
by name, so editing a name never invalidates a reference. ``get_schema()``
returns a sanitized schema dict (see ``core.keypoint_schema``).
"""

from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from ..core.keypoint_schema import is_involution, make_schema, sanitize_schema


class KeypointSchemaDialog(QDialog):
    """Modal editor for one class's keypoint schema.

    Parameters
    ----------
    class_name : str
        The pose class being configured (title only).
    schema : dict | None
        Existing schema to edit, or None for a fresh one.
    lock_k : bool
        When True the keypoint count is frozen (instances already exist):
        add/remove are disabled, names/skeleton/flip stay editable.
    """

    def __init__(self, parent=None, *, class_name="", schema=None, lock_k=False):
        super().__init__(parent)
        self.setWindowTitle(f"Keypoint Schema — {class_name}")
        self.lock_k = lock_k
        self._result_schema = None
        # Suppresses the itemChanged->refresh hook while _set_rows bulk-builds
        # rows (each setItem would otherwise trigger its own full rebuild).
        self._populating = False

        layout = QVBoxLayout(self)

        # ---- Keypoints table: Name | Flip partner -------------------------
        layout.addWidget(QLabel("Keypoints (ordered):"))
        self.points_table = QTableWidget(0, 2)
        self.points_table.setHorizontalHeaderLabels(["Name", "Flip partner"])
        self.points_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        self.points_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents
        )
        layout.addWidget(self.points_table)
        # A typed name must be reflected in every OTHER row's flip-partner combo
        # and the skeleton From/To combos immediately — those combos only get
        # rebuilt on add/remove/move, so without this a freshly-typed name shows
        # as a bare index ("6") in other dropdowns until some unrelated row edit
        # happens to trigger a rebuild.
        self.points_table.itemChanged.connect(self._on_point_name_changed)

        point_btns = QHBoxLayout()
        self.add_point_btn = QPushButton("Add point")
        self.remove_point_btn = QPushButton("Remove point")
        self.up_btn = QPushButton("Move up")
        self.down_btn = QPushButton("Move down")
        for btn in (self.add_point_btn, self.remove_point_btn, self.up_btn, self.down_btn):
            point_btns.addWidget(btn)
        layout.addLayout(point_btns)
        self.add_point_btn.clicked.connect(self._add_point)
        self.remove_point_btn.clicked.connect(self._remove_point)
        self.up_btn.clicked.connect(lambda: self._move_point(-1))
        self.down_btn.clicked.connect(lambda: self._move_point(1))

        # ---- Skeleton table: From | To ------------------------------------
        layout.addWidget(QLabel("Skeleton edges (optional):"))
        self.skeleton_table = QTableWidget(0, 2)
        self.skeleton_table.setHorizontalHeaderLabels(["From", "To"])
        self.skeleton_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        layout.addWidget(self.skeleton_table)

        skel_btns = QHBoxLayout()
        self.add_edge_btn = QPushButton("Add edge")
        self.remove_edge_btn = QPushButton("Remove edge")
        skel_btns.addWidget(self.add_edge_btn)
        skel_btns.addWidget(self.remove_edge_btn)
        layout.addLayout(skel_btns)
        self.add_edge_btn.clicked.connect(self._add_edge)
        self.remove_edge_btn.clicked.connect(self._remove_edge)

        if lock_k:
            self.add_point_btn.setEnabled(False)
            self.remove_point_btn.setEnabled(False)
            layout.addWidget(
                QLabel(
                    "This class already has keypoint instances, so the number of "
                    "keypoints is locked. Names, flip partners and skeleton remain "
                    "editable."
                )
            )

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._populate(schema)

    # ------------------------------------------------------------------ build
    def _populate(self, schema):
        schema = sanitize_schema(schema) if schema else None
        if schema:
            self._set_rows(schema["names"], schema["skeleton"], schema["flip_idx"])
        else:
            # Two blank points so the user has something to fill in.
            self._set_rows(["", ""], [], [0, 1])

    def _set_rows(self, names, skeleton, flip_idx):
        """Rebuild both tables from raw (possibly mid-edit / unsanitized) arrays.
        Used by _populate and _move_point; must NOT sanitize, or empty names
        during editing would be dropped."""
        self._populating = True  # each setItem below would otherwise re-trigger a full rebuild
        try:
            self.points_table.setRowCount(0)
            for name in names:
                self._append_point_row(name)
        finally:
            self._populating = False
        self._refresh_index_combos()
        for i, partner in enumerate(flip_idx):
            combo = self.points_table.cellWidget(i, 1)
            if combo is not None:
                combo.setCurrentIndex(0 if partner == i else partner + 1)
        self.skeleton_table.setRowCount(0)
        for a, b in skeleton:
            self._append_edge_row(a, b)

    def _append_point_row(self, name):
        row = self.points_table.rowCount()
        self.points_table.insertRow(row)
        self.points_table.setItem(row, 0, QTableWidgetItem(name))
        self.points_table.setCellWidget(row, 1, QComboBox())

    def _append_edge_row(self, a=0, b=0):
        row = self.skeleton_table.rowCount()
        self.skeleton_table.insertRow(row)
        from_combo, to_combo = QComboBox(), QComboBox()
        self.skeleton_table.setCellWidget(row, 0, from_combo)
        self.skeleton_table.setCellWidget(row, 1, to_combo)
        self._fill_point_combo(from_combo, selected=a, include_self_label=False)
        self._fill_point_combo(to_combo, selected=b, include_self_label=False)

    # ------------------------------------------------------------- combo sync
    def _point_labels(self):
        labels = []
        for row in range(self.points_table.rowCount()):
            item = self.points_table.item(row, 0)
            text = item.text().strip() if item else ""
            labels.append(f"{row + 1}: {text}" if text else f"{row + 1}")
        return labels

    def _fill_point_combo(self, combo, *, selected, include_self_label):
        """Repopulate a point-index combo, preserving the selected index."""
        prev = selected
        combo.blockSignals(True)
        combo.clear()
        if include_self_label:
            combo.addItem("self")  # index 0 -> "self" (maps to the row's own index)
        combo.addItems(self._point_labels())
        offset = 1 if include_self_label else 0
        target = 0 if (include_self_label and prev is None) else (prev + offset if prev is not None else 0)
        combo.setCurrentIndex(max(0, min(target, combo.count() - 1)))
        combo.blockSignals(False)

    def _refresh_index_combos(self):
        """Rebuild every flip-partner and skeleton combo after the point set
        changes, keeping the current selections where still valid."""
        n = self.points_table.rowCount()
        for row in range(n):
            combo = self.points_table.cellWidget(row, 1)
            if combo is None:
                continue
            # Current partner as a 0-based point index (None => self).
            cur = combo.currentIndex()
            partner = None if cur <= 0 else cur - 1
            if partner is not None and partner >= n:
                partner = None
            self._fill_point_combo(combo, selected=partner, include_self_label=True)
        for row in range(self.skeleton_table.rowCount()):
            for col in (0, 1):
                combo = self.skeleton_table.cellWidget(row, col)
                if combo is None:
                    continue
                cur = combo.currentIndex()
                idx = cur if 0 <= cur < n else 0
                self._fill_point_combo(combo, selected=idx, include_self_label=False)

    def _on_point_name_changed(self, item):
        if item.column() == 0 and not self._populating:
            self._refresh_index_combos()

    # ------------------------------------------------------------- row edits
    def _add_point(self):
        self._append_point_row("")
        self._refresh_index_combos()

    def _remove_point(self):
        row = self.points_table.currentRow()
        if row < 0:
            row = self.points_table.rowCount() - 1
        if self.points_table.rowCount() <= 1:
            return  # keep at least one point
        self.points_table.removeRow(row)
        self._refresh_index_combos()

    def _move_point(self, delta):
        row = self.points_table.currentRow()
        if row < 0:
            return
        n = self.points_table.rowCount()
        target = row + delta
        if not (0 <= target < n):
            return
        # Reorder = a swap permutation applied to EVERY reference, not just the
        # names: a skeleton edge or flip partner that points *at* a moved index
        # must follow it, or the schema silently rewires (senior-review P1).
        names, skeleton, flip_idx = self._collect()
        perm = list(range(n))
        perm[row], perm[target] = perm[target], perm[row]
        new_names = [None] * n
        new_flip = [0] * n
        for i in range(n):
            new_names[perm[i]] = names[i]
            new_flip[perm[i]] = perm[flip_idx[i]]
        new_skeleton = [[perm[a], perm[b]] for a, b in skeleton]
        self._set_rows(new_names, new_skeleton, new_flip)
        self.points_table.setCurrentCell(target, 0)

    def _add_edge(self):
        if self.points_table.rowCount() < 2:
            QMessageBox.warning(self, "Skeleton", "Add at least two keypoints first.")
            return
        self._append_edge_row(0, min(1, self.points_table.rowCount() - 1))

    def _remove_edge(self):
        row = self.skeleton_table.currentRow()
        if row < 0:
            row = self.skeleton_table.rowCount() - 1
        if row >= 0:
            self.skeleton_table.removeRow(row)

    # ------------------------------------------------------------- accept
    def _collect(self):
        names = []
        for row in range(self.points_table.rowCount()):
            item = self.points_table.item(row, 0)
            names.append(item.text().strip() if item else "")
        flip_idx = []
        for row in range(self.points_table.rowCount()):
            combo = self.points_table.cellWidget(row, 1)
            cur = combo.currentIndex()
            flip_idx.append(row if cur <= 0 else cur - 1)
        skeleton = []
        for row in range(self.skeleton_table.rowCount()):
            a = self.skeleton_table.cellWidget(row, 0).currentIndex()
            b = self.skeleton_table.cellWidget(row, 1).currentIndex()
            skeleton.append([a, b])
        return names, skeleton, flip_idx

    def _on_accept(self):
        names, skeleton, flip_idx = self._collect()
        if any(not n for n in names):
            QMessageBox.warning(self, "Keypoint Schema", "Every keypoint needs a name.")
            return
        if len(set(names)) != len(names):
            QMessageBox.warning(self, "Keypoint Schema", "Keypoint names must be unique.")
            return
        # Flip partners must be reciprocal (flipping twice returns the original
        # point) — sanitize_schema would otherwise silently reset ALL flip
        # partners to identity with no warning, discarding the user's input.
        if not is_involution(flip_idx):
            QMessageBox.warning(
                self,
                "Keypoint Schema",
                "Flip partners must be reciprocal: if A's partner is B, "
                "B's partner must be A.",
            )
            return
        try:
            self._result_schema = make_schema(names, skeleton, flip_idx)
        except ValueError:
            QMessageBox.warning(self, "Keypoint Schema", "The schema is invalid.")
            return
        self.accept()

    def get_schema(self):
        """The accepted schema dict, or None if cancelled/invalid."""
        return self._result_schema
