"""Annotation QC audit dialog (issue #70).

A thin view over :mod:`core.annotation_qc`. All the judgement lives in the rule
engine, which is Qt-free so the headless CLI can reuse it (issue #76); this
file only groups, renders and dispatches — core raises, the UI catches
(ADR-031).

The two affordances that make an audit useful rather than merely informative:
**jump to** the offending annotation, and **fix** where the repair is
unambiguous.

A batch fix is one undo entry **per image**, not one per annotation and not one
for the batch. ``AnnotationHistory`` is keyed by image (ADR-026), so a single
snapshot would cover only the image on screen and leave every other repair
permanent — which is what an earlier version of this dialog claimed was a
single Ctrl+Z.
"""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
)

from ..core import annotation_qc
from ..core.logging_config import get_logger

logger = get_logger(__name__)

_RULE_TITLES = {
    annotation_qc.RULE_SELF_INTERSECTING: "Self-intersecting outlines",
    annotation_qc.RULE_TOO_FEW_VERTICES: "Fewer than three vertices",
    annotation_qc.RULE_DEGENERATE_AREA: "Degenerate area",
    annotation_qc.RULE_OUT_OF_BOUNDS: "Coordinates outside the image",
    annotation_qc.RULE_BBOX_MISMATCH: "Bounding box disagrees with the outline",
    annotation_qc.RULE_NEAR_DUPLICATE: "Near-duplicate annotations",
    annotation_qc.RULE_CROSS_CLASS_OVERLAP: "Heavy overlap across classes",
    annotation_qc.RULE_AREA_OUTLIER: "Area outliers",
    annotation_qc.RULE_CLASS_IMBALANCE: "Class imbalance",
    annotation_qc.RULE_EMPTY_IMAGE: "Images with no annotations",
    annotation_qc.RULE_SIMILAR_CLASS_NAMES: "Suspiciously similar class names",
    annotation_qc.RULE_ORPHAN_TEMP_CLASS: "Leftover review classes",
    annotation_qc.RULE_POSE_POINT_OUTSIDE_BBOX: "Keypoints outside their box",
    annotation_qc.RULE_POSE_COUNT_MISMATCH: "num_keypoints disagrees with the flags",
}


class AnnotationQCDialog(QDialog):
    """Findings grouped by rule, with jump-to and repair."""

    def __init__(self, main_window, findings):
        super().__init__(main_window)
        self.mw = main_window
        self.findings = findings
        self.setWindowTitle("Check Annotations")
        self.resize(760, 520)

        layout = QVBoxLayout(self)
        summary = annotation_qc.summarise(findings)
        layout.addWidget(
            QLabel(
                f"{summary['total']} finding(s): "
                f"{summary[annotation_qc.SEVERITY_ERROR]} error, "
                f"{summary[annotation_qc.SEVERITY_WARNING]} warning, "
                f"{summary[annotation_qc.SEVERITY_INFO]} info."
            )
        )

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Finding", "Image", "Class", "#"])
        self.tree.setColumnWidth(0, 380)
        # Structural only — colours come from the active stylesheet, or this
        # punches a bright box into the dark theme (CLAUDE.md).
        self.tree.setStyleSheet(
            "QHeaderView::section { font-weight: bold; padding: 2px; }"
        )
        self.tree.itemDoubleClicked.connect(self._jump_to_selected)
        layout.addWidget(self.tree, 1)
        self._populate()

        buttons_row = QHBoxLayout()
        self.jump_button = QPushButton("Go to")
        self.jump_button.setToolTip("Show the offending annotation on the canvas")
        self.jump_button.clicked.connect(self._jump_to_selected)
        buttons_row.addWidget(self.jump_button)

        self.fix_button = QPushButton("Fix all repairable")
        fixable = sum(1 for f in findings if f.fixable)
        self.fix_button.setEnabled(fixable > 0)
        self.fix_button.setToolTip(
            f"Repair the {fixable} unambiguous finding(s). Undo is per image "
            "(ADR-026): to revert a repair, open that image and press Ctrl+Z "
            "there. Ambiguous findings are never auto-fixed."
        )
        self.fix_button.clicked.connect(self._fix_all)
        buttons_row.addWidget(self.fix_button)
        buttons_row.addStretch(1)
        layout.addLayout(buttons_row)

        close_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        close_box.rejected.connect(self.reject)
        layout.addWidget(close_box)

    def _populate(self):
        self.tree.clear()
        grouped = {}
        for finding in self.findings:
            grouped.setdefault(finding.rule, []).append(finding)

        for rule, findings in grouped.items():
            title = _RULE_TITLES.get(rule, rule)
            parent = QTreeWidgetItem([f"{title}  ({len(findings)})", "", "", ""])
            parent.setFirstColumnSpanned(True)
            for finding in findings:
                child = QTreeWidgetItem([
                    f"[{finding.severity}] {finding.message}",
                    finding.image or "",
                    finding.class_name or "",
                    "" if finding.annotation_number is None
                    else str(finding.annotation_number),
                ])
                child.setData(0, Qt.ItemDataRole.UserRole, finding)
                parent.addChild(child)
            self.tree.addTopLevelItem(parent)
            parent.setExpanded(True)

    def _selected_finding(self):
        item = self.tree.currentItem()
        if item is None:
            return None
        return item.data(0, Qt.ItemDataRole.UserRole)

    def _jump_to_selected(self):
        finding = self._selected_finding()
        if finding is None or not finding.image:
            return
        # Reuse the DINO batch-review navigator: it already handles the mixed
        # regular-image / slice-name namespace that all_annotations keys live
        # in, and a second implementation would drift from it.
        if not self.mw.dino_controller._navigate_to_image_or_slice(finding.image):
            QMessageBox.warning(
                self,
                "Cannot navigate",
                f"'{finding.image}' is no longer in the project.",
            )
            return
        self._select_annotation(finding)

    def _select_annotation(self, finding):
        if finding.annotation_number is None or not finding.class_name:
            return
        for annotation in self.mw.image_label.annotations.get(finding.class_name, []):
            if annotation.get("number") == finding.annotation_number:
                self.mw.annotation_controller.apply_canvas_selection(
                    [annotation], "replace"
                )
                return

    def _fix_all(self):
        repaired, images = self.mw.qc_controller.fix_findings(
            [f for f in self.findings if f.fixable]
        )
        # Undo is keyed by image AND `undo()` acts on whichever image is
        # currently open (ADR-026). "Press Ctrl+Z once per image" is therefore
        # not an instruction a user can follow from here -- they have to open
        # each affected image first. Name them, so the instruction is
        # actionable rather than merely accurate.
        if len(images) <= 1:
            undo_note = "Ctrl+Z undoes them."
        else:
            listed = ", ".join(images[:5])
            if len(images) > 5:
                listed += f", and {len(images) - 5} more"
            undo_note = (
                f"They span {len(images)} images ({listed}). Undo is per image: "
                "open each one and press Ctrl+Z there."
            )
        QMessageBox.information(
            self, "Repairs applied", f"{repaired} finding(s) repaired. {undo_note}"
        )
        self.accept()
