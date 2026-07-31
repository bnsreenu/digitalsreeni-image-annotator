"""Dataset similarity report (issues #72, #82).

A thin view over :class:`CurationController`. Three things make it useful
rather than merely interesting: the threshold slider re-analyses without
re-embedding anything (arithmetic, not inference), selecting a cluster selects
those images in the image list, and the backend is switchable so "CLIP or
DINOv2" can be answered on the dataset in front of you instead of in the
abstract.

There is no delete button, and there will not be one. Removing data on a
similarity heuristic is not recoverable.
"""

from PyQt6.QtCore import QSignalBlocker, Qt, QTimer
from PyQt6.QtGui import QCursor
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
)

from ..core import similarity

# The slider is continuous and each move costs a full pass over every pair.
# On a small project that is milliseconds; at the supported ceiling it is a
# few seconds, and firing it per tick of a drag would queue up dozens of them.
_RECLUSTER_DELAY_MS = 200

_COHESION_TIP = (
    "How tight the cluster is: the least similar pair, and the average pair.\n"
    "Close together means a genuinely compact group. A minimum well below the "
    "mean means a chain — the first and last images may not resemble each "
    "other at all, only their neighbours."
)


class DatasetCurationDialog(QDialog):
    def __init__(self, main_window, controller):
        super().__init__(main_window)
        self.mw = main_window
        self.controller = controller
        self.setWindowTitle("Dataset similarity")
        self.resize(860, 560)

        layout = QVBoxLayout(self)
        self.summary_label = QLabel()
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

        self.coverage_label = QLabel()
        self.coverage_label.setWordWrap(True)
        layout.addWidget(self.coverage_label)

        layout.addLayout(self._build_controls())

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels([
            "Cluster", "Images", "Cohesion (min / mean)", "Review score",
            "Suggested",
        ])
        self.tree.setColumnWidth(0, 260)
        self.tree.headerItem().setToolTip(2, _COHESION_TIP)
        # Structural only; colours come from the active stylesheet (CLAUDE.md).
        self.tree.setStyleSheet(
            "QHeaderView::section { font-weight: bold; padding: 2px; }"
        )
        layout.addWidget(self.tree, 1)

        buttons_row = QHBoxLayout()
        self.select_button = QPushButton("Select cluster in image list")
        self.select_button.setToolTip(
            "Select these images so the existing filters and navigation apply "
            "to them. Nothing is deleted or modified."
        )
        self.select_button.clicked.connect(self._select_cluster)
        buttons_row.addWidget(self.select_button)
        buttons_row.addStretch(1)
        layout.addLayout(buttons_row)

        close_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        close_box.rejected.connect(self.reject)
        layout.addWidget(close_box)

        self._refresh()

    def _build_controls(self):
        row = QHBoxLayout()

        row.addWidget(QLabel("Model"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(self.controller.available_models())
        self.model_combo.setCurrentText(self.controller.model_name)
        self.model_combo.setToolTip(
            "Which backend produces the vectors. Switching recomputes them — "
            "cached per model, so switching back is instant.\n\n"
            "DINOv2 is generally stronger on pure visual similarity; CLIP "
            "carries semantic bias, which helps on natural photographs and "
            "hurts on texture-heavy microscopy. Which suits a given dataset "
            "is a question best settled by trying both on it."
        )
        self.model_combo.currentTextChanged.connect(self._change_model)
        row.addWidget(self.model_combo)

        row.addSpacing(16)
        row.addWidget(QLabel("Similarity threshold"))
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(50, 99)
        self.slider.setValue(int(round(self.controller.threshold * 100)))
        self.slider.setToolTip(
            "How alike two images must be to count as near-duplicates. "
            "Re-analysing never re-embeds — the vectors are already computed."
        )
        self.slider.valueChanged.connect(self._threshold_moved)
        row.addWidget(self.slider, 1)
        self.threshold_label = QLabel()
        row.addWidget(self.threshold_label)

        self._recluster_timer = QTimer(self)
        self._recluster_timer.setSingleShot(True)
        self._recluster_timer.setInterval(_RECLUSTER_DELAY_MS)
        self._recluster_timer.timeout.connect(self._refresh)
        return row

    # --- controls ---

    def _threshold_moved(self, value):
        """Show the new threshold at once; re-analyse once the drag settles."""
        threshold = value / 100.0
        self.threshold_label.setText(f"{threshold:.2f}")
        self.controller.threshold = threshold
        self._recluster_timer.start()

    def _change_model(self, model_name):
        """Re-embed with another backend, restoring the old one on failure.

        A failed switch (no network for the download, or the user cancelling
        the progress dialog) must not leave the report empty: the previous
        model's vectors are put back and the combo follows.
        """
        if self.controller.is_computing():
            # Belt to the controller's braces: the progress dialog is
            # non-modal and `compute` spins the event loop, so without this the
            # combo is live during a run.
            with QSignalBlocker(self.model_combo):
                self.model_combo.setCurrentText(self.controller.model_name)
            return

        previous = self.controller.model_name
        kept = self.controller.embeddings
        if not self.controller.set_model(model_name):
            return

        self.model_combo.setEnabled(False)
        self.slider.setEnabled(False)
        try:
            recomputed = self.controller.compute(self)
        except Exception:
            # Restoring only on a False return would leave the controller on a
            # backend with no vectors if `compute` raised. It catches nearly
            # everything itself, so this is the unlikely path -- but "restore
            # the previous model on failure" should mean any failure.
            recomputed = False
            raise
        finally:
            self.model_combo.setEnabled(True)
            self.slider.setEnabled(True)
            if not recomputed:
                self.controller.set_model(previous)
                self.controller.embeddings = kept
                with QSignalBlocker(self.model_combo):
                    self.model_combo.setCurrentText(previous)

        self._refresh()

    # --- report ---

    def _refresh(self):
        self._recluster_timer.stop()
        threshold = self.controller.threshold
        self.threshold_label.setText(f"{threshold:.2f}")

        QApplication.setOverrideCursor(QCursor(Qt.CursorShape.WaitCursor))
        try:
            result = self.controller.analyse(threshold)
        finally:
            QApplication.restoreOverrideCursor()

        clusters = result["clusters"]
        outliers = result["outliers"]
        stats = similarity.summarise(clusters, len(self.controller.embeddings))

        self.summary_label.setText(
            f"{stats['clusters']} near-duplicate cluster(s) covering "
            f"{stats['clustered_images']} of {stats['total_images']} images. "
            f"Keeping one per cluster would skip {stats['redundant_images']}. "
            f"{len(outliers)} image(s) resemble nothing else."
        )
        self.coverage_label.setText(
            self._coverage_text(result["modes"], result["mode_threshold"])
        )
        self._fill_tree(clusters, outliers)

    def _coverage_text(self, modes, mode_threshold):
        """How many distinct kinds of image the dataset holds, and how evenly.

        Redundancy is only half the picture: a dataset can be free of
        near-duplicates and still cover one situation forty times and another
        twice. The threshold is named because the number means nothing without
        it — it is a coarse heuristic, and CLIP and DINOv2 do not put the same
        numbers on the same pair.
        """
        if not modes:
            return ""
        sizes = [len(mode) for mode in modes]
        alone = sum(1 for size in sizes if size == 1)
        text = (
            f"{len(modes)} appearance mode(s) at similarity "
            f"{mode_threshold:.2f}: the largest holds "
            f"{max(sizes)} image(s), the smallest {min(sizes)}."
        )
        if alone:
            text += f" {alone} image(s) form a mode of their own."
        return text

    def _fill_tree(self, clusters, outliers):
        # Outliers too, not only cluster members: an isolated image is exactly
        # the kind the model is least sure about, and gathering scores from
        # clusters alone hid the column outright on a project where only the
        # isolated ones were scored.
        scores = self.controller.review_scores(
            [name for group in clusters for name in group] + list(outliers)
        )
        # An empty column reads as "no uncertainty here" rather than "nothing
        # measured it", so it is hidden outright when no review has run.
        self.tree.setColumnHidden(3, not scores)

        self.tree.clear()
        for index, names in enumerate(clusters, start=1):
            suggested, reason = self.controller.suggested(names, scores)
            parent = QTreeWidgetItem([
                f"Cluster {index}",
                str(len(names)),
                _cohesion_text(self.controller.cohesion(names)),
                "",
                f"{suggested} ({reason})" if suggested else "",
            ])
            parent.setData(0, Qt.ItemDataRole.UserRole, names)
            parent.setToolTip(2, _COHESION_TIP)
            for name in names:
                score = scores.get(name)
                child = QTreeWidgetItem([
                    name,
                    "",
                    "",
                    "" if score is None else f"{score:.1f}",
                    "✓" if name == suggested else "",
                ])
                child.setData(0, Qt.ItemDataRole.UserRole, [name])
                parent.addChild(child)
            self.tree.addTopLevelItem(parent)

        if outliers:
            parent = QTreeWidgetItem([
                "Isolated images", str(len(outliers)), "", "", "nothing similar"
            ])
            parent.setData(0, Qt.ItemDataRole.UserRole, outliers)
            for name in outliers:
                score = scores.get(name)
                child = QTreeWidgetItem([
                    name, "", "", "" if score is None else f"{score:.1f}", ""
                ])
                child.setData(0, Qt.ItemDataRole.UserRole, [name])
                parent.addChild(child)
            self.tree.addTopLevelItem(parent)

    def _select_cluster(self):
        item = self.tree.currentItem()
        if item is None:
            return
        names = item.data(0, Qt.ItemDataRole.UserRole) or []
        self.controller.select_in_image_list(names)


def _cohesion_text(cohesion):
    if not cohesion:
        return ""
    return f"{cohesion['min']:.2f} / {cohesion['mean']:.2f}"
