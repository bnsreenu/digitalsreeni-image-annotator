"""Dataset similarity report (issue #72).

A thin view over :class:`CurationController`. Two things make it useful rather
than merely interesting: the threshold slider **re-clusters instantly** (the
embeddings are already in memory, so this is arithmetic, not inference), and
selecting a cluster selects those images in the image list — which is what
turns a finding into an actual reduction in work.

There is no delete button, and there will not be one. Removing data on a
similarity heuristic is not recoverable.
"""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
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


class DatasetCurationDialog(QDialog):
    def __init__(self, main_window, controller):
        super().__init__(main_window)
        self.mw = main_window
        self.controller = controller
        self.setWindowTitle("Dataset similarity")
        self.resize(720, 520)

        layout = QVBoxLayout(self)
        self.summary_label = QLabel()
        layout.addWidget(self.summary_label)

        threshold_row = QHBoxLayout()
        threshold_row.addWidget(QLabel("Similarity threshold"))
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(50, 99)
        self.slider.setValue(int(round(controller.threshold * 100)))
        self.slider.setToolTip(
            "How alike two images must be to count as near-duplicates. "
            "Re-clustering is instant — the embeddings are already computed."
        )
        self.slider.valueChanged.connect(self._recluster)
        threshold_row.addWidget(self.slider, 1)
        self.threshold_label = QLabel()
        threshold_row.addWidget(self.threshold_label)
        layout.addLayout(threshold_row)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Cluster", "Images", "Suggested representative"])
        self.tree.setColumnWidth(0, 260)
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

        self._recluster()

    def _recluster(self):
        threshold = self.slider.value() / 100.0
        self.threshold_label.setText(f"{threshold:.2f}")
        self.controller.threshold = threshold

        clusters = self.controller.clusters(threshold)
        outliers = self.controller.outliers(threshold)
        stats = similarity.summarise(clusters, len(self.controller.embeddings))

        self.summary_label.setText(
            f"{stats['clusters']} near-duplicate cluster(s) covering "
            f"{stats['clustered_images']} of {stats['total_images']} images. "
            f"Keeping one per cluster would skip {stats['redundant_images']}. "
            f"{len(outliers)} image(s) resemble nothing else."
        )

        self.tree.clear()
        for index, names in enumerate(clusters, start=1):
            rep = self.controller.representative(names)
            parent = QTreeWidgetItem([
                f"Cluster {index}", str(len(names)), rep or ""
            ])
            parent.setData(0, Qt.ItemDataRole.UserRole, names)
            for name in names:
                child = QTreeWidgetItem([name, "", "✓" if name == rep else ""])
                child.setData(0, Qt.ItemDataRole.UserRole, [name])
                parent.addChild(child)
            self.tree.addTopLevelItem(parent)

        if outliers:
            parent = QTreeWidgetItem([
                "Isolated images", str(len(outliers)), "nothing similar"
            ])
            parent.setData(0, Qt.ItemDataRole.UserRole, outliers)
            for name in outliers:
                child = QTreeWidgetItem([name, "", ""])
                child.setData(0, Qt.ItemDataRole.UserRole, [name])
                parent.addChild(child)
            self.tree.addTopLevelItem(parent)

    def _select_cluster(self):
        item = self.tree.currentItem()
        if item is None:
            return
        names = item.data(0, Qt.ItemDataRole.UserRole) or []
        self.controller.select_in_image_list(names)
