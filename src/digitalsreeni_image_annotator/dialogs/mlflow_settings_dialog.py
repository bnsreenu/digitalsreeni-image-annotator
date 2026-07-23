"""Settings dialog for MLflow experiment tracking (issue #74).

Edits the persisted tracking *destination* (tracking-store URI/path, experiment
name). Tracking itself is always on — there is no enable/disable here. Storage
defaults to ``<project>/mlruns`` when the URI is left blank — see
``training/mlflow_tracker.resolve_tracking_uri``.

No inline colours (No Hardcoded Colors Rule, docs/08_crosscutting_concepts.md);
the global stylesheet themes the widgets.
"""

from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)

from ..app_settings import (
    MLFLOW_EXPERIMENT_DEFAULT,
    load_mlflow_prefs,
    save_mlflow_prefs,
)


class MLflowSettingsDialog(QDialog):
    """Modal editor for the MLflow tracking destination."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Experiment Tracking (MLflow)")
        layout = QVBoxLayout(self)
        form = QFormLayout()

        uri, experiment = load_mlflow_prefs()

        self.uri = QLineEdit(uri)
        self.uri.setPlaceholderText("(blank = <project>/mlruns)")
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        uri_row = QHBoxLayout()
        uri_row.addWidget(self.uri)
        uri_row.addWidget(browse)
        form.addRow("Tracking store:", uri_row)

        self.experiment = QLineEdit(experiment or MLFLOW_EXPERIMENT_DEFAULT)
        form.addRow("Experiment name:", self.experiment)

        layout.addLayout(form)

        note = QLabel(
            "Every training run is tracked with MLflow. Leave the tracking "
            "store blank to default to a local 'mlruns' folder next to the "
            "current project. View runs with Settings → Experiment Tracking → "
            "Open MLflow UI, or run 'mlflow ui' yourself."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _browse(self):
        path = QFileDialog.getExistingDirectory(
            self, "Select MLflow tracking directory", self.uri.text() or ""
        )
        if path:
            self.uri.setText(path)

    def accept(self):
        save_mlflow_prefs(
            self.uri.text().strip(),
            self.experiment.text().strip() or MLFLOW_EXPERIMENT_DEFAULT,
        )
        super().accept()
