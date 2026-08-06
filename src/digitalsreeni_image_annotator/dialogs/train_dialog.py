"""One dialog for all training (issue #73, ADR-042).

Before this, training a YOLO model *and actually using it* took six menu
navigations and about ten dialogs, including a step in the middle where you
reloaded the model you had just produced. SAM fine-tuning was a separate menu
with four more actions, one of which was a manual "Refresh Model Selector".
Eleven menu actions for what is conceptually one operation.

None of that was a decision the user wanted to make. Dataset preparation, YAML
handling and saving are mechanics.

This dialog collects the genuine choices — what to train, from which base, on
how much data, for how long — and the controller does the mechanics implicitly.
The trainers themselves (``YOLOTrainer``, ``SAMFineTuner``) are untouched: this
is a UI and orchestration change, not a training-logic change.

The task is **derived, not asked**: it is entailed by what was annotated, and
the same :mod:`core.task_inference` rules feed the pre-flight checks, so what
the dialog says it will train cannot drift from what the trainer decides to.
"""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..core import task_inference
from ..core.slice_cache import slice_names

TYPE_YOLO = "yolo"
TYPE_SAM = "sam"

# Base checkpoints offered per task. Ultralytics downloads these on first use.
# The suffix has to match the task or training fails deep inside Ultralytics,
# which is exactly the opaque failure the pre-flight check exists to avoid.
YOLO_BASE_MODELS = {
    task_inference.TASK_DETECT: ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt"],
    task_inference.TASK_SEGMENT: [
        "yolo11n-seg.pt", "yolo11s-seg.pt", "yolo11m-seg.pt",
    ],
    task_inference.TASK_POSE: [
        "yolo11n-pose.pt", "yolo11s-pose.pt", "yolo11m-pose.pt",
    ],
}


class TrainDialog(QDialog):
    """The single Train Model dialog.

    Reads the project through ``main_window`` and produces a config dict; it
    starts nothing itself, so the controller keeps ownership of the run and the
    dialog stays testable.
    """

    def __init__(self, main_window):
        super().__init__(main_window)
        self.mw = main_window
        self.setWindowTitle("Train Model")
        self.custom_base_path = None

        # A stack or video contributes its slices, not itself -- count those,
        # or a video with 27 annotated frames reads as "0 of 1 image(s)".
        # Name-only, so building this decodes nothing (issue #45).
        slice_names_by_base = {
            base: slice_names(collection)
            for base, collection in getattr(main_window, "image_slices", {}).items()
        }
        image_names = task_inference.trainable_image_names(
            getattr(main_window, "all_images", []), slice_names_by_base
        )
        self.summary = task_inference.summarise_dataset(
            main_window.all_annotations, image_names
        )
        self.task, self.task_reason = task_inference.infer_task(
            main_window.all_annotations
        )

        layout = QVBoxLayout(self)
        # Kept on self so _on_type_changed can use setRowVisible, which hides
        # a row's label along with its field.
        self.form = form = QFormLayout()

        # --- type ---
        type_row = QHBoxLayout()
        self.yolo_radio = QRadioButton("YOLO")
        self.yolo_radio.setChecked(True)
        self.sam_radio = QRadioButton("SAM 2 fine-tune")
        self.yolo_radio.toggled.connect(self._on_type_changed)
        type_row.addWidget(self.yolo_radio)
        type_row.addWidget(self.sam_radio)
        type_row.addStretch(1)
        form.addRow("Type", type_row)

        # --- base model ---
        base_row = QHBoxLayout()
        self.base_combo = QComboBox()
        # Picking a stock checkpoint has to clear a previously browsed path,
        # or the browse wins forever and the combo selection is ignored.
        self.base_combo.currentIndexChanged.connect(self._on_base_changed)
        base_row.addWidget(self.base_combo, 1)
        browse = QPushButton("Browse…")
        browse.setToolTip("Start from a checkpoint on disk instead")
        browse.clicked.connect(self._browse_base)
        base_row.addWidget(browse)
        self.base_row_widget = _wrap(base_row)
        form.addRow("Base", self.base_row_widget)

        # --- derived task ---
        self.task_label = QLabel()
        self.task_label.setWordWrap(True)
        form.addRow("Task", self.task_label)

        # --- live data summary ---
        self.data_label = QLabel()
        self.data_label.setWordWrap(True)
        form.addRow("Data", self.data_label)

        # --- val split ---
        split_row = QHBoxLayout()
        self.split_slider = QSlider(Qt.Orientation.Horizontal)
        self.split_slider.setRange(0, 50)
        self.split_slider.setValue(20)
        self.split_label = QLabel("20 %")
        self.split_slider.valueChanged.connect(
            lambda v: self.split_label.setText(f"{v} %")
        )
        split_row.addWidget(self.split_slider, 1)
        split_row.addWidget(self.split_label)
        self.split_row_widget = _wrap(split_row)
        form.addRow("Val split", self.split_row_widget)

        # --- epochs + image size ---
        run_row = QHBoxLayout()
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 10000)
        self.epochs_spin.setValue(100)
        run_row.addWidget(QLabel("Epochs"))
        run_row.addWidget(self.epochs_spin)
        run_row.addSpacing(12)
        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(64, 4096)
        self.imgsz_spin.setSingleStep(32)
        self.imgsz_spin.setValue(640)
        run_row.addWidget(QLabel("Image size"))
        run_row.addWidget(self.imgsz_spin)
        run_row.addStretch(1)
        self.run_row_widget = _wrap(run_row)
        form.addRow("Run", self.run_row_widget)

        layout.addLayout(form)

        # --- advanced, collapsed ---
        # ADR-028 deliberately kept these off the main surface; they stay
        # available rather than removed, which is what "collapsed" buys.
        #
        # A disclosure ARROW, not a checkbox. This was a checkable QGroupBox,
        # and Qt disables a checkable group's children when it is unchecked --
        # so the settings looked switched off while `get_config` went on
        # sending them regardless. Early stopping appeared disabled and ran
        # anyway, which is exactly the impression a checkbox creates and this
        # panel must not: the values are the defaults, always in force, and
        # expanding only lets you see and change them.
        self.advanced_toggle = QToolButton()
        self.advanced_toggle.setText("Advanced")
        self.advanced_toggle.setCheckable(True)
        self.advanced_toggle.setChecked(False)
        self.advanced_toggle.setAutoRaise(True)
        self.advanced_toggle.setArrowType(Qt.ArrowType.RightArrow)
        self.advanced_toggle.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self.advanced_toggle.setToolTip(
            "Show the learning-rate and early-stopping settings. They are in "
            "effect either way — expanding only lets you change them."
        )
        self.advanced_toggle.toggled.connect(self._on_advanced_toggled)
        layout.addWidget(self.advanced_toggle)

        self.advanced_box = QWidget()
        self.advanced_box.setVisible(False)
        advanced_form = QFormLayout(self.advanced_box)
        advanced_form.setContentsMargins(16, 0, 0, 0)

        self.cos_lr_check = QCheckBox("Warmup → cosine LR schedule")
        self.cos_lr_check.setChecked(True)
        self.cos_lr_check.setToolTip(
            "Warmup then cosine decay to a 10% floor. Uncheck to hold the peak "
            "learning rate constant."
        )
        advanced_form.addRow(self.cos_lr_check)

        self.lr0_spin = QDoubleSpinBox()
        self.lr0_spin.setDecimals(5)
        self.lr0_spin.setRange(1e-5, 1.0)
        self.lr0_spin.setSingleStep(1e-3)
        self.lr0_spin.setValue(0.01)
        advanced_form.addRow("Peak learning rate (lr0)", self.lr0_spin)

        self.patience_spin = QSpinBox()
        self.patience_spin.setRange(0, 1000)
        self.patience_spin.setValue(20)
        self.patience_spin.setToolTip(
            "Stop when validation hasn't improved for this many epochs; best.pt "
            "is still the best epoch. 0 disables early stopping."
        )
        advanced_form.addRow("Early-stop patience", self.patience_spin)
        layout.addWidget(self.advanced_box)

        # --- blockers ---
        self.blocker_label = QLabel()
        self.blocker_label.setWordWrap(True)
        layout.addWidget(self.blocker_label)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel
        )
        self.train_button = self.buttons.addButton(
            "Train", QDialogButtonBox.ButtonRole.AcceptRole
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        self._on_type_changed()

    # --- state ---

    def training_type(self):
        return TYPE_YOLO if self.yolo_radio.isChecked() else TYPE_SAM

    def blockers(self):
        """Reasons this configuration cannot run, checked **before** the run.

        Both of these used to surface as an opaque failure much later: the
        pose constraint deep inside Ultralytics, the unresolvable-slice one
        during dataset preparation.
        """
        if self.training_type() == TYPE_SAM:
            return []  # SAM fine-tuning has neither constraint
        reasons = list(
            task_inference.pose_training_blockers(
                self.mw.all_annotations, getattr(self.mw, "keypoint_schemas", {})
            )
        )
        reasons += task_inference.unresolvable_stack_blockers(
            getattr(self.mw, "all_images", []),
            getattr(self.mw, "image_slices", {}).keys(),
            [name for name, ann in self.mw.all_annotations.items() if ann],
        )
        if self.task is None:
            reasons.append("There are no annotations to train on.")
        return reasons

    def get_config(self):
        """The user's choices, as a plain dict for the controller to act on.

        In SAM mode only ``type`` is meaningful; the rest is chosen in
        ``SAMTrainConfigDialog``, and the controls that would collect it are
        hidden here precisely so nothing is silently discarded.
        """
        return {
            "type": self.training_type(),
            "task": self.task,
            "base_model": self.custom_base_path or self.base_combo.currentText(),
            "val_split": self.split_slider.value(),
            "epochs": self.epochs_spin.value(),
            "imgsz": self.imgsz_spin.value(),
            "cos_lr": self.cos_lr_check.isChecked(),
            "lr0": self.lr0_spin.value(),
            "patience": self.patience_spin.value(),
        }

    # --- reactions ---

    def _on_type_changed(self):
        is_yolo = self.training_type() == TYPE_YOLO

        # Irrelevant fields hide rather than grey out: a permanently-disabled
        # control invites the user to wonder what would enable it.
        #
        # For SAM this is not cosmetic. SAMTrainController.train_on_project
        # opens its own SAMTrainConfigDialog, which is where the base
        # checkpoint, epochs and schedule are actually chosen. Showing those
        # controls here would collect values this dialog then silently
        # discards -- a user could set 300 epochs, wait hours for a GPU run,
        # and get something else entirely.
        # setRowVisible, not setVisible on the field: hiding a QFormLayout
        # field leaves its LABEL behind, so SAM mode showed three orphan
        # captions -- "Base", "Val split", "Run" -- next to blank space.
        for widget in (
            self.base_row_widget, self.split_row_widget, self.run_row_widget
        ):
            self.form.setRowVisible(widget, is_yolo)
        self.advanced_toggle.setVisible(is_yolo)
        self.advanced_box.setVisible(is_yolo and self.advanced_toggle.isChecked())

        self.base_combo.clear()
        self.custom_base_path = None
        if is_yolo:
            self.base_combo.addItems(
                YOLO_BASE_MODELS.get(
                    self.task, YOLO_BASE_MODELS[task_inference.TASK_DETECT]
                )
            )
            self.task_label.setText(
                f"{self.task or 'unknown'} — inferred from {self.task_reason}"
            )
        else:
            self.task_label.setText(
                "SAM 2 fine-tuning — mask decoder, task not applicable. "
                "Requires a CUDA GPU to be practical.\n"
                "Base checkpoint, epochs and schedule are chosen in the next "
                "dialog."
            )

        self._refresh_summary()
        self._refresh_blockers()

    def _on_advanced_toggled(self, expanded):
        self.advanced_toggle.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        )
        # The is_yolo half matters: these are YOLO-only controls, and SAM mode
        # hides them precisely so it cannot collect values it would discard.
        # Without the guard, expanding while in SAM mode surfaces them again.
        self.advanced_box.setVisible(expanded and self.training_type() == TYPE_YOLO)
        self.adjustSize()

    def _refresh_summary(self):
        summary = self.summary
        text = (
            f"{summary['annotations']} annotation(s) across "
            f"{summary['annotated_images']} of {summary['images']} image(s), "
            f"{len(summary['classes'])} class(es)."
        )
        if summary["unlabelled_images"]:
            # A project where most images have no labels trains badly, and the
            # number is invisible until someone counts it.
            share = (
                summary["unlabelled_images"] / summary["images"]
                if summary["images"]
                else 0
            )
            text += f"\n{summary['unlabelled_images']} image(s) have no labels"
            text += " — a large share of the dataset." if share > 0.3 else "."
        self.data_label.setText(text)

    def _refresh_blockers(self):
        reasons = self.blockers()
        if reasons:
            self.blocker_label.setText(
                "Cannot train:\n• " + "\n• ".join(reasons)
            )
            self.blocker_label.setVisible(True)
            self.train_button.setEnabled(False)
        else:
            self.blocker_label.setVisible(False)
            self.train_button.setEnabled(True)

    def _on_base_changed(self):
        """Drop a browsed path once the user picks something else."""
        if self.base_combo.currentText() != self.custom_base_path:
            self.custom_base_path = None

    def _browse_base(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose a base checkpoint", "", "Model weights (*.pt)"
        )
        if not path:
            return
        self.base_combo.insertItem(0, path)
        # Set custom_base_path AFTER the index change, so _on_base_changed's
        # comparison sees the new value rather than clearing what we just set.
        self.base_combo.setCurrentIndex(0)
        self.custom_base_path = path


def _wrap(layout):
    widget = QWidget()
    widget.setLayout(layout)
    return widget
