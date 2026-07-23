"""Config dialog for SAM 2 fine-tuning.

Collects the hyperparameters for ``training.sam_trainer.SAMFineTuner.train``.
Progress is shown via the shared ``TrainingInfoDialog`` (reused from the YOLO
trainer) — this module only owns the *config* dialog.
"""

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QVBoxLayout,
)

from ..inference.sam_utils import MODEL_NAMES


class SAMTrainConfigDialog(QDialog):
    """Modal config for a fine-tuning run. Read results via :meth:`get_config`."""

    def __init__(self, parent=None, *, default_name="my_finetune"):
        super().__init__(parent)
        self.setWindowTitle("Fine-Tune SAM Model")
        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.base_model = QComboBox()
        self.base_model.addItems(MODEL_NAMES)
        # tiny/small are the realistic choices for desktop fine-tuning.
        form.addRow("Base model:", self.base_model)

        self.out_name = QLineEdit(default_name)
        form.addRow("Save as:", self.out_name)

        self.epochs = QSpinBox()
        self.epochs.setRange(1, 1000)
        self.epochs.setValue(10)
        form.addRow("Epochs:", self.epochs)

        self.lr = QDoubleSpinBox()
        self.lr.setDecimals(6)
        self.lr.setRange(1e-6, 1e-1)
        self.lr.setSingleStep(1e-5)
        self.lr.setValue(1e-4)
        self.lr.setToolTip(
            "Peak learning rate. With the schedule on it ramps up over the first "
            "10% of steps then cosine-decays to a 10% floor."
        )
        form.addRow("Peak learning rate:", self.lr)

        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 64)
        self.batch_size.setValue(2)
        self.batch_size.setToolTip(
            "Gradient-accumulation count — the optimizer steps every N images "
            "(all of an image's objects are backpropagated together)."
        )
        form.addRow("Batch size:", self.batch_size)

        # ── train/val split, schedule & early stopping (issue bnsreenu#85) ──
        self.train_pct = QSpinBox()
        self.train_pct.setRange(0, 100)
        self.train_pct.setValue(80)
        self.train_pct.setSuffix("% train")
        self.train_pct.setToolTip(
            "Fraction of annotated images used for training; the rest are a "
            "held-out validation set (deterministic, per image). At 100% there "
            "is no val set, so val_loss and early stopping are off."
        )
        self.train_pct.valueChanged.connect(self._update_ok_enabled)
        form.addRow("Train split:", self.train_pct)

        self.patience = QSpinBox()
        self.patience.setRange(0, 200)
        self.patience.setValue(20)
        self.patience.setToolTip(
            "Stop early when val_loss hasn't improved for this many epochs and "
            "save the best epoch. 0 disables early stopping (best epoch still "
            "saved). Ignored when there is no validation set."
        )
        form.addRow("Early-stop patience:", self.patience)

        self.lr_schedule = QCheckBox("Warmup → cosine LR schedule")
        self.lr_schedule.setChecked(True)
        self.lr_schedule.setToolTip(
            "Linear warmup over the first 10% of steps, then cosine decay to a "
            "10% floor. Uncheck to hold the peak learning rate constant."
        )
        form.addRow("", self.lr_schedule)

        self.prompt_type = QComboBox()
        self.prompt_type.addItems(["bbox", "point"])
        self.prompt_type.setToolTip("Prompt derived from each ground-truth mask during training.")
        form.addRow("Train prompt:", self.prompt_type)

        self.train_encoder = QCheckBox("Also fine-tune image encoder (slower, needs more VRAM/data)")
        form.addRow("", self.train_encoder)

        layout.addLayout(form)

        note = QLabel(
            "Decoder-only (default) is fast and robust on modest data. "
            "Fine-tuning the image encoder can help on heavily domain-shifted "
            "imagery but needs more GPU memory and labels."
        )
        note.setWordWrap(True)
        # No inline color — inside a QDialog `palette(text)` resolves against the
        # stale OS palette and renders near-white in light mode. Let the global
        # QLabel stylesheet rule provide a theme-correct colour (No Hardcoded
        # Colors Rule, docs/08_crosscutting_concepts.md).
        layout.addWidget(note)

        self._split_hint = QLabel("")
        self._split_hint.setWordWrap(True)
        layout.addWidget(self._split_hint)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._ok_button = buttons.button(QDialogButtonBox.StandardButton.Ok)
        self._update_ok_enabled()

    def _update_ok_enabled(self):
        """Block a 0% train split (nothing to train on) and surface the no-val
        behaviour at 100%."""
        pct = self.train_pct.value()
        if pct <= 0:
            self._ok_button.setEnabled(False)
            self._split_hint.setText("Train split must be above 0% — there would be nothing to train on.")
        else:
            self._ok_button.setEnabled(True)
            self._split_hint.setText(
                "100% train: no validation set — val_loss and early stopping are off."
                if pct >= 100 else ""
            )

    def get_config(self) -> dict:
        return {
            "base_model": self.base_model.currentText(),
            "out_name": self.out_name.text().strip() or "my_finetune",
            "epochs": self.epochs.value(),
            "lr": self.lr.value(),
            "batch_size": self.batch_size.value(),
            "prompt_type": self.prompt_type.currentText(),
            "freeze_image_encoder": not self.train_encoder.isChecked(),
            "train_pct": self.train_pct.value(),
            "patience": self.patience.value(),
            "use_lr_schedule": self.lr_schedule.isChecked(),
        }
