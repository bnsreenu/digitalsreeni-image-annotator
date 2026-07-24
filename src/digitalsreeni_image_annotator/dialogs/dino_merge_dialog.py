"""
Dialog to merge per-image COCO JSONs into train/val splits for fine-tuning.

Ported from annotation_tool_v4.py _MergeCOCODialog.
"""

import json
import math
import random
from collections import defaultdict
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)

from ..core.logging_config import get_logger

logger = get_logger(__name__)


class DinoMergeDialog(QDialog):
    """
    Merge per-image COCO JSONs into train.json and val.json.
    """

    # Phrases used when the category name matches a known structure.
    KNOWN_PHRASES = {
        "glomerulus": [
            "glomerulus", "glomeruli", "renal glomerulus",
            "small circular structure", "round cellular cluster",
            "spherical capillary tuft",
        ],
        "mitochondria": [
            "mitochondria", "mitochondrion",
            "elongated oval organelle", "rod-shaped structure",
        ],
        "nucleus": [
            "nucleus", "nuclei", "cell nucleus",
            "round dark structure",
        ],
        "cell": [
            "cell", "cells", "individual cell",
        ],
    }

    def __init__(self, parent=None, extra_phrases=None):
        super().__init__(parent)
        self.setWindowTitle("Merge COCO for Fine-Tuning")
        self.setMinimumWidth(540)
        self.setMinimumHeight(520)
        if extra_phrases:
            self.KNOWN_PHRASES.update(extra_phrases)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        info = QLabel(
            "Merges per-image COCO JSONs into a single dataset and splits "
            "it into train.json and val.json for fine-tuning.\n\n"
            "Input folder: directory containing *_coco.json files.\n"
            "Images folder: directory with original images.\n"
            "Output folder: where train.json / val.json will be written."
        )
        info.setWordWrap(True)
        info.setStyleSheet("font-size:11px;color:#444;padding:4px;")
        layout.addWidget(info)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        def browse_row(placeholder, pick_dir=True):
            row = QHBoxLayout()
            edit = QLineEdit()
            edit.setPlaceholderText(placeholder)
            btn = QPushButton("Browse")
            btn.setFixedWidth(70)
            if pick_dir:
                btn.clicked.connect(
                    lambda: edit.setText(
                        QFileDialog.getExistingDirectory(self, placeholder)
                        or edit.text()
                    )
                )
            row.addWidget(edit, 1)
            row.addWidget(btn)
            return row, edit

        masks_row,  self._masks_edit  = browse_row("Folder with *_coco.json files")
        images_row, self._images_edit = browse_row("Folder with original images")
        output_row, self._output_edit = browse_row("Output folder")

        form.addRow("COCO folder:", masks_row)
        form.addRow("Images folder:", images_row)
        form.addRow("Output folder:", output_row)

        self._val_spin = QLineEdit("0.20")
        self._val_spin.setPlaceholderText("0.0 - 0.5")
        form.addRow("Val fraction:", self._val_spin)

        self._stratify_combo = QComboBox()
        self._stratify_combo.addItem("Auto (IHC vs HE from filename)")
        self._stratify_combo.addItem("None (random split)")
        form.addRow("Stratification:", self._stratify_combo)

        layout.addLayout(form)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setFixedHeight(160)
        self._log.setStyleSheet(
            "QTextEdit{background:#1e1e1e;color:#d4d4d4;"
            "font-family:Consolas,monospace;font-size:11px;}")
        layout.addWidget(self._log)

        btn_row = QHBoxLayout()
        self._btn_run = QPushButton("Merge and Split")
        self._btn_run.setStyleSheet(
            "QPushButton{background:#2E75B6;color:white;font-weight:bold;"
            "padding:7px;border-radius:4px;}"
            "QPushButton:hover{background:#1a5490;}")
        self._btn_run.clicked.connect(self._run)
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        btn_row.addWidget(self._btn_run)
        btn_row.addWidget(btn_close)
        layout.addLayout(btn_row)

    def _log_msg(self, msg):
        self._log.append(msg)
        self._log.verticalScrollBar().setValue(
            self._log.verticalScrollBar().maximum())
        QApplication.processEvents()

    def _run(self):
        masks_dir = self._masks_edit.text().strip()
        images_dir = self._images_edit.text().strip()
        output_dir = self._output_edit.text().strip()

        errors = []
        if not masks_dir or not Path(masks_dir).exists():
            errors.append("COCO folder not found.")
        if not output_dir:
            errors.append("Output folder not set.")
        if errors:
            QMessageBox.warning(self, "Missing inputs", "\n".join(errors))
            return

        try:
            val_frac = float(self._val_spin.text().strip())
            val_frac = max(0.05, min(val_frac, 0.50))
        except ValueError:
            val_frac = 0.20

        self._btn_run.setEnabled(False)
        self._log.clear()
        try:
            masks_path = Path(masks_dir)
            images_path = Path(images_dir) if images_dir else None
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)

            coco_files = sorted(masks_path.glob("*_coco.json"))
            if not coco_files:
                self._log_msg("No *_coco.json files found.")
                return
            self._log_msg(f"Found {len(coco_files)} COCO file(s).")

            # Load and validate
            records = []
            for path in coco_files:
                with open(path, encoding='utf-8') as f:
                    data = json.load(f)
                if not data.get("images") or not data.get("annotations"):
                    self._log_msg(f"  [skip] {path.name}: empty.")
                    continue
                records.append({"source": path.name, "data": data})
            self._log_msg(f"Valid records: {len(records)}")

            # Category map
            name_to_gid = {}
            gid = 1
            for rec in records:
                for cat in rec["data"].get("categories", []):
                    if cat["name"] not in name_to_gid:
                        name_to_gid[cat["name"]] = gid
                        gid += 1

            global_cats = []
            for name, g in sorted(name_to_gid.items(), key=lambda x: x[1]):
                phrases = self.KNOWN_PHRASES.get(name, [name])
                global_cats.append({
                    "id": g, "name": name,
                    "supercategory": "object", "phrases": phrases,
                })

            local_to_global = {}
            for rec in records:
                src = rec["source"]
                for cat in rec["data"].get("categories", []):
                    local_to_global[(src, cat["id"])] = name_to_gid[cat["name"]]

            self._log_msg(f"Categories: {[c['name'] for c in global_cats]}")

            # Flatten records with global IDs
            flat = []
            img_ctr = ann_ctr = 1
            for rec in records:
                src = rec["source"]
                data = rec["data"]
                img_info = data["images"][0]

                file_name = img_info["file_name"]
                p = Path(file_name)
                if p.parent == Path(".") and images_path:
                    file_name = str(images_path / p.name)

                local_anns = [
                    a for a in data["annotations"]
                    if a["image_id"] == img_info["id"]
                ]

                new_anns = []
                for ann in local_anns:
                    gcat = local_to_global.get((src, ann["category_id"]))
                    if gcat is None:
                        continue
                    bbox = ann["bbox"]
                    new_anns.append({
                        "id": ann_ctr, "image_id": img_ctr,
                        "category_id": gcat,
                        "bbox": bbox,
                        "area": ann.get("area", bbox[2] * bbox[3]),
                        "iscrowd": 0,
                    })
                    ann_ctr += 1

                flat.append({
                    "image_id": img_ctr,
                    "file_name": file_name,
                    "height": img_info["height"],
                    "width": img_info["width"],
                    "annotations": new_anns,
                })
                img_ctr += 1

            # Split
            stratify = self._stratify_combo.currentIndex() == 0
            rng = random.Random(42)

            if stratify:
                def stain(fn):
                    return "IHC" if "IHC" in Path(fn).stem.upper() else "HE"

                by_stain = defaultdict(list)
                for r in flat:
                    by_stain[stain(r["file_name"])].append(r)

                train_imgs, val_imgs = [], []
                for group in by_stain.values():
                    rng.shuffle(group)
                    n_val = max(1, math.floor(len(group) * val_frac))
                    val_imgs.extend(group[:n_val])
                    train_imgs.extend(group[n_val:])
            else:
                rng.shuffle(flat)
                n_val = max(1, math.floor(len(flat) * val_frac))
                val_imgs = flat[:n_val]
                train_imgs = flat[n_val:]

            def _build_coco(imgs):
                return {
                    "images": [
                        {"id": r["image_id"], "file_name": r["file_name"],
                         "height": r["height"], "width": r["width"]}
                        for r in imgs
                    ],
                    "annotations": [
                        a for r in imgs for a in r["annotations"]
                    ],
                    "categories": global_cats,
                }

            train_data = _build_coco(train_imgs)
            val_data = _build_coco(val_imgs)

            with open(out_path / "train.json", "w", encoding='utf-8') as f:
                json.dump(train_data, f, indent=2)
            with open(out_path / "val.json", "w", encoding='utf-8') as f:
                json.dump(val_data, f, indent=2)

            self._log_msg(f"Train images: {len(train_imgs)}, annotations: {len(train_data['annotations'])}")
            self._log_msg(f"Val images:   {len(val_imgs)}, annotations: {len(val_data['annotations'])}")
            self._log_msg(f"Saved to {output_dir}")
            QMessageBox.information(
                self, "Done",
                f"train.json and val.json saved to\n{output_dir}"
            )

        except Exception as e:
            self._log_msg(f"ERROR: {e}")
            logger.exception("Failed to merge and split COCO datasets")
        finally:
            self._btn_run.setEnabled(True)


def show_dino_merge_dialog(parent=None):
    dialog = DinoMergeDialog(parent)
    dialog.exec()
