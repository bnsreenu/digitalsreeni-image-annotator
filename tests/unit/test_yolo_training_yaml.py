"""Unit tests for the in-app YOLO trainer's dataset-yaml path resolution (#83).

`_resolve_training_yaml` is the piece that turns a prepared/loaded data.yaml
into the absolute train/val pointers written to `temp_train.yaml` just before
`YOLO.train(...)`. It's the one spot where a wrong `val` pointer crashes
`yolo train` on an empty validation set, so it gets its own coverage even though
the surrounding `train_model` needs a real model to run end-to-end.
"""

import os
from pathlib import Path

from src.digitalsreeni_image_annotator.dialogs.yolo_trainer import (
    _resolve_training_yaml,
)


def test_honors_held_out_val_split(tmp_path):
    # Shape that export_yolo_v5plus writes when a val set WAS routed.
    yc = {
        "path": str(tmp_path),
        "train": os.path.join("images", "train"),
        "val": os.path.join("images", "val"),
        "nc": 1,
        "names": ["cell"],
    }
    out = _resolve_training_yaml(str(tmp_path), yc)

    assert out["train"] == str((tmp_path / "images" / "train").resolve())
    assert out["val"] == str((tmp_path / "images" / "val").resolve())
    # The split survives — val is a different dir from train (the whole point).
    assert out["val"] != out["train"]
    # train/val are absolute now, so the dataset-root key is redundant.
    assert "path" not in out
    # Non-routing keys are preserved.
    assert out["names"] == ["cell"]


def test_val_falls_back_to_train_when_export_did(tmp_path):
    # Shape export_yolo_v5plus writes for val_split=0 / single-image projects:
    # val already points at images/train so `yolo train` never sees an empty
    # val dir. Resolution must preserve that, not re-empty it.
    yc = {
        "path": str(tmp_path),
        "train": os.path.join("images", "train"),
        "val": os.path.join("images", "train"),
    }
    out = _resolve_training_yaml(str(tmp_path), yc)

    assert out["val"] == out["train"] == str((tmp_path / "images" / "train").resolve())
    assert "path" not in out


def test_missing_val_pointer_defaults_to_train(tmp_path):
    # A yaml with no val key must still yield a non-empty val (== train),
    # never a bare/relative pointer that would fail to resolve at train time.
    out = _resolve_training_yaml(str(tmp_path), {"names": ["x"]})

    assert out["train"] == str((tmp_path / "images" / "train").resolve())
    assert out["val"] == out["train"]


def test_does_not_mutate_input(tmp_path):
    yc = {
        "path": str(tmp_path),
        "train": os.path.join("images", "train"),
        "val": os.path.join("images", "val"),
    }
    _resolve_training_yaml(str(tmp_path), yc)

    # Caller's dict is untouched (resolution returns a fresh copy).
    assert yc["path"] == str(tmp_path)
    assert yc["val"] == os.path.join("images", "val")


def test_accepts_path_like_yaml_dir(tmp_path):
    # yaml_dir may arrive as a Path (yaml_path.parent) or a str — both work.
    yc = {"train": "images/train", "val": "images/val"}
    from_path = _resolve_training_yaml(Path(tmp_path), yc)
    from_str = _resolve_training_yaml(str(tmp_path), yc)

    assert from_path == from_str
    assert from_path["val"] == str((tmp_path / "images" / "val").resolve())
