"""ProjectController `.iap` save/load roundtrip tests (fork issue #35).

`.iap` save/load is the persistence backbone of the app, yet
`controllers/project_controller.py` had no tests. These build one real
ImageAnnotator (offscreen) and exercise the controller directly: a
save -> reload roundtrip that deep-compares state (classes, colours,
annotations, keypoint schemas), plus regression coverage for the
`is_loading_project` guard (the v0.8.12 autosave-during-load corruption
bug documented as critical in CLAUDE.md) and malformed-input handling.

No production code is changed — tests only.
"""

import json
from pathlib import Path

import pytest
from PyQt6.QtGui import QColor, QImage


@pytest.fixture
def window(qt_application):
    from digitalsreeni_image_annotator.annotator_window import ImageAnnotator

    w = ImageAnnotator()
    yield w
    w.deleteLater()


@pytest.fixture(autouse=True)
def _no_native_dialogs(monkeypatch):
    """No modal may ever open in an offscreen run — it hangs the test.

    save_project prompts a QFileDialog (unset project file) or a copy
    QMessageBox (images outside <dir>/images/); load can prompt for missing
    images. The tests avoid triggering these, but patch every reachable modal
    to a non-blocking default as a hard safety net.
    """
    from PyQt6.QtWidgets import QFileDialog, QMessageBox

    monkeypatch.setattr(
        QMessageBox, "question",
        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes),
    )
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(QMessageBox, "critical", staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(
        QFileDialog, "getOpenFileNames", staticmethod(lambda *a, **k: ([], ""))
    )


POLY = {"segmentation": [1.0, 1.0, 10.0, 1.0, 10.0, 8.0], "area": 31.5,
        "category_id": 1, "category_name": "cell", "number": 1}
POSE = {"keypoints": [3.0, 3.0, 2, 0.0, 0.0, 0], "num_keypoints": 1,
        "bbox": [2.0, 2.0, 4.0, 4.0], "category_id": 2,
        "category_name": "pose", "number": 1}
SCHEMA = {"names": ["head", "tail"], "skeleton": [[0, 1]], "flip_idx": [0, 1]}


def make_project(window, tmp_path):
    """Point the window at tmp_path as its project dir, with one real PNG
    already inside <project_dir>/images/ so save_project never prompts.

    current_project_file is set *before* add_class so the auto_save it
    triggers writes to the temp project instead of opening a QFileDialog.
    Returns the `.iap` path (state only — the caller saves).
    """
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    img = QImage(16, 12, QImage.Format.Format_RGB32)
    img.fill(QColor("red"))
    img.save(str(images_dir / "a.png"))

    window.current_project_file = str(tmp_path / "proj.iap")
    window.current_project_dir = str(tmp_path)
    window.add_class("cell", QColor("#ff0000"))
    window.add_class("pose", QColor("#00aa00"))
    window.keypoint_schemas["pose"] = dict(SCHEMA)
    window.all_images.append({"file_name": "a.png", "width": 16, "height": 12,
                              "id": 1, "is_multi_slice": False})
    window.image_paths["a.png"] = str(images_dir / "a.png")
    window.all_annotations["a.png"] = {"cell": [dict(POLY)], "pose": [dict(POSE)]}
    return tmp_path / "proj.iap"


def test_save_writes_valid_json(window, tmp_path):
    """The harness validator: a save produces a parseable `.iap` with the
    expected top-level keys, and a keypoint_schema only on the pose class."""
    proj = make_project(window, tmp_path)
    window.project_controller.save_project(show_message=False)

    assert proj.exists()
    data = json.loads(proj.read_text(encoding="utf-8"))
    for key in ("classes", "images", "image_paths", "notes",
                "creation_date", "last_modified"):
        assert key in data, f"missing top-level key {key!r}"

    by_name = {c["name"]: c for c in data["classes"]}
    assert "keypoint_schema" in by_name["pose"]
    assert "keypoint_schema" not in by_name["cell"]


def test_roundtrip_classes_and_colors(window, tmp_path):
    proj = make_project(window, tmp_path)
    window.project_controller.save_project(show_message=False)

    window.project_controller.open_specific_project(str(proj))

    assert "cell" in window.class_mapping
    assert "pose" in window.class_mapping
    assert window.image_label.class_colors["cell"].name() == "#ff0000"


def test_roundtrip_annotations_deep_equal(window, tmp_path):
    proj = make_project(window, tmp_path)
    window.project_controller.save_project(show_message=False)

    window.project_controller.open_specific_project(str(proj))

    loaded = window.all_annotations["a.png"]
    assert loaded["cell"][0]["segmentation"] == POLY["segmentation"]

    pose = loaded["pose"][0]
    assert pose["keypoints"] == POSE["keypoints"]
    assert pose["num_keypoints"] == POSE["num_keypoints"]
    assert pose["bbox"] == POSE["bbox"]
    # A pose instance is discriminated by the ABSENCE of a segmentation key
    # (ADR-029) — assert the key is absent, not None.
    assert "segmentation" not in pose


def test_roundtrip_keypoint_schema(window, tmp_path):
    proj = make_project(window, tmp_path)
    window.project_controller.save_project(show_message=False)

    window.project_controller.open_specific_project(str(proj))

    assert window.keypoint_schemas["pose"] == SCHEMA


def test_malformed_keypoint_schema_dropped(window, tmp_path):
    """A malformed pose schema on disk is dropped with a warning; the class
    still loads. Mirrors the DINO validate-on-load pattern."""
    proj = make_project(window, tmp_path)
    window.project_controller.save_project(show_message=False)

    data = json.loads(proj.read_text(encoding="utf-8"))
    for cls in data["classes"]:
        if cls["name"] == "pose":
            cls["keypoint_schema"] = {"names": "oops"}  # names must be a list
    proj.write_text(json.dumps(data), encoding="utf-8")

    window.project_controller.open_specific_project(str(proj))

    assert "pose" in window.class_mapping
    assert "pose" not in window.keypoint_schemas


def test_autosave_noop_while_loading(window, tmp_path):
    """v0.8.12 regression: auto_save must be a no-op while a project is
    loading, or it corrupts the file mid-load. This fails if the
    `is_loading_project` guard at project_controller.py:553-554 is removed."""
    proj = make_project(window, tmp_path)
    window.project_controller.save_project(show_message=False)

    before = Path(proj).read_bytes()
    window.is_loading_project = True
    window.project_controller.auto_save()
    assert Path(proj).read_bytes() == before


def test_open_malformed_json_resets_flag(window, tmp_path):
    """A failed open must reset is_loading_project (else every later save is
    silently suppressed by the guard)."""
    bad = tmp_path / "bad.iap"
    bad.write_text("{ not json", encoding="utf-8")

    with pytest.raises(Exception):
        window.project_controller.open_specific_project(str(bad))

    assert window.is_loading_project is False


def test_multi_slice_save_shape(window, tmp_path):
    """Save-side shape for a multi-slice image: per-slice annotations plus
    dimensions/shape fields. (Full multi-dim reload is covered elsewhere.)"""
    proj = make_project(window, tmp_path)

    stack_path = tmp_path / "images" / "stack.tif"
    stack_path.write_bytes(b"placeholder")  # only presence matters for save
    window.all_images.append({"file_name": "stack.tif", "width": 4, "height": 4,
                              "id": 2, "is_multi_slice": True})
    window.image_slices["stack"] = [
        ("stack_Z1", QImage(4, 4, QImage.Format.Format_RGB32))
    ]
    window.image_dimensions["stack"] = ["Z", "H", "W"]
    window.image_shapes["stack"] = (1, 4, 4)
    window.all_annotations["stack_Z1"] = {"cell": [dict(POLY)]}
    window.image_paths["stack.tif"] = str(stack_path)

    window.project_controller.save_project(show_message=False)

    data = json.loads(proj.read_text(encoding="utf-8"))
    stack = next(i for i in data["images"] if i["file_name"] == "stack.tif")
    assert stack["is_multi_slice"] is True
    assert stack["slices"][0]["name"] == "stack_Z1"
    assert (stack["slices"][0]["annotations"]["cell"][0]["segmentation"]
            == POLY["segmentation"])
    assert "dimensions" in stack
    assert "shape" in stack


def test_dino_config_roundtrip(window, tmp_path):
    proj = make_project(window, tmp_path)
    # The phrase panel stores a list of phrases per class (dict[str, list[str]]).
    window.dino_phrase_panel.set_phrases({"cell": ["a cell"]})
    window.project_controller.save_project(show_message=False)

    data = json.loads(proj.read_text(encoding="utf-8"))
    assert "dino_config" in data
    assert "phrases" in data["dino_config"]
    assert data["dino_config"]["phrases"]["cell"] == ["a cell"]

    window.project_controller.open_specific_project(str(proj))
    assert window.dino_phrase_panel.get_all_phrases().get("cell") == ["a cell"]
