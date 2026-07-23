"""
End-to-end coverage of controllers.io_controller.import_annotations (issue
#35 PR-2 senior-review follow-up) — drives the real function through a real
offscreen ImageAnnotator, with QFileDialog/QMessageBox mocked out.

The rebuild-branch logic itself is unit-tested in isolation in
test_io_controller_rebuild.py; this file covers the wiring around it: the
3-tuple unpack from import_coco_json, the schema-registration loop, and a
malformed COCO file surfacing as a QMessageBox warning instead of an
uncaught exception.
"""

import json

import pytest
from PIL import Image

from digitalsreeni_image_annotator.controllers import io_controller as iomod


@pytest.fixture
def window(qt_application):
    from digitalsreeni_image_annotator.annotator_window import ImageAnnotator

    w = ImageAnnotator()
    yield w
    w.deleteLater()


def _write_coco_with_pose(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    Image.new("RGB", (100, 100)).save(images_dir / "img.png")

    coco = {
        "images": [{"id": 1, "file_name": "img.png", "width": 100, "height": 100}],
        "categories": [
            {
                "id": 1, "name": "person",
                "keypoints": ["nose", "l_eye", "r_eye"],
                "skeleton": [[1, 2], [1, 3]],
                "flip_idx": [0, 2, 1],
            },
            {"id": 2, "name": "cell"},
        ],
        "annotations": [
            {
                "id": 1, "image_id": 1, "category_id": 1,
                "keypoints": [10, 10, 2, 20, 20, 1, 0, 0, 0],
                "num_keypoints": 2,
                "bbox": [5, 5, 20, 20],
            },
            {"id": 2, "image_id": 1, "category_id": 2, "bbox": [50, 50, 10, 10]},
        ],
    }
    json_path = tmp_path / "annotations.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(coco, f)
    return json_path


def test_import_registers_schema_and_rebuilds_both_annotation_shapes(window, monkeypatch, tmp_path):
    json_path = _write_coco_with_pose(tmp_path)

    # No project file exists yet -> auto_save() would pop a blocking "Save
    # Project" QFileDialog (see test_smoke.py's test_keypoint_tool_activates
    # for the same pattern).
    monkeypatch.setattr(window, "auto_save", lambda: None)
    monkeypatch.setattr(
        iomod.QFileDialog, "getOpenFileName", lambda *a, **k: (str(json_path), "")
    )
    monkeypatch.setattr(iomod.QMessageBox, "information", lambda *a, **k: None)
    monkeypatch.setattr(iomod.QMessageBox, "warning", lambda *a, **k: None)
    # Deterministic today because the fixture writes a real img.png (so
    # images_not_found stays empty) -- stub the "missing images" prompt too
    # so a future fixture change fails loudly instead of hanging on a modal.
    monkeypatch.setattr(
        iomod.QMessageBox, "question", lambda *a, **k: iomod.QMessageBox.StandardButton.Yes
    )

    window.import_format_selector.setCurrentText("COCO JSON")
    iomod.import_annotations(window)

    assert window.keypoint_schemas["person"] == {
        "names": ["nose", "l_eye", "r_eye"],
        "skeleton": [[0, 1], [0, 2]],
        "flip_idx": [0, 2, 1],
    }

    person_ann = window.all_annotations["img.png"]["person"][0]
    assert person_ann["keypoints"] == [10.0, 10.0, 2.0, 20.0, 20.0, 1.0, 0.0, 0.0, 0.0]
    assert "segmentation" not in person_ann
    assert "type" not in person_ann

    cell_ann = window.all_annotations["img.png"]["cell"][0]
    assert "keypoints" not in cell_ann
    assert cell_ann["bbox"] == [50.0, 50.0, 10.0, 10.0]


def test_double_click_does_not_crash_when_keypoint_instance_present(window):
    """ADR-029's crash rationale, locked behind an executable assertion
    rather than prose (senior-review follow-up): start_polygon_edit
    iterates EVERY annotation across every class unconditionally, doing
    `annotation["segmentation"][0::2]` behind an existence-only (not
    None-guarded) `if "segmentation" in annotation` check. Feeds the real
    `_rebuild_imported_annotation` output straight into the real
    `ImageLabel.start_polygon_edit` -- if the rebuild ever regresses back to
    a shared dict with `"segmentation": None` on a keypoint annotation, this
    raises `TypeError: 'NoneType' object is not subscriptable` on ANY
    double-click anywhere on the canvas, regardless of what was clicked."""
    raw_pose_ann = {
        "keypoints": [10, 10, 2, 20, 20, 1],
        "num_keypoints": 2,
        "bbox": [5, 5, 20, 20],
        "category_id": 1,
    }
    raw_plain_ann = {
        "segmentation": [50, 50, 60, 50, 60, 60, 50, 60],
        "bbox": [50, 50, 10, 10],
        "category_id": 2,
        "type": "polygon",
    }
    il = window.image_label
    il.annotations = {
        "person": [iomod._rebuild_imported_annotation(raw_pose_ann, "person", 1)],
        "cell": [iomod._rebuild_imported_annotation(raw_plain_ann, "cell", 1)],
    }

    il.start_polygon_edit((55, 55))  # inside the plain "cell" polygon


def test_malformed_coco_file_surfaces_as_warning_not_crash(window, monkeypatch, tmp_path):
    bad_path = tmp_path / "not_json.json"
    bad_path.write_text("{not valid json", encoding="utf-8")

    monkeypatch.setattr(
        iomod.QFileDialog, "getOpenFileName", lambda *a, **k: (str(bad_path), "")
    )
    warnings = []
    monkeypatch.setattr(
        iomod.QMessageBox, "warning", lambda *a, **k: warnings.append(a)
    )

    window.import_format_selector.setCurrentText("COCO JSON")
    iomod.import_annotations(window)  # must not raise

    assert warnings


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
