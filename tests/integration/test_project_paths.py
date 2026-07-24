"""Project portability (relative paths), load validation, and unsaved-project
recovery — integration tests (#41 / #42).

One real offscreen ImageAnnotator. Saves a project and confirms both absolute
and relative image-path dicts land in the `.iap`; moves the whole project dir
and confirms it still opens with no missing-image prompt; covers the v1
(no image_paths_rel) fallback and structural-validation rejection; and exercises
the silent recovery write when there is no project file yet.
"""

import json
import shutil
from pathlib import Path

import pytest
from PyQt6.QtCore import QSettings
from PyQt6.QtGui import QColor, QImage


@pytest.fixture
def window(qt_application):
    from digitalsreeni_image_annotator.annotator_window import ImageAnnotator

    w = ImageAnnotator()
    yield w
    w.deleteLater()


@pytest.fixture(autouse=True)
def _no_native_dialogs(monkeypatch):
    """No modal may open in an offscreen run — patch every reachable one."""
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


def _make_project(window, project_dir):
    """A project dir with one real PNG in images/, wired onto the window."""
    images_dir = project_dir / "images"
    images_dir.mkdir(parents=True)
    img = QImage(16, 12, QImage.Format.Format_RGB32)
    img.fill(QColor("red"))
    img.save(str(images_dir / "a.png"))

    window.current_project_file = str(project_dir / "proj.iap")
    window.current_project_dir = str(project_dir)
    window.add_class("cell", QColor("#ff0000"))
    window.all_images.append({"file_name": "a.png", "width": 16, "height": 12,
                              "id": 1, "is_multi_slice": False})
    window.image_paths["a.png"] = str(images_dir / "a.png")
    window.all_annotations["a.png"] = {"cell": []}
    return Path(window.current_project_file)


def test_save_writes_both_path_dicts(window, tmp_path):
    proj = _make_project(window, tmp_path / "proj")
    window.project_controller.save_project(show_message=False)

    data = json.loads(proj.read_text(encoding="utf-8"))
    assert "image_paths" in data and "image_paths_rel" in data
    assert data["image_paths_rel"]["a.png"] == "images/a.png"
    assert "\\" not in data["image_paths_rel"]["a.png"]  # POSIX separators


def test_moved_project_reopens_without_missing_prompt(window, tmp_path, monkeypatch):
    src = tmp_path / "proj"
    _make_project(window, src)
    window.project_controller.save_project(show_message=False)

    # Move the whole project dir to a new path (a different machine, in effect).
    dst = tmp_path / "moved"
    shutil.move(str(src), str(dst))

    called = {"missing": False}
    monkeypatch.setattr(
        window.project_controller, "handle_missing_images",
        lambda missing: called.__setitem__("missing", True),
    )
    window.project_controller.open_specific_project(str(dst / "proj.iap"))

    assert called["missing"] is False
    assert Path(window.image_paths["a.png"]).resolve() == (dst / "images" / "a.png").resolve()


def test_v1_project_without_rel_resolves_via_images_convention(window, tmp_path):
    src = tmp_path / "proj"
    _make_project(window, src)
    window.project_controller.save_project(show_message=False)

    proj = src / "proj.iap"
    data = json.loads(proj.read_text(encoding="utf-8"))
    del data["image_paths_rel"]                       # simulate a pre-#42 file
    data["image_paths"] = {"a.png": "/nonexistent/a.png"}  # abs fallback dead
    proj.write_text(json.dumps(data), encoding="utf-8")

    window.project_controller.open_specific_project(str(proj))
    assert Path(window.image_paths["a.png"]).exists()


def test_structurally_broken_project_raises(window, tmp_path):
    proj = tmp_path / "broken.iap"
    proj.write_text(json.dumps({"images": "not-a-list"}), encoding="utf-8")
    with pytest.raises(ValueError):
        window.project_controller.open_specific_project(str(proj))


def test_autosave_with_no_project_writes_recovery(window, tmp_path, monkeypatch):
    """With no project file, auto_save writes a silent recovery snapshot (no
    dialog) that a fresh window can load back."""
    from digitalsreeni_image_annotator.core import recovery

    rec_dir = tmp_path / "rec"
    rec_dir.mkdir()
    ini = QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat)
    monkeypatch.setattr(recovery, "recovery_dir", lambda: str(rec_dir))
    monkeypatch.setattr(
        recovery, "_settings",
        lambda settings=None: settings if settings is not None else ini,
    )

    # A real image on disk so the snapshot's absolute path resolves on restore.
    images_dir = tmp_path / "imgs"
    images_dir.mkdir()
    img = QImage(4, 4, QImage.Format.Format_RGB32)
    img.fill(QColor("red"))
    img.save(str(images_dir / "a.png"))

    if hasattr(window, "current_project_file"):
        del window.current_project_file
    window.add_class("cell", QColor("#ff0000"))
    window.all_images.append({"file_name": "a.png", "width": 4, "height": 4,
                              "id": 1, "is_multi_slice": False})
    window.image_paths["a.png"] = str(images_dir / "a.png")
    window.all_annotations["a.png"] = {
        "cell": [{"segmentation": [0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
                  "category_name": "cell"}]
    }

    window.project_controller.auto_save()

    path = recovery.pending_recovery()
    assert path and Path(path).exists()

    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    assert "cell" in {c["name"] for c in data["classes"]}
    assert any(i["file_name"] == "a.png" for i in data["images"])


def _redirect_recovery(tmp_path, monkeypatch):
    from digitalsreeni_image_annotator.core import recovery

    rec_dir = tmp_path / "rec"
    rec_dir.mkdir()
    ini = QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat)
    monkeypatch.setattr(recovery, "recovery_dir", lambda: str(rec_dir))
    monkeypatch.setattr(
        recovery, "_settings",
        lambda settings=None: settings if settings is not None else ini,
    )
    return recovery


def test_offer_recovery_keeps_snapshot_on_successful_restore(window, tmp_path, monkeypatch):
    # The restored project is still unsaved, so the snapshot must survive until
    # the first real save — a re-crash before then can still re-offer it.
    recovery = _redirect_recovery(tmp_path, monkeypatch)
    recovery.write_recovery(
        {"classes": [{"name": "cell", "color": "#ff0000"}], "images": []}
    )
    window.project_controller.offer_recovery()  # question() is patched to Yes
    assert "cell" in window.class_mapping
    assert recovery.pending_recovery() is not None


def test_offer_recovery_clears_corrupt_snapshot(window, tmp_path, monkeypatch):
    # A structurally broken snapshot is unusable — drop it so it isn't re-offered
    # on every launch.
    recovery = _redirect_recovery(tmp_path, monkeypatch)
    recovery.write_recovery({"images": "nope"})  # validation will reject this
    window.project_controller.offer_recovery()
    assert recovery.pending_recovery() is None
