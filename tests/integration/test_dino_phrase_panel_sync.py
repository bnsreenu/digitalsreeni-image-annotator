"""Regression test for issue #63 -- the DINO phrase panel must follow the
top class-list selection, not only the threshold-table row.

The phrase panel ("Phrases for: X") was bound only to the DINO threshold
table's ``itemSelectionChanged``. Selecting a different class in the *top*
class list retargeted the annotation tools' ``current_class`` but left the
phrase editor pointing at the previously-selected class, so you couldn't
add/edit phrases for the top-list class. The fix makes the top class list
the single source of truth: ``ClassController.on_class_selected`` selects
the matching threshold-table row, which cascades to the phrase panel via
the existing signal.

One real offscreen ImageAnnotator, driven through the real ``class_list``
``itemClicked`` signal, so the actual sidebar wiring (built in
ui.sidebar.build_sidebar) is exercised end-to-end -- deleting that ``connect``
must fail these tests, not just bypass them.
"""

import json

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFileDialog, QInputDialog, QMessageBox


@pytest.fixture(autouse=True)
def _no_native_dialogs(monkeypatch):
    """No modal may ever open in an offscreen run -- it hangs the test.
    Same hard safety net as tests/integration/test_project_roundtrip.py.
    """
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


@pytest.fixture
def window(qt_application, monkeypatch):
    from digitalsreeni_image_annotator.annotator_window import ImageAnnotator

    w = ImageAnnotator()
    # add_class / on_class_selected call auto_save; on an unsaved project that
    # pops a modal. Stub it out so the test never blocks on a dialog.
    monkeypatch.setattr(w, "auto_save", lambda: None)
    yield w
    w.deleteLater()


def _select_in_top_list(window, name):
    """Click a class in the top list, exactly as the user does.

    Goes through ``class_list.itemClicked`` (the signal ui.sidebar connects to
    ``on_class_selected``) rather than calling the handler directly, so the
    wiring itself is under test.
    """
    item = window.class_list.findItems(name, Qt.MatchFlag.MatchExactly)[0]
    window.class_list.setCurrentItem(item)
    window.class_list.itemClicked.emit(item)


def test_phrase_panel_follows_top_class_list_selection(window):
    for name in ("Drone", "rotor", "camera"):
        window.add_class(name)

    # After the adds, the last-added class drives both selectors (add_class
    # selects the freshly-added threshold row, which cascades to the panel).
    assert window.dino_class_table.selected_class_name() == "camera"
    assert window.dino_phrase_panel._active_class == "camera"

    # Select a DIFFERENT class in the TOP class list -- the #63 path.
    _select_in_top_list(window, "Drone")

    assert window.current_class == "Drone"
    # Both the threshold table and the phrase panel now track the top list.
    assert window.dino_class_table.selected_class_name() == "Drone"
    assert window.dino_phrase_panel._active_class == "Drone"


def test_phrase_panel_add_targets_top_list_class(window, monkeypatch):
    """A phrase added via the Add Phrase button after a top-list selection
    lands on that class, not the stale threshold-table row -- the concrete
    user-visible symptom in #63.

    Drives the real ``_add_phrase`` (which reads ``_active_class``) with a
    stubbed input dialog; asserting on a hardcoded dict key instead would be a
    tautology that passes even with the fix reverted.
    """
    for name in ("Drone", "camera"):
        window.add_class(name)
    # add_class leaves "camera" active in the panel; now pick "Drone" up top.
    _select_in_top_list(window, "Drone")

    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("quadcopter", True))
    window.dino_phrase_panel._add_phrase()

    assert "quadcopter" in window.dino_phrase_panel.get_phrases_for("Drone")
    assert "quadcopter" not in window.dino_phrase_panel.get_phrases_for("camera")


def test_select_class_by_name_is_noop_for_unknown_class(window):
    """Selecting a class absent from the threshold table (e.g. a Temp-* review
    class) leaves the current selection intact rather than clearing it."""
    for name in ("Drone", "camera"):
        window.add_class(name)
    _select_in_top_list(window, "Drone")

    assert window.dino_class_table.select_class_by_name("Temp-camera") is False
    # Selection unchanged by the failed lookup.
    assert window.dino_class_table.selected_class_name() == "Drone"
    assert window.dino_phrase_panel._active_class == "Drone"


def test_rename_class_carries_dino_threshold_row_and_phrases(window, monkeypatch):
    """Renaming a class must retarget its DINO threshold row and phrase list.

    Both registries are keyed by class name, so a rename that skips them leaves
    detection running under a dead class name and makes the next project load
    silently discard that class's phrases and thresholds. Found while asserting
    the #63 "top class list is the single source of truth" invariant.
    """
    window.add_class("Drone")
    window.add_class("camera")
    _select_in_top_list(window, "Drone")

    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("rotor blade", True))
    window.dino_phrase_panel._add_phrase()
    window.dino_class_table.set_thresholds("Drone", 0.4, 0.35, 0.6)

    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("UAV", True))
    item = window.class_list.findItems("Drone", Qt.MatchFlag.MatchExactly)[0]
    window.class_controller.rename_class(item)

    assert "UAV" in window.dino_class_table.get_class_names()
    assert "Drone" not in window.dino_class_table.get_class_names()
    # Thresholds ride along with the row rather than resetting to defaults.
    assert window.dino_class_table.get_thresholds_dict()["UAV"]["box"] == pytest.approx(0.4)
    # Phrases re-key; the custom phrase survives and the class-name phrase
    # (row 0, untouched by the user) follows the rename.
    phrases = window.dino_phrase_panel.get_phrases_for("UAV")
    assert phrases[0] == "UAV"
    assert "rotor blade" in phrases
    assert window.dino_phrase_panel._active_class == "UAV"

    # Consequence 1: detection no longer runs under the dead class name, and
    # the renamed class appears exactly once (not duplicated).
    config_names = [c["name"] for c in window.dino_controller._build_dino_class_configs()]
    assert "Drone" not in config_names
    assert config_names.count("UAV") == 1


def test_rename_class_phrases_survive_project_roundtrip(window, monkeypatch, tmp_path):
    """Consequence 2: a renamed class's phrases and thresholds survive save +
    reopen. Both are filtered against the live class list on load, so before
    the rename sync they were silently discarded on the next open.
    """
    proj = tmp_path / "p.iap"
    window.current_project_file = str(proj)
    window.current_project_dir = str(tmp_path)
    (tmp_path / "images").mkdir()

    window.add_class("Drone")
    _select_in_top_list(window, "Drone")
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("rotor blade", True))
    window.dino_phrase_panel._add_phrase()
    window.dino_class_table.set_thresholds("Drone", 0.4, 0.35, 0.6)

    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("UAV", True))
    item = window.class_list.findItems("Drone", Qt.MatchFlag.MatchExactly)[0]
    window.class_controller.rename_class(item)

    window.project_controller.save_project(show_message=False)
    saved = json.loads(proj.read_text(encoding="utf-8"))
    assert "UAV" in saved["dino_config"]["phrases"]
    assert "Drone" not in saved["dino_config"]["phrases"]

    window.project_controller.open_specific_project(str(proj))
    assert "rotor blade" in window.dino_phrase_panel.get_phrases_for("UAV")
    assert window.dino_class_table.get_thresholds_dict()["UAV"]["box"] == pytest.approx(0.4)


def test_rename_to_existing_class_is_rejected_intact(window, monkeypatch):
    """Renaming onto an existing class name must abort cleanly, leaving every
    name-keyed registry untouched.

    Before the guard this half-clobbered everything: the old class's id and
    colour overwrote the target's in class_mapping/class_colors, and the DINO
    table ended up with two rows named the same -- so get_thresholds_dict()
    dropped one row and _build_dino_class_configs() emitted the class twice.
    """
    window.add_class("Drone")
    window.add_class("camera")
    drone_id = window.class_mapping["Drone"]
    camera_id = window.class_mapping["camera"]

    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("camera", True))
    item = window.class_list.findItems("Drone", Qt.MatchFlag.MatchExactly)[0]
    window.class_controller.rename_class(item)

    assert window.class_mapping == {"Drone": drone_id, "camera": camera_id}
    assert window.dino_class_table.get_class_names() == ["Drone", "camera"]
    assert sorted(window.dino_class_table.get_thresholds_dict()) == ["Drone", "camera"]
    names = [c["name"] for c in window.dino_controller._build_dino_class_configs()]
    assert names.count("camera") == 1
    # The top list item keeps its original text -- no half-applied rename.
    assert item.text() == "Drone"


def test_rename_onto_pending_temp_class_is_rejected(window, monkeypatch):
    """Renaming onto a *pending* Temp-* review class must be rejected.

    Temp-* classes are registered in ``class_colors`` + ``class_list`` only --
    ``add_temp_classes`` never writes ``class_mapping`` -- so a collision check
    against ``class_mapping`` would let this through. The real class's
    annotation bucket would then merge into the temp one, and the user's next
    Reject (which sweeps Temp-* and *deletes*) would destroy it silently.
    """
    window.add_class("Drone")
    window.image_label.annotations["Drone"] = [
        {"category_name": "Drone", "number": 1, "segmentation": [1, 1, 5, 1, 5, 5]}
    ]
    window.dino_controller.add_temp_classes(
        {"Temp-camera": [{"category_name": "Temp-camera", "number": 1,
                          "segmentation": [2, 2, 6, 2, 6, 6]}]}
    )

    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("Temp-camera", True))
    item = window.class_list.findItems("Drone", Qt.MatchFlag.MatchExactly)[0]
    window.class_controller.rename_class(item)

    # Rejected outright -- nothing renamed, no duplicate list entry.
    assert item.text() == "Drone"
    assert "Drone" in window.class_mapping
    assert len(window.class_list.findItems("Temp-camera", Qt.MatchFlag.MatchExactly)) == 1
    assert len(window.image_label.annotations["Drone"]) == 1

    # The decisive assertion: the real class survives the review sweep.
    window.dino_controller.reject_visible_temp_classes()
    assert len(window.image_label.annotations["Drone"]) == 1
    assert "Temp-camera" not in window.image_label.annotations


def test_rename_into_temp_namespace_is_rejected(window, monkeypatch):
    """'Temp-' is a reserved prefix with destructive semantics (the review
    sweeps Temp-* wholesale), so renaming *into* it is refused even when the
    name is otherwise free."""
    window.add_class("Drone")

    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("Temp-fresh", True))
    item = window.class_list.findItems("Drone", Qt.MatchFlag.MatchExactly)[0]
    window.class_controller.rename_class(item)

    assert item.text() == "Drone"
    assert "Drone" in window.class_mapping
    assert "Temp-fresh" not in window.image_label.class_colors
    assert window.dino_class_table.get_class_names() == ["Drone"]


def test_renaming_a_pending_temp_class_warns_instead_of_failing_silently(
    window, monkeypatch
):
    """A Temp-* class isn't in class_mapping, so it used to hit the bailout and
    return with no dialog at all -- the rejection a user is most likely to hit
    by accident was the one that gave no feedback."""
    window.dino_controller.add_temp_classes(
        {"Temp-camera": [{"category_name": "Temp-camera", "number": 1,
                          "segmentation": [2, 2, 6, 2, 6, 6]}]}
    )
    warned = []
    monkeypatch.setattr(
        QMessageBox, "warning", staticmethod(lambda *a, **k: warned.append(a))
    )
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("camera", True))

    item = window.class_list.findItems("Temp-camera", Qt.MatchFlag.MatchExactly)[0]
    window.class_controller.rename_class(item)

    assert warned, "renaming a pending review class must tell the user why"
    assert item.text() == "Temp-camera"
    assert "camera" not in window.class_mapping


def test_rename_strips_surrounding_whitespace(window, monkeypatch):
    """Unstripped input would let "Drone " clear both the collision check and
    the Temp- prefix check, creating a class distinguishable from "Drone" only
    by whitespace -- which then keys all seven registries."""
    window.add_class("Drone")
    item = window.class_list.findItems("Drone", Qt.MatchFlag.MatchExactly)[0]

    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("  UAV  ", True))
    window.class_controller.rename_class(item)

    assert "UAV" in window.class_mapping
    assert item.text() == "UAV"
    assert window.dino_class_table.get_class_names() == ["UAV"]


def test_rename_to_same_name_with_whitespace_is_a_noop(window, monkeypatch):
    """"Drone " strips to the current name, so there is nothing to do -- it must
    not be treated as a collision with itself."""
    window.add_class("Drone")
    item = window.class_list.findItems("Drone", Qt.MatchFlag.MatchExactly)[0]

    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("  Drone ", True))
    window.class_controller.rename_class(item)

    assert list(window.class_mapping) == ["Drone"]
    assert item.text() == "Drone"


def test_rename_class_carries_visibility(window, monkeypatch):
    """Visibility is name-keyed and read via .get(name, True), so a rename that
    skips it silently un-hides a hidden class."""
    window.add_class("Drone")
    item = window.class_list.findItems("Drone", Qt.MatchFlag.MatchExactly)[0]
    # Hide it the way the user does -- unchecking the list checkbox, which
    # fires itemChanged -> toggle_class_visibility. Poking class_visibility
    # directly would leave the checkbox checked and misrepresent the state.
    item.setCheckState(Qt.CheckState.Unchecked)
    assert window.image_label.class_visibility["Drone"] is False

    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("UAV", True))
    window.class_controller.rename_class(item)

    assert window.image_label.class_visibility.get("UAV") is False
    # No stale key left behind under the old name.
    assert "Drone" not in window.image_label.class_visibility


def test_rename_class_keeps_user_customised_first_phrase(window, monkeypatch):
    """Row 0 is renameable independently of the class, so a rename must not
    clobber a prompt the user deliberately customised."""
    window.add_class("Drone")
    window.dino_phrase_panel._phrases["Drone"][0] = "small quadcopter"

    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("UAV", True))
    item = window.class_list.findItems("Drone", Qt.MatchFlag.MatchExactly)[0]
    window.class_controller.rename_class(item)

    assert window.dino_phrase_panel.get_phrases_for("UAV") == ["small quadcopter"]
