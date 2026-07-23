"""Unit tests for KeypointSchemaDialog's stateful bits (issue #35).

Focus: the point-reorder must remap skeleton edges AND flip partners through the
swap permutation, not just the display names (senior-review P1), and a plain
build round-trips through _collect/get_schema.
"""

import pytest

from src.digitalsreeni_image_annotator.dialogs.keypoint_schema_dialog import (
    KeypointSchemaDialog,
)


@pytest.fixture
def dialog(qtbot):
    def _make(schema):
        d = KeypointSchemaDialog(None, class_name="person", schema=schema)
        qtbot.addWidget(d)
        return d

    return _make


def _accept(d):
    d._on_accept()
    return d.get_schema()


def test_build_roundtrips(dialog):
    schema = {"names": ["a", "b", "c"], "skeleton": [[0, 2]], "flip_idx": [0, 2, 1]}
    d = dialog(schema)
    assert _accept(d) == schema


def test_move_point_remaps_skeleton_and_flip(dialog):
    # a=self, b<->c ; edge a-c. Move row 0 (a) down past b (swap 0,1).
    d = dialog({"names": ["a", "b", "c"], "skeleton": [[0, 2]], "flip_idx": [0, 2, 1]})
    d.points_table.setCurrentCell(0, 0)
    d._move_point(1)
    out = _accept(d)
    # New order: b, a, c. b<->c must now be 0<->2; a self; edge a-c now 1-2.
    assert out["names"] == ["b", "a", "c"]
    assert out["flip_idx"] == [2, 1, 0]
    assert out["skeleton"] == [[1, 2]]


def test_move_point_is_reversible(dialog):
    schema = {"names": ["a", "b", "c"], "skeleton": [[0, 2]], "flip_idx": [0, 2, 1]}
    d = dialog(schema)
    d.points_table.setCurrentCell(0, 0)
    d._move_point(1)   # a down
    d.points_table.setCurrentCell(1, 0)
    d._move_point(-1)  # a back up
    assert _accept(d) == schema


def test_reject_duplicate_names(dialog, monkeypatch):
    d = dialog({"names": ["a", "b"], "skeleton": [], "flip_idx": [0, 1]})
    d.points_table.item(1, 0).setText("a")  # make them duplicate
    warned = []
    monkeypatch.setattr(
        "digitalsreeni_image_annotator.dialogs.keypoint_schema_dialog.QMessageBox.warning",
        lambda *a, **k: warned.append(a),
    )
    d._on_accept()
    assert d.get_schema() is None and warned


def test_reject_non_reciprocal_flip_partners(dialog, monkeypatch):
    # Build a 3-cycle a->b->c->a (valid bijection, but not self-inverse: a
    # flips to b, yet b doesn't flip back to a). sanitize_schema would
    # otherwise silently reset ALL flip partners to identity with no warning;
    # the dialog must instead refuse to close and tell the user why.
    d = dialog({"names": ["a", "b", "c"], "skeleton": [], "flip_idx": [0, 1, 2]})
    d.points_table.cellWidget(0, 1).setCurrentIndex(2)  # a's partner -> b
    d.points_table.cellWidget(1, 1).setCurrentIndex(3)  # b's partner -> c
    d.points_table.cellWidget(2, 1).setCurrentIndex(1)  # c's partner -> a
    warned = []
    monkeypatch.setattr(
        "digitalsreeni_image_annotator.dialogs.keypoint_schema_dialog.QMessageBox.warning",
        lambda *a, **k: warned.append(a),
    )
    d._on_accept()
    assert d.get_schema() is None and warned


# --- combo-label sync on a plain name edit (no add/remove/move in between) --
#
# _refresh_index_combos() only ran on add/remove/move, so typing a name into
# an existing row left every OTHER row's flip-partner combo (and the skeleton
# From/To combos) showing a bare index ("6") until some unrelated row op
# happened to trigger a rebuild. itemChanged on the Name column now triggers
# it directly.

def test_typed_name_immediately_labels_other_flip_combo(dialog):
    d = dialog(None)  # two blank default rows
    d.points_table.item(0, 0).setText("nose")
    combo = d.points_table.cellWidget(1, 1)  # row 1's flip-partner combo
    assert combo.itemText(1) == "1: nose"  # index 0 = "self", index 1 = point 1


def test_typed_name_immediately_labels_skeleton_combo(dialog):
    d = dialog(None)
    d._add_edge()  # edge between the two default blank points (0, 1)
    d.points_table.item(0, 0).setText("nose")
    from_combo = d.skeleton_table.cellWidget(0, 0)
    assert from_combo.itemText(0) == "1: nose"


def test_renaming_last_added_point_labels_earlier_rows(dialog):
    # Mirrors the reported bug: name the newest row and check an EARLIER row's
    # flip-partner combo already shows it, with no add/remove/move in between.
    d = dialog(None)
    d._add_point()
    d.points_table.item(2, 0).setText("shoulder")
    combo = d.points_table.cellWidget(0, 1)  # row 0's flip-partner combo
    assert combo.itemText(3) == "3: shoulder"


def test_renaming_does_not_disturb_current_selection(dialog):
    d = dialog(None)
    d._add_point()
    combo = d.points_table.cellWidget(0, 1)
    combo.setCurrentIndex(2)  # select point 2 (0-based index 1) as flip partner
    d.points_table.item(2, 0).setText("shoulder")
    assert combo.currentIndex() == 2  # selection preserved through the rebuild
