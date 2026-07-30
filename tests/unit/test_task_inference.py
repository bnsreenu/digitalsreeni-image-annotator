"""Training task inference and dataset summary (issue #73, ADR-042).

What kind of model to train is entailed by what was annotated — boxes mean
detect, polygons mean segment, keypoints mean pose — so asking the user is
asking them to restate their own data.

The reason this is a module of its own rather than a method on the dialog:
``train_model`` already infers the task a second time, from the prepared
dataset YAML, and raises pre-flight if the loaded model disagrees. A dialog
that announced one task while the trainer decided on another would be a bug by
construction. Deriving both from one place is the fix, and these tests are what
pin the rules.
"""

import subprocess
import sys

import pytest
from PyQt6.QtWidgets import QWidget

from src.digitalsreeni_image_annotator.core import task_inference as ti


def _polygon(name="cell"):
    return {"segmentation": [0, 0, 10, 0, 10, 10], "category_name": name}


def _bbox(name="cell"):
    return {"bbox": [0, 0, 10, 10], "category_name": name}


def _pose(k=3, name="person"):
    return {
        "keypoints": [c for i in range(k) for c in (i, i, 2)],
        "num_keypoints": k,
        "bbox": [0, 0, 10, 10],
        "category_name": name,
    }


def _project(**by_class):
    return {"img.png": dict(by_class)}


# --- Qt-free ---------------------------------------------------------------


def test_task_inference_imports_without_qt():
    code = (
        "import sys;"
        "sys.path.insert(0, 'src');"
        "import digitalsreeni_image_annotator.core.task_inference as m;"
        "qt = [n for n in sys.modules if n.startswith('PyQt6')];"
        "assert not qt, qt;"
        "print('clean')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "clean" in result.stdout


# --- task inference --------------------------------------------------------


def test_boxes_only_means_detect():
    task, reason = ti.infer_task(_project(cell=[_bbox(), _bbox()]))
    assert task == ti.TASK_DETECT
    assert "no polygons" in reason


def test_polygons_mean_segment():
    task, reason = ti.infer_task(_project(cell=[_polygon()]))
    assert task == ti.TASK_SEGMENT
    assert "polygon" in reason


def test_keypoints_mean_pose():
    task, reason = ti.infer_task(_project(person=[_pose()]))
    assert task == ti.TASK_POSE
    assert "pose instance" in reason


def test_pose_wins_over_everything_and_says_so():
    """A pose instance cannot be trained as anything else, so it takes
    precedence — but a mixed project must be told, since it cannot export."""
    task, reason = ti.infer_task(_project(cell=[_polygon()], person=[_pose()]))
    assert task == ti.TASK_POSE
    assert "cannot be trained alongside" in reason


def test_polygons_win_over_boxes_and_say_so():
    """A polygon carries strictly more information than the box it implies, but
    "segment" on a mostly-boxes project is a surprise worth explaining."""
    task, reason = ti.infer_task(_project(cell=[_polygon()], other=[_bbox()]))
    assert task == ti.TASK_SEGMENT
    assert "box-only" in reason


def test_an_empty_project_has_no_task():
    """Guessing a default here would surface as a confusing failure later."""
    task, reason = ti.infer_task({})
    assert task is None
    assert "no annotations" in reason


def test_temp_classes_are_not_training_data():
    """Pending review results are not labels yet."""
    project = {"img.png": {"Temp-cell": [_polygon("Temp-cell")]}}
    task, _reason = ti.infer_task(project)
    assert task is None


def test_shape_counts_span_slices():
    project = {
        "stack_Z1": {"cell": [_polygon()]},
        "stack_Z2": {"cell": [_polygon(), _bbox()]},
    }
    counts = ti.count_shapes(project)
    assert counts == {"polygon": 2, "bbox": 1, "pose": 0}


def test_a_pose_is_never_counted_as_a_polygon():
    """The absence of a segmentation key is the discriminator (ADR-029)."""
    counts = ti.count_shapes(_project(person=[_pose()]))
    assert counts["pose"] == 1
    assert counts["polygon"] == 0


# --- dataset summary -------------------------------------------------------


def test_summary_counts_annotated_and_unlabelled_images():
    project = {"a.png": {"cell": [_polygon()]}, "b.png": {}}
    summary = ti.summarise_dataset(project, ["a.png", "b.png", "c.png"])
    assert summary["images"] == 3
    assert summary["annotated_images"] == 1
    assert summary["unlabelled_images"] == 2


def test_summary_lists_the_classes_actually_used():
    project = {"a.png": {"cell": [_polygon()], "empty": []}}
    summary = ti.summarise_dataset(project, ["a.png"])
    assert summary["classes"] == ["cell"], "a class with no annotations is not used"


def test_summary_excludes_temp_classes():
    project = {"a.png": {"Temp-cell": [_polygon("Temp-cell")]}}
    summary = ti.summarise_dataset(project, ["a.png"])
    assert summary["classes"] == []
    assert summary["annotated_images"] == 0


def test_summary_of_an_empty_project():
    summary = ti.summarise_dataset({}, [])
    assert summary["images"] == 0
    assert summary["annotations"] == 0


# --- pre-flight blockers ---------------------------------------------------


def test_a_clean_pose_project_has_no_blockers():
    schemas = {"person": {"names": ["a", "b", "c"], "skeleton": [], "flip_idx": []}}
    assert ti.pose_training_blockers(_project(person=[_pose(3)]), schemas) == []


def test_pose_mixed_with_polygons_is_blocked():
    """YOLO-pose datasets cannot contain both, and finding that out deep inside
    Ultralytics is what the pre-flight exists to prevent."""
    schemas = {"person": {"names": ["a", "b", "c"], "skeleton": [], "flip_idx": []}}
    blockers = ti.pose_training_blockers(
        _project(cell=[_polygon()], person=[_pose(3)]), schemas
    )
    assert blockers and "cannot contain both" in blockers[0]


def test_mixed_k_pose_classes_are_blocked():
    """YOLO-pose carries one dataset-global kpt_shape."""
    project = {
        "a.png": {
            "person": [_pose(3, "person")],
            "hand": [_pose(5, "hand")],
        }
    }
    schemas = {
        "person": {"names": ["a", "b", "c"], "skeleton": [], "flip_idx": []},
        "hand": {"names": ["a", "b", "c", "d", "e"], "skeleton": [], "flip_idx": []},
    }
    blockers = ti.pose_training_blockers(project, schemas)
    assert blockers
    assert "one keypoint count" in blockers[0]
    assert "K=3" in blockers[0] and "K=5" in blockers[0]


def test_a_non_pose_project_has_no_pose_blockers():
    assert ti.pose_training_blockers(_project(cell=[_polygon()]), {}) == []


# --- stacks and videos ARE trainable ---------------------------------------
#
# The blocker here used to refuse every stack and video outright, citing a note
# that predated slice-aware export. A video's image_slices[base] is an ordinary
# LazySliceList and the exporters resolve slice pixels through it (#45/#47), so
# annotating 27 frames of a video produces a perfectly good dataset -- which
# training then declined to use.


def test_a_video_with_loaded_frames_does_not_block_training():
    images = [{"file_name": "clip.mp4", "is_multi_slice": True, "is_video": True}]
    assert ti.unresolvable_stack_blockers(images, loaded_stack_bases=["clip"]) == []


def test_an_annotated_stack_with_no_loaded_slices_blocks_training():
    """The genuine failure: annotations exist but there are no pixels to write,
    so the exporter would drop them with nothing but a log line -- training on
    less data than the user believes."""
    images = [
        {"file_name": "flat.png"},
        {"file_name": "stack.tif", "is_multi_slice": True},
    ]
    blockers = ti.unresolvable_stack_blockers(
        images, loaded_stack_bases=[], annotated_names=["stack_Z1", "flat.png"]
    )
    assert len(blockers) == 1
    assert "stack.tif" in blockers[0]
    assert "flat.png" not in blockers[0]


def test_an_unannotated_stack_does_not_block():
    """It contributes no keys to the export, so it cannot break anything.
    Refusing to train because a 4 GB CZI sits unopened in the project would be
    a refusal with no failure behind it."""
    images = [{"file_name": "big.czi", "is_multi_slice": True}]
    assert ti.unresolvable_stack_blockers(
        images, loaded_stack_bases=[], annotated_names=["flat.png"]
    ) == []


def test_plain_images_do_not_block():
    assert ti.unresolvable_stack_blockers([{"file_name": "a.png"}], []) == []
    assert ti.unresolvable_stack_blockers([], []) == []


def test_the_blocker_list_is_truncated_for_readability():
    images = [
        {"file_name": f"stack{i}.tif", "is_multi_slice": True} for i in range(12)
    ]
    blockers = ti.unresolvable_stack_blockers(
        images,
        loaded_stack_bases=[],
        annotated_names=[f"stack{i}_Z1" for i in range(12)],
    )
    assert "and 7 more" in blockers[0]


# --- what actually counts as an image --------------------------------------


def test_a_video_contributes_its_frames_not_itself():
    """Counting the parent entry reported a video with annotated frames as
    "368 annotation(s) across 0 of 1 image(s)" -- both halves wrong, and the
    second alarming enough to read as data loss."""
    images = [
        {"file_name": "clip.mp4", "is_multi_slice": True, "is_video": True},
        {"file_name": "flat.png"},
    ]
    names = ti.trainable_image_names(
        images, {"clip": ["clip_F00001", "clip_F00002", "clip_F00003"]}
    )
    assert names == ["clip_F00001", "clip_F00002", "clip_F00003", "flat.png"]


def test_a_stack_whose_slices_are_unknown_contributes_nothing():
    images = [{"file_name": "stack.tif", "is_multi_slice": True}]
    assert ti.trainable_image_names(images, {}) == []


def test_the_summary_counts_annotated_frames():
    """End to end for the Data row: frames annotated, frames total."""
    images = [{"file_name": "clip.mp4", "is_multi_slice": True, "is_video": True}]
    by_base = {"clip": [f"clip_F{i:05d}" for i in range(4)]}
    project = {
        "clip_F00000": {"bee": [_polygon()]},
        "clip_F00002": {"bee": [_polygon()]},
    }

    summary = ti.summarise_dataset(project, ti.trainable_image_names(images, by_base))

    assert summary["images"] == 4
    assert summary["annotated_images"] == 2
    assert summary["unlabelled_images"] == 2


# --- the dialog's derived config ------------------------------------------


class _FakeWindow(QWidget):
    """Just the three attributes the dialog reads. A QWidget because QDialog
    requires a real widget parent."""

    def __init__(self, annotations, images, schemas=None, image_slices=None):
        super().__init__()
        self.all_annotations = annotations
        self.all_images = images
        self.keypoint_schemas = schemas or {}
        self.image_slices = image_slices or {}


@pytest.fixture
def dialog_factory(qtbot):
    from src.digitalsreeni_image_annotator.dialogs.train_dialog import TrainDialog

    def _make(annotations, images, schemas=None, image_slices=None):
        window = _FakeWindow(annotations, images, schemas, image_slices)
        qtbot.addWidget(window)
        dialog = TrainDialog(window)
        qtbot.addWidget(dialog)
        return dialog

    return _make


def test_the_dialog_derives_the_task_rather_than_asking(dialog_factory):
    dialog = dialog_factory(_project(cell=[_polygon()]), [{"file_name": "img.png"}])
    assert dialog.task == ti.TASK_SEGMENT
    assert "segment" in dialog.task_label.text()


def test_the_base_model_list_matches_the_derived_task(dialog_factory):
    """A detect checkpoint on a segment dataset fails deep inside Ultralytics;
    offering only matching bases is the cheap half of preventing that."""
    dialog = dialog_factory(_project(cell=[_polygon()]), [{"file_name": "img.png"}])
    offered = [dialog.base_combo.itemText(i) for i in range(dialog.base_combo.count())]
    assert all("-seg" in name for name in offered)


def test_the_dialog_refuses_a_mixed_k_pose_project(dialog_factory):
    project = {
        "a.png": {"person": [_pose(3, "person")], "hand": [_pose(5, "hand")]}
    }
    schemas = {
        "person": {"names": ["a", "b", "c"], "skeleton": [], "flip_idx": []},
        "hand": {"names": list("abcde"), "skeleton": [], "flip_idx": []},
    }
    dialog = dialog_factory(project, [{"file_name": "a.png"}], schemas)
    assert dialog.train_button.isEnabled() is False
    assert "one keypoint count" in dialog.blocker_label.text()


def test_advanced_settings_apply_whether_or_not_they_are_expanded(dialog_factory):
    """Collapsed is a *disclosure*, not an off switch.

    This was a checkable QGroupBox, and Qt disables a checkable group's
    children when it is unchecked -- so the settings looked switched off while
    get_config sent them anyway. Early stopping appeared disabled and ran.
    """
    dialog = dialog_factory(_project(cell=[_polygon()]), [{"file_name": "i.png"}])

    assert dialog.advanced_toggle.isChecked() is False
    # isVisibleTo, not isVisible: the dialog is never shown headlessly, so
    # isVisible() is False for everything and would assert nothing.
    assert dialog.advanced_box.isVisibleTo(dialog) is False
    collapsed = dialog.get_config()

    dialog.advanced_toggle.setChecked(True)
    assert dialog.advanced_box.isVisibleTo(dialog) is True

    assert dialog.get_config() == collapsed
    assert collapsed["patience"] == 20, "early stopping is on while collapsed"
    assert collapsed["cos_lr"] is True


def test_an_edited_advanced_value_survives_re_collapsing(dialog_factory):
    dialog = dialog_factory(_project(cell=[_polygon()]), [{"file_name": "i.png"}])

    dialog.advanced_toggle.setChecked(True)
    dialog.patience_spin.setValue(5)
    dialog.advanced_toggle.setChecked(False)

    assert dialog.get_config()["patience"] == 5


def test_the_dialog_refuses_an_annotated_stack_with_no_loaded_slices(dialog_factory):
    dialog = dialog_factory(
        {"stack_Z1": {"cell": [_polygon()]}},
        [{"file_name": "stack.tif", "is_multi_slice": True}],
    )
    assert dialog.train_button.isEnabled() is False
    assert "stack.tif" in dialog.blocker_label.text()


def test_the_dialog_allows_training_on_annotated_video_frames(dialog_factory):
    """The regression: 368 polygons across a video's frames, and Train was
    disabled with a message claiming videos cannot be used."""
    dialog = dialog_factory(
        {"clip_F00001": {"bee": [_polygon()]}},
        [{"file_name": "clip.mp4", "is_multi_slice": True, "is_video": True}],
        image_slices={"clip": [("clip_F00001", None), ("clip_F00002", None)]},
    )
    assert dialog.train_button.isEnabled() is True
    assert dialog.blocker_label.isVisible() is False
    assert "1 of 2 image(s)" in dialog.data_label.text()


def test_switching_to_sam_lifts_the_yolo_only_blockers(dialog_factory):
    """SAM fine-tuning has neither the pose nor the multi-dimensional
    constraint, so the dialog must not carry YOLO's refusals across."""
    dialog = dialog_factory(
        {"stack_Z1": {"cell": [_polygon()]}},
        [{"file_name": "stack.tif", "is_multi_slice": True}],
    )
    assert dialog.train_button.isEnabled() is False

    dialog.sam_radio.setChecked(True)
    assert dialog.blockers() == []
    assert dialog.train_button.isEnabled() is True


def test_yolo_only_fields_hide_for_sam(dialog_factory):
    """Hidden rather than greyed out: a permanently-disabled control invites
    the user to wonder what would enable it.

    Hiding matters more than cosmetics here — SAM fine-tuning collects its real
    settings in SAMTrainConfigDialog, so leaving these visible would gather
    values this dialog then discards.
    """
    dialog = dialog_factory(_project(cell=[_polygon()]), [{"file_name": "img.png"}])
    # isVisibleTo, not isVisible: the dialog is never shown in a headless test,
    # so isVisible would be False for everything and assert nothing.
    assert dialog.imgsz_spin.isVisibleTo(dialog) is True
    dialog.sam_radio.setChecked(True)
    assert dialog.imgsz_spin.isVisibleTo(dialog) is False
    assert dialog.split_row_widget.isVisibleTo(dialog) is False
    assert dialog.base_row_widget.isVisibleTo(dialog) is False
    assert dialog.advanced_box.isVisibleTo(dialog) is False
    assert dialog.advanced_toggle.isVisibleTo(dialog) is False
    # ...and expanding it in SAM mode must not surface YOLO-only controls.
    dialog.advanced_toggle.setChecked(True)
    assert dialog.advanced_box.isVisibleTo(dialog) is False


def test_hiding_a_row_hides_its_label_too(dialog_factory):
    """setVisible on a QFormLayout field leaves the caption behind, so SAM mode
    showed orphan "Base" / "Val split" / "Run" labels next to blank space."""
    dialog = dialog_factory(_project(cell=[_polygon()]), [{"file_name": "img.png"}])
    dialog.sam_radio.setChecked(True)

    for field in (
        dialog.base_row_widget, dialog.split_row_widget, dialog.run_row_widget
    ):
        label = dialog.form.labelForField(field)
        assert label is None or label.isVisibleTo(dialog) is False


def test_a_browsed_base_is_dropped_when_a_stock_model_is_picked(dialog_factory):
    """Otherwise the browse wins forever and the combo selection is ignored."""
    dialog = dialog_factory(_project(cell=[_polygon()]), [{"file_name": "img.png"}])
    dialog.base_combo.insertItem(0, "/tmp/custom.pt")
    dialog.base_combo.setCurrentIndex(0)
    dialog.custom_base_path = "/tmp/custom.pt"
    assert dialog.get_config()["base_model"] == "/tmp/custom.pt"

    dialog.base_combo.setCurrentIndex(1)
    assert dialog.custom_base_path is None
    assert dialog.get_config()["base_model"] == dialog.base_combo.currentText()


def test_the_config_carries_the_advanced_values(dialog_factory):
    """build_yolo_train_opts must still receive them (ADR-028)."""
    dialog = dialog_factory(_project(cell=[_polygon()]), [{"file_name": "img.png"}])
    dialog.epochs_spin.setValue(42)
    dialog.lr0_spin.setValue(0.005)
    dialog.patience_spin.setValue(7)
    dialog.cos_lr_check.setChecked(False)
    dialog.split_slider.setValue(30)

    config = dialog.get_config()
    assert config["epochs"] == 42
    assert config["lr0"] == pytest.approx(0.005)
    assert config["patience"] == 7
    assert config["cos_lr"] is False
    assert config["val_split"] == 30
    assert config["task"] == ti.TASK_SEGMENT


def test_the_data_summary_warns_about_unlabelled_images(dialog_factory):
    project = {"a.png": {"cell": [_polygon()]}}
    images = [{"file_name": f"{c}.png"} for c in "abcdef"]
    dialog = dialog_factory(project, images)
    assert "5 image(s) have no labels" in dialog.data_label.text()
    assert "large share" in dialog.data_label.text()
