"""
Unit tests for the SAM fine-tuner's deterministic per-image train/val split
(issue #85).

`split_groups` partitions a list of `SampleGroup`s into (train, val), reusing
the YOLO export's stable-hash `assign_train_val` so the SAM split is
reproducible. 100% train (or a single image) yields an empty val set.
"""

from src.digitalsreeni_image_annotator.training.sam_dataset import split_groups
from src.digitalsreeni_image_annotator.training.sam_trainer import SampleGroup


def _groups(n):
    return [
        SampleGroup(lambda: None, [{"bbox": [0, 0, 1, 1]}], name=f"img{i}.png")
        for i in range(n)
    ]


def test_split_80_20_counts():
    train, val = split_groups(_groups(10), 80)
    assert len(train) == 8 and len(val) == 2


def test_split_is_deterministic_across_calls():
    groups = _groups(20)
    a = [g.name for g in split_groups(groups, 75)[1]]
    b = [g.name for g in split_groups(groups, 75)[1]]
    assert a == b


def test_100_pct_has_no_val():
    train, val = split_groups(_groups(10), 100)
    assert len(train) == 10 and val == []


def test_single_group_is_all_train():
    train, val = split_groups(_groups(1), 80)
    assert len(train) == 1 and val == []


def test_split_is_disjoint_and_complete():
    groups = _groups(13)
    train, val = split_groups(groups, 70)
    names = {g.name for g in train} | {g.name for g in val}
    assert len(names) == 13 and len(train) + len(val) == 13
