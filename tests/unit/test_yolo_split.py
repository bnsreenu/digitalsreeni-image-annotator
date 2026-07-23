"""
Unit tests for the deterministic YOLO train/val split helper (issue #83).

`assign_train_val` partitions annotated image names into train/val sets using a
stable filename hash, so the split is reproducible and the val set is never
accidentally empty when a split is requested.
"""

from src.digitalsreeni_image_annotator.io.export_formats import assign_train_val


def _names(n):
    return [f"img_{i:03d}.png" for i in range(n)]


def test_zero_split_keeps_everything_in_train():
    names = _names(10)
    train, val = assign_train_val(names, 0)
    assert train == set(names)
    assert val == set()


def test_negative_split_treated_as_zero():
    names = _names(5)
    train, val = assign_train_val(names, -5)
    assert train == set(names)
    assert val == set()


def test_single_image_never_emptied_into_val():
    # With one image a split would leave train empty; keep it in train.
    train, val = assign_train_val(["only.png"], 50)
    assert train == {"only.png"}
    assert val == set()


def test_split_is_deterministic_across_calls():
    names = _names(50)
    a_train, a_val = assign_train_val(names, 20)
    b_train, b_val = assign_train_val(list(reversed(names)), 20)
    # Order of the input must not change the partition.
    assert a_val == b_val
    assert a_train == b_train


def test_split_count_is_proportional():
    names = _names(100)
    _, val = assign_train_val(names, 20)
    assert len(val) == 20


def test_split_rounds_to_nearest():
    names = _names(10)
    _, val = assign_train_val(names, 25)  # 2.5 -> 2
    assert len(val) == 2


def test_val_non_empty_for_small_set_with_split():
    # Two images, 20% rounds to 0 but the val set must stay non-empty.
    train, val = assign_train_val(_names(2), 20)
    assert len(val) == 1
    assert len(train) == 1


def test_train_never_emptied_by_high_split():
    # 100% would drain train; at least one image must remain.
    train, val = assign_train_val(_names(4), 100)
    assert len(train) == 1
    assert len(val) == 3


def test_train_and_val_are_disjoint_and_cover_all():
    names = _names(37)
    train, val = assign_train_val(names, 30)
    assert train.isdisjoint(val)
    assert train | val == set(names)
