"""Unit tests for the keypoint/pose schema helpers (issue #35).

``sanitize_schema`` validates + normalizes a (file- or user-supplied) schema,
``make_schema`` builds one from parts, and both must survive the ``.iap``
serialization path (``convert_to_serializable`` -> json -> back) unchanged.
"""

import json

from src.digitalsreeni_image_annotator.core.image_utils import convert_to_serializable
from src.digitalsreeni_image_annotator.core.keypoint_schema import (
    is_involution,
    make_schema,
    sanitize_schema,
    schema_k,
)


def test_make_schema_defaults_flip_to_identity():
    schema = make_schema(["nose", "left_eye", "right_eye"])
    assert schema["names"] == ["nose", "left_eye", "right_eye"]
    assert schema["skeleton"] == []
    assert schema["flip_idx"] == [0, 1, 2]
    assert schema_k(schema) == 3


def test_make_schema_keeps_valid_skeleton_and_flip():
    schema = make_schema(
        ["l_eye", "r_eye"], skeleton=[[0, 1]], flip_idx=[1, 0]
    )
    assert schema["skeleton"] == [[0, 1]]
    assert schema["flip_idx"] == [1, 0]


def test_sanitize_drops_out_of_range_and_self_edges():
    schema = sanitize_schema(
        {"names": ["a", "b"], "skeleton": [[0, 1], [0, 5], [1, 1], "x"], "flip_idx": [0, 1]}
    )
    assert schema["skeleton"] == [[0, 1]]  # [0,5] out of range, [1,1] self, "x" malformed


def test_sanitize_bad_flip_falls_back_to_identity():
    # Wrong length / not a permutation -> identity.
    assert sanitize_schema({"names": ["a", "b", "c"], "flip_idx": [9, 9, 9]})["flip_idx"] == [0, 1, 2]
    assert sanitize_schema({"names": ["a", "b"], "flip_idx": [0]})["flip_idx"] == [0, 1]


def test_sanitize_non_involution_permutation_falls_back_to_identity():
    # [1, 2, 0] is a valid bijection (a 3-cycle) but not self-inverse — flipping
    # point 0 twice would land on point 2, not back on 0. Must fall back too.
    assert sanitize_schema({"names": ["a", "b", "c"], "flip_idx": [1, 2, 0]})["flip_idx"] == [0, 1, 2]


def test_is_involution():
    assert is_involution([0, 1, 2]) is True   # identity
    assert is_involution([1, 0, 2]) is True   # one swap, one fixed point
    assert is_involution([1, 2, 0]) is False  # 3-cycle: valid permutation, not self-inverse
    assert is_involution([0, 0]) is False     # not even a permutation (duplicate target)


def test_sanitize_rejects_empty_or_duplicate_names():
    assert sanitize_schema({"names": []}) is None
    assert sanitize_schema({"names": ["a", "a"]}) is None
    assert sanitize_schema({"names": ["a", "  "]}) is None
    assert sanitize_schema(None) is None
    assert sanitize_schema("not a dict") is None


def test_schema_survives_iap_serialization_roundtrip():
    schema = make_schema(["nose", "l", "r"], skeleton=[[0, 1], [0, 2]], flip_idx=[0, 2, 1])
    restored = json.loads(json.dumps(convert_to_serializable({"keypoint_schema": schema})))
    assert restored["keypoint_schema"] == schema
    # And it re-sanitizes to itself (idempotent).
    assert sanitize_schema(restored["keypoint_schema"]) == schema
