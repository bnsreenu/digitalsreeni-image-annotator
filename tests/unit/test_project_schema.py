"""Unit tests for load-time `.iap` structural validation (#42).

validate_project_data answers only "is this shaped enough to load without a
mid-load crash?" — leniently, and it must never reject unknown keys (the format
grows: keypoint schemas, DINO config, relative paths).
"""

from digitalsreeni_image_annotator.core.project_schema import validate_project_data


def test_valid_minimal_returns_empty():
    assert validate_project_data({"images": []}) == []


def test_valid_full_returns_empty():
    data = {
        "images": [{"file_name": "a.png"}],
        "classes": [{"name": "cell", "color": "#ff0000"}],
        "image_paths": {"a.png": "/abs/a.png"},
        "image_paths_rel": {"a.png": "images/a.png"},
        "notes": "hello",
    }
    assert validate_project_data(data) == []


def test_non_dict_top_level_is_rejected():
    assert validate_project_data([1, 2, 3])


def test_images_not_a_list_is_rejected():
    assert any("images" in p for p in validate_project_data({"images": "nope"}))


def test_image_entry_without_file_name_is_rejected():
    problems = validate_project_data({"images": [{"width": 10}]})
    assert any("file_name" in p for p in problems)


def test_class_missing_color_is_rejected():
    data = {"images": [], "classes": [{"name": "cell"}]}
    assert any("color" in p for p in validate_project_data(data))


def test_non_str_path_value_is_rejected():
    data = {"images": [], "image_paths": {"a.png": 123}}
    assert any("image_paths" in p for p in validate_project_data(data))


def test_notes_wrong_type_is_rejected():
    assert any("notes" in p for p in validate_project_data({"images": [], "notes": 5}))


def test_unknown_keys_are_allowed():
    data = {"images": [], "keypoint_schema": {"x": 1}, "dino_config": {"phrases": {}}}
    assert validate_project_data(data) == []
