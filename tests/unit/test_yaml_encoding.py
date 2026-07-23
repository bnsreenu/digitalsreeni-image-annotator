"""
Regression guard for UTF-8 file encoding (upstream issue #44).

Reading a COCO JSON that contains non-ASCII category names must succeed
regardless of the platform's default code page. Before the fix, open()
without encoding used cp1252 on Windows and crashed on these bytes. The
test writes genuine non-ASCII bytes (ensure_ascii=False) and asserts the
unicode survives the round-trip through import_coco_json's open().
"""

import json

from src.digitalsreeni_image_annotator.io.import_formats import import_coco_json

UNICODE_CLASS = "Zellkörper-Ü-中"  # German + a CJK char


def test_import_coco_json_preserves_unicode_class(tmp_path):
    coco = {
        "images": [{"id": 1, "file_name": "img.png", "width": 10, "height": 10}],
        "categories": [{"id": 1, "name": UNICODE_CLASS}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 5, 5]}
        ],
    }
    json_path = tmp_path / "annotations.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False)

    imported_annotations, image_info, keypoint_schemas = import_coco_json(str(json_path), {})

    assert UNICODE_CLASS in imported_annotations["img.png"]
    assert image_info[1]["file_name"] == "img.png"
