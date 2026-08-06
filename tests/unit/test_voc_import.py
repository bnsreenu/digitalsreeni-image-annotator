"""Pascal VOC annotation import (issue #75).

Closes a plain asymmetry in ``io/``: the app has exported VOC for a long time
but could never read its own output back, let alone the large amount of
VOC-format data in the wild.

The two tests that matter most are the round-trip (export then import must
produce an equivalent project — the thing the asymmetry made impossible) and
the substring-collision one: ``"bee.jpg" in "honeybee.jpg"`` is True, and
substring-only filename matching attaches annotations to the wrong image.
"""

import os
import xml.etree.ElementTree as ET

import pytest

from src.digitalsreeni_image_annotator.io.import_formats import (
    import_pascal_voc,
    process_import_format,
)


def _write_voc(directory, file_name, objects, size=(200, 200)):
    """Write one VOC XML in the layout export_pascal_voc_bbox produces."""
    annotations_dir = os.path.join(directory, "Annotations")
    os.makedirs(annotations_dir, exist_ok=True)

    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text = "images"
    ET.SubElement(root, "filename").text = file_name
    node = ET.SubElement(root, "size")
    ET.SubElement(node, "width").text = str(size[0])
    ET.SubElement(node, "height").text = str(size[1])
    ET.SubElement(node, "depth").text = "3"
    ET.SubElement(root, "segmented").text = "0"

    for class_name, (xmin, ymin, xmax, ymax) in objects:
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = class_name
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "1"
        box = ET.SubElement(obj, "bndbox")
        ET.SubElement(box, "xmin").text = str(xmin)
        ET.SubElement(box, "ymin").text = str(ymin)
        ET.SubElement(box, "xmax").text = str(xmax)
        ET.SubElement(box, "ymax").text = str(ymax)

    stem = os.path.splitext(file_name)[0]
    ET.ElementTree(root).write(
        os.path.join(annotations_dir, f"{stem}.xml"), encoding="utf-8"
    )
    return annotations_dir


# --- coordinate conversion -------------------------------------------------


def test_corners_convert_to_x_y_width_height(tmp_path):
    """VOC stores xmin/ymin/xmax/ymax; the app stores [x, y, w, h]. Convert,
    never copy."""
    _write_voc(str(tmp_path), "a.png", [("cell", (10, 20, 60, 90))])
    annotations, _info, _schemas = import_pascal_voc(str(tmp_path), {})

    bbox = annotations["a.png"]["cell"][0]["bbox"]
    assert bbox == [10.0, 20.0, 50.0, 70.0]


def test_out_of_bounds_coordinates_are_clamped(tmp_path):
    """Producers disagree on 0- vs 1-based indexing, so the file is not
    trusted (ADR-024)."""
    _write_voc(str(tmp_path), "a.png", [("cell", (-10, -10, 500, 500))],
               size=(200, 200))
    annotations, _info, _schemas = import_pascal_voc(str(tmp_path), {})

    x, y, w, h = annotations["a.png"]["cell"][0]["bbox"]
    assert x >= 0 and y >= 0
    assert x + w <= 200 and y + h <= 200


def test_a_zero_area_box_does_not_go_negative(tmp_path):
    _write_voc(str(tmp_path), "a.png", [("cell", (50, 50, 40, 40))])
    annotations, _info, _schemas = import_pascal_voc(str(tmp_path), {})
    _x, _y, w, h = annotations["a.png"]["cell"][0]["bbox"]
    assert w >= 0 and h >= 0


# --- shape of the result ---------------------------------------------------


def test_the_importer_returns_the_uniform_triple(tmp_path):
    """Every io.import_formats entry point returns
    ``(annotations, image_info, schemas)``. Breaking that breaks the caller."""
    _write_voc(str(tmp_path), "a.png", [("cell", (10, 10, 50, 50))])
    result = import_pascal_voc(str(tmp_path), {})

    assert isinstance(result, tuple) and len(result) == 3
    annotations, image_info, schemas = result
    assert isinstance(annotations, dict)
    assert isinstance(image_info, dict)
    assert schemas == {}, "VOC has no keypoint concept"


def test_image_info_carries_the_file_name_the_caller_reads(tmp_path):
    """io_controller iterates image_info.values() and reads info['file_name']."""
    _write_voc(str(tmp_path), "a.png", [("cell", (10, 10, 50, 50))], size=(320, 240))
    _annotations, image_info, _schemas = import_pascal_voc(str(tmp_path), {})

    info = image_info["a.png"]
    assert info["file_name"] == "a.png"
    assert (info["width"], info["height"]) == (320, 240)


def test_a_plain_annotation_never_carries_a_none_segmentation(tmp_path):
    """Several existence-only ``"segmentation" in ann`` checks are not
    None-guarded and would misfire on a None-valued key."""
    _write_voc(str(tmp_path), "a.png", [("cell", (10, 10, 50, 50))])
    annotations, _info, _schemas = import_pascal_voc(str(tmp_path), {})

    annotation = annotations["a.png"]["cell"][0]
    assert "segmentation" not in annotation or annotation["segmentation"]


def test_difficult_and_truncated_are_ignored_not_invented_into_fields(tmp_path):
    _write_voc(str(tmp_path), "a.png", [("cell", (10, 10, 50, 50))])
    annotations, _info, _schemas = import_pascal_voc(str(tmp_path), {})
    annotation = annotations["a.png"]["cell"][0]
    assert "difficult" not in annotation
    assert "truncated" not in annotation


# --- classes ---------------------------------------------------------------


def test_unknown_classes_are_created_with_fresh_ids(tmp_path):
    _write_voc(
        str(tmp_path), "a.png",
        [("cell", (10, 10, 50, 50)), ("nucleus", (60, 60, 90, 90))],
    )
    annotations, _info, _schemas = import_pascal_voc(str(tmp_path), {"cell": 1})

    assert annotations["a.png"]["cell"][0]["category_id"] == 1
    assert annotations["a.png"]["nucleus"][0]["category_id"] == 2


def test_an_object_without_a_name_is_skipped(tmp_path):
    annotations_dir = _write_voc(str(tmp_path), "a.png", [("cell", (10, 10, 50, 50))])
    tree = ET.parse(os.path.join(annotations_dir, "a.xml"))
    obj = ET.SubElement(tree.getroot(), "object")
    ET.SubElement(obj, "name").text = ""
    box = ET.SubElement(obj, "bndbox")
    for tag, value in (("xmin", 1), ("ymin", 1), ("xmax", 5), ("ymax", 5)):
        ET.SubElement(box, tag).text = str(value)
    tree.write(os.path.join(annotations_dir, "a.xml"), encoding="utf-8")

    annotations, _info, _schemas = import_pascal_voc(str(tmp_path), {})
    assert list(annotations["a.png"]) == ["cell"]


def test_an_object_with_neither_a_bndbox_nor_a_polygon_is_skipped(tmp_path):
    """No geometry at all is nothing to import. An object with only a polygon
    is a different case and must NOT be skipped — see
    test_an_object_with_a_polygon_and_no_bndbox_still_imports."""
    annotations_dir = _write_voc(str(tmp_path), "a.png", [("cell", (10, 10, 50, 50))])
    tree = ET.parse(os.path.join(annotations_dir, "a.xml"))
    obj = ET.SubElement(tree.getroot(), "object")
    ET.SubElement(obj, "name").text = "ghost"
    tree.write(os.path.join(annotations_dir, "a.xml"), encoding="utf-8")

    annotations, _info, _schemas = import_pascal_voc(str(tmp_path), {})
    assert "ghost" not in annotations["a.png"]


# --- directory layouts a user actually picks -------------------------------


def test_the_dataset_root_can_be_selected(tmp_path):
    _write_voc(str(tmp_path), "a.png", [("cell", (10, 10, 50, 50))])
    annotations, _info, _schemas = import_pascal_voc(str(tmp_path), {})
    assert "a.png" in annotations


def test_the_annotations_directory_can_be_selected(tmp_path):
    annotations_dir = _write_voc(str(tmp_path), "a.png", [("cell", (10, 10, 50, 50))])
    annotations, _info, _schemas = import_pascal_voc(annotations_dir, {})
    assert "a.png" in annotations


def test_a_directory_with_no_xml_is_an_error(tmp_path):
    os.makedirs(os.path.join(str(tmp_path), "Annotations"), exist_ok=True)
    with pytest.raises(ValueError, match="No .xml"):
        import_pascal_voc(str(tmp_path), {})


def test_a_file_path_is_rejected(tmp_path):
    path = tmp_path / "not-a-dir.xml"
    path.write_text("<annotation/>", encoding="utf-8")
    with pytest.raises(ValueError, match="not a directory"):
        import_pascal_voc(str(path), {})


def test_malformed_xml_refuses_rather_than_importing_partial_data(tmp_path):
    """A half-imported project is harder to recover from than a refused
    import."""
    annotations_dir = os.path.join(str(tmp_path), "Annotations")
    os.makedirs(annotations_dir, exist_ok=True)
    with open(os.path.join(annotations_dir, "broken.xml"), "w", encoding="utf-8") as f:
        f.write("<annotation><object>")

    with pytest.raises(ValueError, match="Malformed"):
        import_pascal_voc(str(tmp_path), {})


# --- dispatcher ------------------------------------------------------------


def test_the_format_is_registered_with_the_dispatcher(tmp_path):
    """The selector, process_import_format and io_controller all have to know
    the new value; this covers the dispatcher leg."""
    _write_voc(str(tmp_path), "a.png", [("cell", (10, 10, 50, 50))])
    annotations, _info, schemas = process_import_format(
        "Pascal VOC", str(tmp_path), {}
    )
    assert "a.png" in annotations
    assert schemas == {}


# --- filename matching -----------------------------------------------------


def test_similar_filenames_do_not_collide(tmp_path):
    '''"bee.jpg" in "honeybee.jpg" is True; substring-only matching would
    attach these annotations to the wrong image.'''
    _write_voc(str(tmp_path), "bee.jpg", [("bee", (10, 10, 20, 20))])
    _write_voc(str(tmp_path), "honeybee.jpg", [("honeybee", (30, 30, 40, 40))])

    annotations, image_info, _schemas = import_pascal_voc(str(tmp_path), {})

    assert set(annotations) == {"bee.jpg", "honeybee.jpg"}
    assert list(annotations["bee.jpg"]) == ["bee"]
    assert list(annotations["honeybee.jpg"]) == ["honeybee"]
    assert image_info["bee.jpg"]["file_name"] == "bee.jpg"


# --- round trip ------------------------------------------------------------


def _drawn_polygon(outline, name="cell", number=1):
    """An annotation in the shape the app ACTUALLY produces for a drawn mask.

    **No ``bbox`` key.** Every drawing path — polygon, rectangle, paint,
    eraser, SAM accept, DINO accept — emits segmentation-only;
    ``edit_gestures.sync_bbox_key`` states the rule outright ("drawn shapes
    have no bbox key"). Seeding a bbox in a round-trip fixture certifies a
    shape the app never creates, which is exactly how the missing-bndbox bug
    survived its own test.
    """
    return {
        "segmentation": list(outline),
        "category_id": 1,
        "category_name": name,
        "number": number,
    }


def test_a_drawn_polygon_survives_the_voc_round_trip(tmp_path):
    """The regression that the first version of this importer shipped.

    ``export_pascal_voc_both`` writes the outline inline, but emitted no
    ``<bndbox>`` for a segmentation-only annotation — and the importer required
    one. Exporting a drawn polygon and re-importing it returned an EMPTY
    project while the UI reported "imported successfully". Total silent loss.
    """
    from PIL import Image

    from src.digitalsreeni_image_annotator.io.export_formats import (
        export_pascal_voc_both,
    )

    source = tmp_path / "source"
    source.mkdir()
    Image.new("RGB", (200, 200), (64, 64, 64)).save(source / "a.png")

    outline = [10.0, 10.0, 60.0, 10.0, 60.0, 80.0, 10.0, 80.0]
    all_annotations = {"a.png": {"cell": [_drawn_polygon(outline)]}}
    out_dir = str(tmp_path / "voc_seg")
    os.makedirs(out_dir, exist_ok=True)
    export_pascal_voc_both(
        all_annotations, {"cell": 1}, {"a.png": str(source / "a.png")},
        [], {}, out_dir,
    )

    annotations, _info, _schemas = import_pascal_voc(out_dir, {})

    assert annotations.get("a.png"), "the whole image came back empty"
    imported = annotations["a.png"]["cell"][0]
    assert imported.get("segmentation"), "the outline was dropped on import"
    assert imported["segmentation"] == pytest.approx(outline)
    assert imported["type"] == "polygon"
    # The box is derived from the outline, since VOC needs one and the drawn
    # annotation had none.
    assert imported["bbox"] == pytest.approx([10.0, 10.0, 50.0, 70.0])


def test_the_bbox_exporter_also_emits_a_box_for_a_drawn_polygon(tmp_path):
    """VOC without a bndbox is not VOC — a foreign consumer would read nothing
    either, so the fix belongs in both exporters, not just the importer."""
    from PIL import Image

    from src.digitalsreeni_image_annotator.io.export_formats import (
        export_pascal_voc_bbox,
    )

    source = tmp_path / "source"
    source.mkdir()
    Image.new("RGB", (200, 200), (64, 64, 64)).save(source / "a.png")

    outline = [10.0, 10.0, 60.0, 10.0, 60.0, 80.0, 10.0, 80.0]
    out_dir = str(tmp_path / "voc")
    os.makedirs(out_dir, exist_ok=True)
    export_pascal_voc_bbox(
        {"a.png": {"cell": [_drawn_polygon(outline)]}},
        {"cell": 1}, {"a.png": str(source / "a.png")}, [], {}, out_dir,
    )

    xml = ET.parse(os.path.join(out_dir, "Annotations", "a.xml"))
    box = xml.getroot().find("object/bndbox")
    assert box is not None, "a drawn polygon exported with no geometry at all"
    assert box.findtext("xmin") == "10"
    assert box.findtext("xmax") == "60"


def test_an_object_with_a_polygon_and_no_bndbox_still_imports(tmp_path):
    """Directly, without going through the exporter — foreign VOC producers
    write this shape too."""
    annotations_dir = os.path.join(str(tmp_path), "Annotations")
    os.makedirs(annotations_dir, exist_ok=True)

    root = ET.Element("annotation")
    ET.SubElement(root, "filename").text = "a.png"
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = "200"
    ET.SubElement(size, "height").text = "200"
    obj = ET.SubElement(root, "object")
    ET.SubElement(obj, "name").text = "cell"
    segmentation = ET.SubElement(obj, "segmentation")
    polygon = ET.SubElement(segmentation, "polygon")
    for index, (x, y) in enumerate([(10, 10), (60, 10), (60, 80)], start=1):
        point = ET.SubElement(polygon, f"pt{index}")
        ET.SubElement(point, "x").text = str(x)
        ET.SubElement(point, "y").text = str(y)
    ET.ElementTree(root).write(
        os.path.join(annotations_dir, "a.xml"), encoding="utf-8"
    )

    annotations, _info, _schemas = import_pascal_voc(str(tmp_path), {})
    imported = annotations["a.png"]["cell"][0]
    assert imported["segmentation"] == pytest.approx([10, 10, 60, 10, 60, 80])
    assert imported["bbox"] == pytest.approx([10, 10, 50, 70])


def test_an_object_without_an_inline_polygon_stays_a_box(tmp_path):
    _write_voc(str(tmp_path), "a.png", [("cell", (10, 10, 50, 50))])
    annotations, _info, _schemas = import_pascal_voc(str(tmp_path), {})
    annotation = annotations["a.png"]["cell"][0]
    assert "segmentation" not in annotation
    assert annotation["type"] == "rectangle"


def test_each_object_gets_its_own_outline(tmp_path):
    """Inline polygons belong to their object, so two annotations of the same
    class need no pairing heuristic — which the mask-based version got wrong."""
    annotations_dir = _write_voc(
        str(tmp_path), "a.png",
        [("cell", (10, 10, 50, 50)), ("cell", (100, 100, 150, 150))],
    )
    tree = ET.parse(os.path.join(annotations_dir, "a.xml"))
    for obj, ring in zip(
        tree.getroot().findall("object"),
        ([10, 10, 50, 10, 50, 50], [100, 100, 150, 100, 150, 150]),
    ):
        segmentation = ET.SubElement(obj, "segmentation")
        polygon = ET.SubElement(segmentation, "polygon")
        for index in range(0, len(ring), 2):
            point = ET.SubElement(polygon, f"pt{index // 2 + 1}")
            ET.SubElement(point, "x").text = str(ring[index])
            ET.SubElement(point, "y").text = str(ring[index + 1])
    tree.write(os.path.join(annotations_dir, "a.xml"), encoding="utf-8")

    annotations, _info, _schemas = import_pascal_voc(str(tmp_path), {})
    first, second = annotations["a.png"]["cell"]
    assert first["segmentation"][:2] == [10.0, 10.0]
    assert second["segmentation"][:2] == [100.0, 100.0]


def test_a_polygon_with_too_few_points_is_ignored(tmp_path):
    annotations_dir = _write_voc(str(tmp_path), "a.png", [("cell", (10, 10, 50, 50))])
    tree = ET.parse(os.path.join(annotations_dir, "a.xml"))
    obj = tree.getroot().find("object")
    segmentation = ET.SubElement(obj, "segmentation")
    polygon = ET.SubElement(segmentation, "polygon")
    point = ET.SubElement(polygon, "pt1")
    ET.SubElement(point, "x").text = "1"
    ET.SubElement(point, "y").text = "2"
    tree.write(os.path.join(annotations_dir, "a.xml"), encoding="utf-8")

    annotations, _info, _schemas = import_pascal_voc(str(tmp_path), {})
    assert "segmentation" not in annotations["a.png"]["cell"][0]


def test_export_then_import_produces_an_equivalent_project(tmp_path, qtbot):
    """The whole point of the issue: a VOC export from this app must re-import."""
    from PyQt6.QtGui import QColor, QImage

    from src.digitalsreeni_image_annotator.io.export_formats import (
        export_pascal_voc_bbox,
    )

    source = tmp_path / "source"
    source.mkdir()
    image_path = source / "a.png"
    image = QImage(200, 200, QImage.Format.Format_RGB888)
    image.fill(QColor("#404040"))
    image.save(str(image_path))

    all_annotations = {
        "a.png": {
            "cell": [
                {"bbox": [10.0, 20.0, 50.0, 70.0], "category_name": "cell",
                 "number": 1},
            ]
        }
    }
    out_dir = str(tmp_path / "voc")
    os.makedirs(out_dir, exist_ok=True)
    export_pascal_voc_bbox(
        all_annotations, {"cell": 1}, {"a.png": str(image_path)}, [], {}, out_dir
    )

    annotations, image_info, schemas = import_pascal_voc(out_dir, {})

    assert schemas == {}
    assert "a.png" in annotations
    roundtripped = annotations["a.png"]["cell"][0]["bbox"]
    assert roundtripped == pytest.approx([10.0, 20.0, 50.0, 70.0])
    assert image_info["a.png"]["width"] == 200
