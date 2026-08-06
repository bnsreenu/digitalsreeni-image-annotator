"""Rule-based annotation QC audit (issue #70).

**Every test here runs without a QApplication, and one of them proves it.**
That is the load-bearing constraint of the whole issue: the headless CLI's
``validate`` command (issue #76) imports these rules to run label quality as a
CI gate, and a stray Qt import anywhere in the chain would make label
validation require a display. A test is the only thing that keeps that honest,
because the import would work fine on a developer machine.
"""

import subprocess
import sys

import pytest

from src.digitalsreeni_image_annotator.core import annotation_qc as qc


def _square(x0, y0, side, name="cell", number=1):
    return {
        "segmentation": [x0, y0, x0 + side, y0, x0 + side, y0 + side, x0, y0 + side],
        "category_name": name,
        "number": number,
    }


def _pose(points, name="person", number=1, num_keypoints=None):
    flat = [c for p in points for c in p]
    labelled = [p for p in points if p[2] > 0]
    xs = [p[0] for p in labelled] or [0]
    ys = [p[1] for p in labelled] or [0]
    return {
        "keypoints": flat,
        "num_keypoints": len(labelled) if num_keypoints is None else num_keypoints,
        "bbox": [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)],
        "category_name": name,
        "number": number,
    }


def _project(annotations_by_class, image="img.png"):
    return {image: annotations_by_class}


def _rules(findings):
    return {f.rule for f in findings}


# --- the Qt-free guarantee -------------------------------------------------


def test_the_rule_engine_imports_without_qt():
    """The CLI's validate command depends on this. Run in a subprocess because
    the test session has already imported PyQt6 — checking sys.modules in-process
    would pass regardless of what this module does."""
    code = (
        "import sys;"
        "sys.path.insert(0, 'src');"
        "import digitalsreeni_image_annotator.core.annotation_qc as m;"
        "qt = [n for n in sys.modules if n.startswith('PyQt6')];"
        "assert not qt, qt;"
        "print('clean')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "clean" in result.stdout


# --- geometry --------------------------------------------------------------


def test_a_clean_project_reports_nothing():
    project = _project({"cell": [_square(10, 10, 40)]})
    findings = qc.run_audit(project, {"img.png": (200, 200)})
    assert findings == []


def test_self_intersecting_polygon_is_an_error():
    bowtie = {"segmentation": [0, 0, 40, 40, 40, 0, 0, 40],
              "category_name": "cell", "number": 1}
    findings = qc.run_audit(_project({"cell": [bowtie]}), {"img.png": (200, 200)})
    match = next(f for f in findings if f.rule == qc.RULE_SELF_INTERSECTING)
    assert match.severity == qc.SEVERITY_ERROR
    assert match.fixable is True


def test_fewer_than_three_vertices_is_an_error():
    stub = {"segmentation": [0, 0, 10, 10], "category_name": "cell", "number": 1}
    findings = qc.run_audit(_project({"cell": [stub]}), {"img.png": (200, 200)})
    assert qc.RULE_TOO_FEW_VERTICES in _rules(findings)


def test_degenerate_area_is_reported():
    sliver = {"segmentation": [0, 0, 100, 0, 100, 0.001],
              "category_name": "cell", "number": 1}
    findings = qc.run_audit(_project({"cell": [sliver]}), {"img.png": (200, 200)})
    assert qc.RULE_DEGENERATE_AREA in _rules(findings)


def test_out_of_bounds_coordinates_are_reported():
    findings = qc.run_audit(
        _project({"cell": [_square(150, 150, 100)]}), {"img.png": (200, 200)}
    )
    match = next(f for f in findings if f.rule == qc.RULE_OUT_OF_BOUNDS)
    assert match.fixable is True


def test_bounds_rules_are_skipped_when_the_size_is_unknown():
    """A caller that cannot resolve every size should still get the rest of the
    audit rather than a refusal."""
    findings = qc.run_audit(_project({"cell": [_square(150, 150, 100)]}), {})
    assert qc.RULE_OUT_OF_BOUNDS not in _rules(findings)


def test_bbox_disagreeing_with_the_outline_is_reported():
    annotation = _square(10, 10, 40)
    annotation["bbox"] = [0, 0, 5, 5]
    findings = qc.run_audit(
        _project({"cell": [annotation]}), {"img.png": (200, 200)}
    )
    match = next(f for f in findings if f.rule == qc.RULE_BBOX_MISMATCH)
    assert match.fixable is True
    assert match.detail["derived"] == [10, 10, 40, 40]


def test_a_correct_bbox_is_not_reported():
    annotation = _square(10, 10, 40)
    annotation["bbox"] = [10, 10, 40, 40]
    findings = qc.run_audit(
        _project({"cell": [annotation]}), {"img.png": (200, 200)}
    )
    assert qc.RULE_BBOX_MISMATCH not in _rules(findings)


# --- pose rules stay separate from polygon rules ---------------------------


def test_a_pose_instance_never_triggers_polygon_rules():
    """Polygon validity and area are meaningless for a keypoint instance; the
    absence of a segmentation key is what routes it (ADR-029)."""
    instance = _pose([(10, 10, 2), (30, 30, 2), (50, 10, 2)])
    findings = qc.run_audit(
        _project({"person": [instance]}), {"img.png": (200, 200)}
    )
    polygon_rules = {
        qc.RULE_SELF_INTERSECTING,
        qc.RULE_TOO_FEW_VERTICES,
        qc.RULE_DEGENERATE_AREA,
    }
    assert not (_rules(findings) & polygon_rules)


def test_num_keypoints_disagreeing_with_the_flags_is_reported():
    instance = _pose([(10, 10, 2), (30, 30, 0)], num_keypoints=2)
    findings = qc.run_audit(
        _project({"person": [instance]}), {"img.png": (200, 200)}
    )
    match = next(f for f in findings if f.rule == qc.RULE_POSE_COUNT_MISMATCH)
    assert match.fixable is True


def test_a_keypoint_outside_its_instance_box_is_reported():
    instance = _pose([(10, 10, 2), (30, 30, 2)])
    instance["bbox"] = [0, 0, 5, 5]
    findings = qc.run_audit(
        _project({"person": [instance]}), {"img.png": (200, 200)}
    )
    match = next(f for f in findings if f.rule == qc.RULE_POSE_POINT_OUTSIDE_BBOX)
    assert match.detail["indices"] == [0, 1]


def test_unlabelled_keypoints_do_not_count_as_out_of_bounds():
    """v=0 points are padding pinned at (0, 0), not real coordinates."""
    instance = _pose([(10, 10, 2), (0, 0, 0)])
    findings = qc.run_audit(
        _project({"person": [instance]}), {"img.png": (200, 200)}
    )
    assert qc.RULE_OUT_OF_BOUNDS not in _rules(findings)


# --- redundancy ------------------------------------------------------------


def test_near_duplicates_within_a_class_are_reported():
    project = _project({
        "cell": [_square(10, 10, 40, number=1), _square(11, 10, 40, number=2)]
    })
    findings = qc.run_audit(project, {"img.png": (200, 200)})
    match = next(f for f in findings if f.rule == qc.RULE_NEAR_DUPLICATE)
    assert match.detail["other"] == 2


def test_two_distinct_objects_are_not_duplicates():
    project = _project({
        "cell": [_square(10, 10, 40, number=1), _square(120, 120, 40, number=2)]
    })
    findings = qc.run_audit(project, {"img.png": (200, 200)})
    assert qc.RULE_NEAR_DUPLICATE not in _rules(findings)


def test_heavy_cross_class_overlap_is_reported():
    project = _project({
        "cell": [_square(10, 10, 40, name="cell")],
        "nucleus": [_square(10, 10, 40, name="nucleus")],
    })
    findings = qc.run_audit(project, {"img.png": (200, 200)})
    assert qc.RULE_CROSS_CLASS_OVERLAP in _rules(findings)


def test_a_nucleus_inside_a_cell_is_not_flagged_as_overlap():
    """Nested objects of different classes are the normal case, not a defect."""
    project = _project({
        "cell": [_square(10, 10, 100, name="cell")],
        "nucleus": [_square(40, 40, 20, name="nucleus")],
    })
    findings = qc.run_audit(project, {"img.png": (200, 200)})
    assert qc.RULE_CROSS_CLASS_OVERLAP not in _rules(findings)


# --- statistics ------------------------------------------------------------


def test_area_outliers_are_informational_only():
    annotations = [_square(10 * i, 10, 10, number=i) for i in range(1, 6)]
    annotations.append(_square(0, 120, 150, number=6))
    findings = qc.run_audit(
        _project({"cell": annotations}), {"img.png": (400, 400)}
    )
    match = next(f for f in findings if f.rule == qc.RULE_AREA_OUTLIER)
    assert match.severity == qc.SEVERITY_INFO
    assert match.fixable is False, "an outlier may be a genuinely large object"


def test_no_outlier_verdict_from_too_few_samples():
    """A median over three samples says nothing."""
    annotations = [_square(10, 10, 5, number=1), _square(40, 10, 200, number=2)]
    findings = qc.run_audit(
        _project({"cell": annotations}), {"img.png": (400, 400)}
    )
    assert qc.RULE_AREA_OUTLIER not in _rules(findings)


def test_class_imbalance_is_reported():
    project = _project({
        "cell": [_square(10 * i, 10, 5, name="cell", number=i) for i in range(1, 31)],
        "rare": [_square(10, 100, 5, name="rare")],
    })
    findings = qc.run_audit(project, {"img.png": (500, 500)})
    match = next(f for f in findings if f.rule == qc.RULE_CLASS_IMBALANCE)
    assert match.detail["counts"]["rare"] == 1


def test_images_with_no_annotations_are_listed():
    findings = qc.run_audit(
        {"img.png": {"cell": [_square(10, 10, 20)]}},
        {"img.png": (200, 200), "empty.png": (200, 200)},
    )
    match = next(f for f in findings if f.rule == qc.RULE_EMPTY_IMAGE)
    assert match.image == "empty.png"


def test_slices_are_audited_not_just_top_level_images():
    """Slices live under their own keys in all_annotations; iterating a list of
    top-level images instead would silently skip every one of them."""
    project = {
        "stack.tif_Z1": {"cell": [_square(150, 150, 100)]},
        "stack.tif_Z2": {"cell": [_square(10, 10, 20)]},
    }
    findings = qc.run_audit(
        project, {"stack.tif_Z1": (200, 200), "stack.tif_Z2": (200, 200)}
    )
    assert any(
        f.rule == qc.RULE_OUT_OF_BOUNDS and f.image == "stack.tif_Z1"
        for f in findings
    )


# --- hygiene ---------------------------------------------------------------


def test_class_names_differing_only_by_case_are_reported():
    findings = qc.run_audit({}, {}, class_names=["cell", "Cell"])
    match = next(f for f in findings if f.rule == qc.RULE_SIMILAR_CLASS_NAMES)
    assert "case" in match.message


def test_class_names_one_edit_apart_are_reported():
    findings = qc.run_audit({}, {}, class_names=["cell", "cells"])
    assert qc.RULE_SIMILAR_CLASS_NAMES in _rules(findings)


def test_clearly_different_class_names_are_left_alone():
    findings = qc.run_audit({}, {}, class_names=["cell", "mitochondrion"])
    assert qc.RULE_SIMILAR_CLASS_NAMES not in _rules(findings)


def test_orphan_temp_classes_are_errors():
    findings = qc.run_audit({}, {}, class_names=["cell", "Temp-Auto"])
    match = next(f for f in findings if f.rule == qc.RULE_ORPHAN_TEMP_CLASS)
    assert match.severity == qc.SEVERITY_ERROR


def test_temp_classes_are_exempt_from_the_similar_name_rule():
    """Temp-cell vs cell is by construction, not a typo."""
    findings = qc.run_audit({}, {}, class_names=["cell", "Temp-cell"])
    assert qc.RULE_SIMILAR_CLASS_NAMES not in _rules(findings)


@pytest.mark.parametrize(
    "a,b,expected",
    [("cell", "cell", 0), ("cell", "cells", 1), ("cell", "", 4), ("", "ab", 2),
     ("kitten", "sitting", 3)],
)
def test_edit_distance(a, b, expected):
    assert qc.edit_distance(a, b) == expected


# --- ordering and summary --------------------------------------------------


def test_findings_are_sorted_most_severe_first():
    project = _project({
        "cell": [
            {"segmentation": [0, 0, 40, 40, 40, 0, 0, 40],
             "category_name": "cell", "number": 1},
        ]
    })
    findings = qc.run_audit(
        project, {"img.png": (200, 200), "empty.png": (200, 200)}
    )
    severities = [f.severity for f in findings]
    assert severities == sorted(
        severities, key=lambda s: {"error": 0, "warning": 1, "info": 2}[s]
    )


def test_summarise_counts_by_severity():
    findings = [
        qc.Finding(qc.RULE_OUT_OF_BOUNDS, qc.SEVERITY_ERROR, "x"),
        qc.Finding(qc.RULE_BBOX_MISMATCH, qc.SEVERITY_WARNING, "y"),
        qc.Finding(qc.RULE_EMPTY_IMAGE, qc.SEVERITY_INFO, "z"),
    ]
    summary = qc.summarise(findings)
    assert summary == {"error": 1, "warning": 1, "info": 1, "total": 3}


# --- repairs ---------------------------------------------------------------


def test_repairing_a_self_intersection_produces_valid_geometry():
    annotation = {"segmentation": [0, 0, 40, 40, 40, 0, 0, 40],
                  "category_name": "cell", "number": 1}
    assert qc.apply_fix(annotation, qc.RULE_SELF_INTERSECTING) is True

    findings = qc.run_audit(_project({"cell": [annotation]}), {"img.png": (200, 200)})
    assert qc.RULE_SELF_INTERSECTING not in _rules(findings)


def test_repairing_a_bowtie_keeps_one_annotation_not_two():
    """buffer(0) splits a bow-tie into two lobes; silently turning one
    annotation into two would be a bigger surprise than losing the smaller."""
    annotation = {"segmentation": [0, 0, 40, 40, 40, 0, 0, 40],
                  "category_name": "cell", "number": 1}
    qc.apply_fix(annotation, qc.RULE_SELF_INTERSECTING)
    assert isinstance(annotation["segmentation"], list)
    assert all(isinstance(c, (int, float)) for c in annotation["segmentation"])


def test_repairing_a_bbox_recomputes_it_from_the_outline():
    annotation = _square(10, 10, 40)
    annotation["bbox"] = [0, 0, 1, 1]
    assert qc.apply_fix(annotation, qc.RULE_BBOX_MISMATCH) is True
    assert annotation["bbox"] == [10, 10, 40, 40]


def test_repairing_out_of_bounds_clamps_into_the_image():
    annotation = _square(150, 150, 100)
    assert qc.apply_fix(annotation, qc.RULE_OUT_OF_BOUNDS, 200, 200) is True
    assert all(0 <= c <= 200 for c in annotation["segmentation"])


def test_repairing_out_of_bounds_needs_a_known_size():
    annotation = _square(150, 150, 100)
    assert qc.apply_fix(annotation, qc.RULE_OUT_OF_BOUNDS, None, None) is False


def test_repairing_num_keypoints_recounts_the_flags():
    instance = _pose([(10, 10, 2), (0, 0, 0)], num_keypoints=7)
    assert qc.apply_fix(instance, qc.RULE_POSE_COUNT_MISMATCH) is True
    assert instance["num_keypoints"] == 1


def test_repairing_a_pose_box_encloses_its_points():
    instance = _pose([(10, 10, 2), (30, 30, 2)])
    instance["bbox"] = [0, 0, 5, 5]
    assert qc.apply_fix(instance, qc.RULE_POSE_POINT_OUTSIDE_BBOX, 200, 200) is True
    findings = qc.run_audit(
        _project({"person": [instance]}), {"img.png": (200, 200)}
    )
    assert qc.RULE_POSE_POINT_OUTSIDE_BBOX not in _rules(findings)


@pytest.mark.parametrize(
    "rule",
    [qc.RULE_AREA_OUTLIER, qc.RULE_CLASS_IMBALANCE, qc.RULE_NEAR_DUPLICATE,
     qc.RULE_EMPTY_IMAGE, qc.RULE_SIMILAR_CLASS_NAMES],
)
def test_ambiguous_rules_have_no_auto_fix(rule):
    """Auto-fixing any of these would be guessing at what the user meant."""
    annotation = _square(10, 10, 40)
    assert qc.apply_fix(annotation, rule) is False


# --- config ----------------------------------------------------------------


def test_thresholds_are_configurable():
    project = _project({
        "cell": [_square(10, 10, 40, number=1), _square(20, 10, 40, number=2)]
    })
    strict = qc.QCConfig(duplicate_iou=0.3)
    lenient = qc.QCConfig(duplicate_iou=0.99)

    assert qc.RULE_NEAR_DUPLICATE in _rules(
        qc.run_audit(project, {"img.png": (200, 200)}, config=strict)
    )
    assert qc.RULE_NEAR_DUPLICATE not in _rules(
        qc.run_audit(project, {"img.png": (200, 200)}, config=lenient)
    )


def test_empty_project_is_handled():
    assert qc.run_audit({}, {}) == []
    assert qc.run_audit(None, None) == []
