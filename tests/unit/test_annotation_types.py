"""Typed annotation shapes and the type-check gate (issue #78).

The codebase's most meaning-carrying structure was a dict whose valid shapes
lived in prose. The sharpest case: a pose instance is distinguished from a
polygon by the **absence** of a key (ADR-029) — a rule only a comment protected.

The last test here is the important one. A type-check gate that does not
actually check anything is worse than no gate: it reports success forever while
teaching everyone to trust it. So it is verified by feeding mypy a deliberately
wrong call and asserting it complains.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

# tomllib is stdlib only from 3.12; the project floor is 3.10, so fall back to
# the `tomli` backport (a dev dependency on those versions). Without this the
# whole module fails to import on 3.10/3.11 and takes the suite down with it.
try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - version-dependent
    import tomli as tomllib

import pytest

from src.digitalsreeni_image_annotator.core import annotation_types as at

REPO_ROOT = Path(__file__).resolve().parents[2]


def _has_mypy() -> bool:
    """mypy is a dev extra, so the gate test skips where it is not installed
    rather than failing a runtime-only environment."""
    import importlib.util

    return importlib.util.find_spec("mypy") is not None


# --- the discriminator, expressed once -------------------------------------


def test_a_pose_is_identified_by_the_absence_of_a_segmentation():
    instance = {"keypoints": [1, 2, 2], "num_keypoints": 1, "category_name": "p"}
    assert at.is_pose(instance) is True
    assert at.is_polygon(instance) is False


def test_a_polygon_is_not_a_pose():
    mask = {"segmentation": [0, 0, 1, 0, 1, 1], "category_name": "cell"}
    assert at.is_pose(mask) is False
    assert at.is_polygon(mask) is True


def test_a_none_valued_segmentation_does_not_read_as_a_polygon():
    """A bbox-only import carries ``"segmentation": None``, which is exactly why
    existence-only ``"segmentation" in ann`` checks are a hazard and truthiness
    is the safe test."""
    imported = {"segmentation": None, "bbox": [0, 0, 5, 5], "category_name": "c"}
    assert at.is_polygon(imported) is False
    assert at.is_bbox_only(imported) is True


def test_a_pose_that_somehow_carries_a_segmentation_is_treated_as_a_polygon():
    """Documents the precedence rather than pretending the case cannot occur:
    an annotation with a usable mask is routed by that mask."""
    hybrid = {
        "keypoints": [1, 2, 2],
        "segmentation": [0, 0, 1, 0, 1, 1],
        "category_name": "p",
    }
    assert at.is_pose(hybrid) is False
    assert at.is_polygon(hybrid) is True


def test_bbox_only_excludes_poses_and_masks():
    assert at.is_bbox_only({"bbox": [0, 0, 1, 1]}) is True
    assert at.is_bbox_only({"keypoints": [1, 2, 2], "bbox": [0, 0, 1, 1]}) is False
    assert at.is_bbox_only({"segmentation": [0, 0, 1, 0, 1, 1]}) is False


def test_the_pose_typed_dict_declares_no_segmentation_key():
    """The type definition itself expresses the discriminator — declaring a
    segmentation key on PoseAnnotation, even optional, would legitimise writing
    one, and writing one breaks every existence-only check."""
    assert "segmentation" not in at.PoseAnnotation.__annotations__
    assert "keypoints" in at.PoseAnnotation.__annotations__
    assert "segmentation" in at.PolygonAnnotation.__annotations__


def test_the_typed_dicts_are_all_total_false():
    """The annotation dict legitimately gains keys at runtime — segmentation_raw
    (ADR-025), source and track_run (ADR-040), assigned_class (#69). A rigid
    schema would produce false errors and teach people to ignore the checker."""
    for typed_dict in (
        at.PolygonAnnotation, at.BBoxAnnotation, at.PoseAnnotation,
        at.KeypointSchema,
    ):
        assert typed_dict.__total__ is False


def test_the_types_module_imports_without_qt():
    """Both the GUI and the CLI import these."""
    code = (
        "import sys;"
        "sys.path.insert(0, 'src');"
        "import digitalsreeni_image_annotator.core.annotation_types as m;"
        "qt = [n for n in sys.modules if n.startswith('PyQt6')];"
        "assert not qt, qt;"
        "print('clean')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


# --- the gate itself -------------------------------------------------------


def _mypy_config():
    with open(REPO_ROOT / "pyproject.toml", "rb") as handle:
        return tomllib.load(handle)["tool"]["mypy"]


def _checked_modules():
    with open(REPO_ROOT / "pyproject.toml", "rb") as handle:
        config = tomllib.load(handle)["tool"]["mypy"]
    checked = []
    for override in config.get("overrides", []):
        if override.get("ignore_errors") is False:
            checked.extend(override["module"])
    return checked


def test_unannotated_modules_are_excluded_explicitly_not_silently():
    """Global ignore_errors plus a per-module opt-in means the boundary is
    visible. The inverse — checking everything and suppressing failures — would
    hide which modules are actually covered."""
    config = _mypy_config()
    assert config["ignore_errors"] is True, "global default must stay permissive"
    assert _checked_modules(), "nothing is opted in; the gate would be vacuous"


def test_the_annotated_core_modules_are_in_scope():
    checked = set(_checked_modules())
    for name in (
        "annotation_types", "annotation_qc", "disagreement", "project_io",
        "similarity", "task_inference", "model_sidecar", "image_size", "onion",
    ):
        assert f"digitalsreeni_image_annotator.core.{name}" in checked, name


def test_third_party_ignores_are_listed_not_global():
    """A global ignore_missing_imports would hide a genuinely missing internal
    module; an explicit list stays visible and shrinks as upstreams ship stubs."""
    config = _mypy_config()
    assert config.get("ignore_missing_imports") is not True
    ignored = [
        override["module"]
        for override in config.get("overrides", [])
        if override.get("ignore_missing_imports") is True
    ]
    assert ignored, "expected an explicit third-party ignore list"


@pytest.mark.skipif(
    not _has_mypy(),
    reason="mypy is a dev extra; the gate is verified where it is installed",
)
def test_the_gate_actually_catches_a_wrong_call(tmp_path):
    """THE test that keeps the gate honest.

    A type-check step that checks nothing reports success forever while
    teaching everyone to trust it. Feeding mypy a deliberately wrong call
    against an in-scope module proves the configuration reaches real code.
    """
    # Run the REAL gate (`python -m mypy`, project config, no extra flags)
    # against a copy of the tree with one deliberate error injected into an
    # in-scope module. Checking a scratch file outside the package would prove
    # nothing: the global `ignore_errors` correctly applies to it, so it would
    # pass however broken the configuration was.
    shutil.copytree(REPO_ROOT / "src", tmp_path / "src")
    shutil.copy2(REPO_ROOT / "pyproject.toml", tmp_path / "pyproject.toml")

    target = tmp_path / "src" / "digitalsreeni_image_annotator" / "core" / "onion.py"
    target.write_text(
        target.read_text(encoding="utf-8")
        + "\n\ndef _deliberately_wrong() -> str:\n    return clamp_offset(3)\n",
        encoding="utf-8",
    )

    # One file, not the whole tree: the project config still applies (that is
    # what is under test), and checking 100+ modules to prove one point would
    # add half a minute to every suite run.
    result = subprocess.run(
        [sys.executable, "-m", "mypy",
         os.path.join("src", "digitalsreeni_image_annotator", "core", "onion.py")],
        capture_output=True, text=True, cwd=str(tmp_path), env=dict(os.environ),
    )

    assert result.returncode != 0, (
        "the type-check gate accepted a wrong return type, so it is checking "
        f"nothing:\n{result.stdout}"
    )
    assert "onion.py" in result.stdout
    assert "return-value" in result.stdout or "Incompatible return" in result.stdout
