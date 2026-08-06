"""Guards on the declared dependencies (issue #92).

The PyQt6 requirement had no upper bound, so every fresh install silently took
whatever Qt minor had shipped most recently -- including ones the CI matrix has never
run. This asserts the ceiling is still there, because it is a one-character deletion
away from being gone and nothing else would notice.
"""

from pathlib import Path

# tomllib is stdlib only from 3.12; the project floor is 3.10, so fall back to the
# `tomli` backport (a dev dependency on those versions), matching
# tests/unit/test_annotation_types.py.
try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - version-dependent
    import tomli as tomllib

REPO_ROOT = Path(__file__).resolve().parents[2]


def _dependencies():
    with open(REPO_ROOT / "pyproject.toml", "rb") as handle:
        return tomllib.load(handle)["project"]["dependencies"]


def _requirement(name):
    prefix = name.lower()
    for entry in _dependencies():
        # Split on the first comparison character; the name is everything before it.
        head = entry.split(">")[0].split("<")[0].split("=")[0].split("!")[0].split("~")[0]
        if head.strip().lower() == prefix:
            return entry
    return None


def test_pyqt6_is_declared():
    assert _requirement("PyQt6") is not None


def test_pyqt6_has_an_upper_bound():
    """Without a ceiling, an untested Qt minor reaches users the day it is released.

    The bound is a tested ceiling, NOT the fix for issue #92 -- that report is DLL
    shadowing inside a Conda environment (ADR-046). If this test ever fails because
    someone tightened the cap to work around #92, read the comment in pyproject.toml
    before changing it.
    """
    requirement = _requirement("PyQt6")
    assert "<" in requirement, (
        f"PyQt6 requirement {requirement!r} has no upper bound"
    )


def test_pyqt6_floor_still_admits_the_documented_minimum():
    """docs/02_architecture_constraints.md and ADR-014 both promise 6.7 as the floor.

    Parsed rather than substring-matched: `">=6.7" in spec` also passes on `>=6.70`.
    """
    import re

    floors = re.findall(r">=\s*([0-9][0-9.]*)", _requirement("PyQt6"))
    assert floors, "PyQt6 requirement has no lower bound"
    assert _version_tuple(floors[0]) <= (6, 7, 0)


def _version_tuple(spec):
    parts = [int(part) for part in spec.split(".") if part.isdigit()]
    return tuple(parts + [0] * (3 - len(parts)))


def test_ultralytics_stays_bounded():
    """It carries a ceiling for the same reason as PyQt6 (asserted above): a major bump
    upstream has broken this app before, and CI cannot see it coming."""
    assert "<" in _requirement("ultralytics")
