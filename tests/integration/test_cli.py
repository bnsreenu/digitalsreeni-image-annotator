"""Headless CLI: import guard, exit codes and round-trips (issue #76, ADR-041).

**The import guard is the crux of the whole issue**, which is why it comes
first here. An accidental ``from PyQt6 ...`` in a shared module would silently
make headless validation require a display — and it would work perfectly on the
machine of whoever added it, only failing on a CI runner with no X server. The
same goes for torch: ``main.py`` imports it eagerly before ``QApplication`` to
work around a Windows DLL conflict (ADR-017), and inheriting that would make
``validate`` take seconds instead of milliseconds on every commit.

The guards run in a subprocess because this test session has already imported
both; an in-process ``sys.modules`` check would pass no matter what the CLI does.
"""

import json
import os
import subprocess
import sys

import pytest

from src.digitalsreeni_image_annotator.cli.main import (
    EXIT_ERROR,
    EXIT_FINDINGS,
    EXIT_OK,
    build_parser,
    main,
)


def _subprocess_import_check(module, forbidden):
    code = (
        "import sys;"
        "sys.path.insert(0, 'src');"
        f"import {module};"
        f"bad = [n for n in sys.modules if n.split('.')[0] in {forbidden!r}];"
        "assert not bad, bad;"
        "print('clean')"
    )
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)


# --- the import guard ------------------------------------------------------


def test_importing_the_cli_pulls_in_neither_qt_nor_torch():
    result = _subprocess_import_check(
        "digitalsreeni_image_annotator.cli", ["PyQt6", "torch"]
    )
    assert result.returncode == 0, result.stderr
    assert "clean" in result.stdout


def test_the_command_module_pulls_in_neither_qt_nor_torch():
    result = _subprocess_import_check(
        "digitalsreeni_image_annotator.cli.commands", ["PyQt6", "torch"]
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "module",
    [
        "digitalsreeni_image_annotator.io.export_formats",
        "digitalsreeni_image_annotator.io.import_formats",
        "digitalsreeni_image_annotator.core.project_io",
        "digitalsreeni_image_annotator.core.annotation_qc",
        # The doctor command's engine. This one matters most of all: it exists to
        # explain a Qt that will not import, so importing Qt would make it fail in
        # precisely the environment it was written for (issue #92).
        "digitalsreeni_image_annotator.core.qt_diagnostics",
        # The top-level package itself. main.py's Qt import guard (ADR-046) only ever
        # runs if importing the package has not already pulled Qt in -- if anyone
        # un-lazies __init__.py (ADR-017), the ImportError fires while importing the
        # parent and the user is back to a raw traceback, with the new code never
        # reached. Nothing else covers the package as opposed to its submodules.
        "digitalsreeni_image_annotator",
    ],
)
def test_the_shared_layer_the_cli_depends_on_is_qt_free(module):
    """These are the modules the CLI reaches through. A Qt import in any one of
    them re-breaks headless operation, however clean the cli/ package is."""
    result = _subprocess_import_check(module, ["PyQt6"])
    assert result.returncode == 0, result.stderr


def test_help_works_without_a_display():
    result = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, 'src');"
         "from digitalsreeni_image_annotator.cli import main;"
         "sys.argv = ['sreeni-cli', '--help'];"
         "sys.exit(main())"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    for command in ("export", "convert", "validate", "predict", "doctor"):
        assert command in result.stdout


def test_train_is_deliberately_absent():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["train"])


# --- fixtures --------------------------------------------------------------


def _square(x0, y0, side, name="cell", number=1):
    return {
        "segmentation": [x0, y0, x0 + side, y0, x0 + side, y0 + side, x0, y0 + side],
        "bbox": [x0, y0, side, side],
        "category_id": 1,
        "category_name": name,
        "number": number,
    }


@pytest.fixture
def project(tmp_path):
    """A minimal .iap project with one real image on disk."""
    from PIL import Image

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    Image.new("RGB", (200, 200), (64, 64, 64)).save(images_dir / "a.png")

    data = {
        "classes": [{"name": "cell", "id": 1, "color": "#1F77B4"}],
        "images": [
            {
                "file_name": "a.png",
                "width": 200,
                "height": 200,
                "id": 1,
                "is_multi_slice": False,
                "annotations": {"cell": [_square(10, 10, 40)]},
            }
        ],
        "image_paths": {"a.png": str(images_dir / "a.png")},
        "image_paths_rel": {"a.png": os.path.join("images", "a.png")},
    }
    path = tmp_path / "data.iap"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.fixture
def broken_project(tmp_path):
    """A project with a self-intersecting polygon and an out-of-bounds shape."""
    from PIL import Image

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    Image.new("RGB", (100, 100), (64, 64, 64)).save(images_dir / "a.png")

    bowtie = {
        "segmentation": [0, 0, 40, 40, 40, 0, 0, 40],
        "category_id": 1, "category_name": "cell", "number": 1,
    }
    data = {
        "classes": [{"name": "cell", "id": 1, "color": "#1F77B4"}],
        "images": [
            {
                "file_name": "a.png", "width": 100, "height": 100, "id": 1,
                "is_multi_slice": False,
                "annotations": {"cell": [bowtie, _square(80, 80, 60, number=2)]},
            }
        ],
        "image_paths": {"a.png": str(images_dir / "a.png")},
        "image_paths_rel": {"a.png": os.path.join("images", "a.png")},
    }
    path = tmp_path / "broken.iap"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


# --- validate: the CI gate -------------------------------------------------


def test_validate_exits_zero_on_a_clean_project(project, capsys):
    assert main(["validate", "--project", str(project)]) == EXIT_OK
    summary = json.loads(capsys.readouterr().out.strip())
    assert summary["total"] == 0


def test_validate_exits_two_when_findings_exceed_the_threshold(broken_project):
    """The exit code is the primary output — this is what makes it a gate."""
    assert main(["validate", "--project", str(broken_project)]) == EXIT_FINDINGS


def test_fail_on_never_reports_without_failing(broken_project):
    assert main([
        "validate", "--project", str(broken_project), "--fail-on", "never"
    ]) == EXIT_OK


def test_fail_on_warning_includes_errors(broken_project):
    """Severities are a scale, so the threshold is inclusive-upward — which is
    what makes the flag useful for tightening a gate over time."""
    assert main([
        "validate", "--project", str(broken_project), "--fail-on", "warning"
    ]) == EXIT_FINDINGS


def test_validate_writes_a_json_report_matching_the_findings(
    broken_project, tmp_path
):
    report = tmp_path / "report.json"
    main([
        "validate", "--project", str(broken_project), "--json", str(report)
    ])

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["summary"]["total"] == len(payload["findings"])
    assert any(f["rule"] == "self_intersecting" for f in payload["findings"])
    assert all({"rule", "severity", "message"} <= set(f) for f in payload["findings"])


def test_validate_reports_a_missing_project_rather_than_crashing(tmp_path):
    assert main(["validate", "--project", str(tmp_path / "nope.iap")]) == EXIT_ERROR


def test_validate_rejects_a_file_that_is_not_a_project(tmp_path):
    path = tmp_path / "notes.json"
    path.write_text('{"hello": 1}', encoding="utf-8")
    assert main(["validate", "--project", str(path)]) == EXIT_ERROR


# --- read-only guarantee ---------------------------------------------------


def test_the_cli_never_modifies_the_project_it_reads(project):
    """Autosave and recovery exist to protect interactive editing (ADR-005/032).
    A build script silently rewriting the file it was asked to read is exactly
    the surprise a CI gate must not spring."""
    before = project.read_bytes()
    before_mtime = project.stat().st_mtime

    main(["validate", "--project", str(project)])

    assert project.read_bytes() == before
    assert project.stat().st_mtime == before_mtime


def test_export_does_not_modify_the_project(project, tmp_path):
    before = project.read_bytes()
    main([
        "export", "--project", str(project), "--format", "coco",
        "--out", str(tmp_path / "out"),
    ])
    assert project.read_bytes() == before


def test_no_autosave_or_recovery_file_appears(project, tmp_path):
    before = set(os.listdir(tmp_path))
    main(["validate", "--project", str(project)])
    assert set(os.listdir(tmp_path)) == before


# --- export ----------------------------------------------------------------


def test_a_headless_export_writes_no_frames_and_splits_only_real_images(tmp_path):
    """Why the CLI carries no split warning (ADR-044), asserted through the CLI.

    A video's frames have no pixels headlessly, so they are dropped before the
    split sees them and every surviving name is a file on disk — hence its own
    group, hence nothing to leak. ADR-044 rests its whole "the CLI needs no
    warning" argument on that, so it is pinned here by running the real command
    rather than by restating the arguments it passes.

    An earlier revision shipped a warning branch here that could never fire,
    with a test that passed only on unrelated stderr noise (issue #84).
    """
    from PIL import Image

    images_dir = tmp_path / "images"
    images_dir.mkdir()

    images, paths, rel = [], {}, {}
    for index in range(5):
        name = f"photo{index}.png"
        Image.new("RGB", (60, 60), (64, 64, 64)).save(images_dir / name)
        paths[name] = str(images_dir / name)
        rel[name] = os.path.join("images", name)
        images.append({
            "file_name": name, "width": 60, "height": 60, "id": index + 1,
            "is_multi_slice": False,
            "annotations": {"cell": [_square(5, 5, 20)]},
        })
    # A video: its annotated frames live as slice entries with no file of
    # their own, which is exactly what the CLI cannot resolve. The container
    # itself has to exist, or `run_export` refuses the whole project as having
    # missing images before any of this matters.
    (images_dir / "clip.mp4").write_bytes(b"")
    paths["clip.mp4"] = str(images_dir / "clip.mp4")
    rel["clip.mp4"] = os.path.join("images", "clip.mp4")
    images.append({
        "file_name": "clip.mp4", "width": 60, "height": 60, "id": 99,
        "is_multi_slice": True,
        "slices": [
            {"name": f"clip_F{i:05d}", "annotations": {"cell": [_square(5, 5, 20)]}}
            for i in range(20)
        ],
    })

    project_path = tmp_path / "mixed.iap"
    project_path.write_text(json.dumps({
        "classes": [{"name": "cell", "id": 1, "color": "#1F77B4"}],
        "images": images,
        "image_paths": paths,
        "image_paths_rel": rel,
    }), encoding="utf-8")

    out = tmp_path / "out"
    assert main([
        "export", "--project", str(project_path), "--format", "yolov5",
        "--out", str(out), "--val-split", "20",
    ]) == EXIT_OK

    train = os.listdir(out / "images" / "train")
    val = os.listdir(out / "images" / "val")
    assert not any(name.startswith("clip_F") for name in train + val), (
        "a frame was written headlessly; the no-warning argument no longer holds"
    )
    # The 20% is over the five real images, not over the 25 annotated names.
    assert len(train) == 4 and len(val) == 1


def test_export_writes_coco(project, tmp_path):
    out = tmp_path / "coco_out"
    assert main([
        "export", "--project", str(project), "--format", "coco", "--out", str(out)
    ]) == EXIT_OK

    written = list(out.rglob("*.json"))
    assert written, "no COCO json produced"
    payload = json.loads(written[0].read_text(encoding="utf-8"))
    assert payload["images"][0]["width"] == 200, "dimensions read without Qt"
    assert len(payload["annotations"]) == 1


def test_export_writes_yolo(project, tmp_path):
    out = tmp_path / "yolo_out"
    assert main([
        "export", "--project", str(project), "--format", "yolov5", "--out", str(out)
    ]) == EXIT_OK
    assert list(out.rglob("*.txt")), "no YOLO label files produced"


def test_export_refuses_rather_than_writing_a_partial_dataset(tmp_path):
    """A partial export that looks complete is worse than a refusal: the
    dataset would train on fewer images than the user believes."""
    data = {
        "classes": [{"name": "cell", "id": 1, "color": "#1F77B4"}],
        "images": [{
            "file_name": "gone.png", "width": 10, "height": 10, "id": 1,
            "is_multi_slice": False,
            "annotations": {"cell": [_square(1, 1, 4)]},
        }],
        "image_paths": {"gone.png": str(tmp_path / "nowhere" / "gone.png")},
    }
    path = tmp_path / "missing.iap"
    path.write_text(json.dumps(data), encoding="utf-8")

    assert main([
        "export", "--project", str(path), "--format", "coco",
        "--out", str(tmp_path / "out"),
    ]) == EXIT_ERROR


# --- convert ---------------------------------------------------------------


def test_convert_coco_to_yolo_and_back(project, tmp_path):
    """Round-trip through two formats without a project involved."""
    coco_dir = tmp_path / "coco"
    assert main([
        "export", "--project", str(project), "--format", "coco",
        "--out", str(coco_dir),
    ]) == EXIT_OK
    coco_json = next(coco_dir.rglob("*.json"))

    yolo_dir = tmp_path / "yolo"
    assert main([
        "convert", "--in", str(coco_json), "--from", "coco",
        "--to", "yolov5", "--out", str(yolo_dir),
        "--images", str(coco_dir / "images"),
    ]) == EXIT_OK
    assert list(yolo_dir.rglob("*.txt"))


def test_convert_reports_an_unreadable_input(tmp_path):
    assert main([
        "convert", "--in", str(tmp_path / "nope.json"), "--from", "coco",
        "--to", "yolov5", "--out", str(tmp_path / "out"),
    ]) == EXIT_ERROR


# --- predict ---------------------------------------------------------------


def test_predict_reports_a_missing_image_directory(tmp_path):
    assert main([
        "predict", "--model", "x.pt", "--images", str(tmp_path / "nope"),
        "--out", str(tmp_path / "out"),
    ]) == EXIT_ERROR


def test_predict_reports_an_empty_image_directory(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert main([
        "predict", "--model", "x.pt", "--images", str(empty),
        "--out", str(tmp_path / "out"),
    ]) == EXIT_ERROR
