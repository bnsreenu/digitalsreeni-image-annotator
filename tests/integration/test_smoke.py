"""
Smoke tests: verify the package's public API surface and every internal
module can be imported.

These tests are the safety net for the modular refactoring. They catch
the most common refactor regressions (renames, missing re-exports, broken
intra-package imports) without needing a real image, a SAM model, or a
running Qt event loop.

If a module gets moved into a subpackage, the corresponding line below
must be updated. That is the test's whole point.
"""

import importlib

import pytest


def test_public_api_exports():
    """The five names documented in __init__.py must remain importable."""
    import digitalsreeni_image_annotator as pkg

    assert pkg.__version__ == "0.9.1"
    assert hasattr(pkg, "ImageAnnotator")
    assert hasattr(pkg, "ImageLabel")
    assert hasattr(pkg, "SAMUtils")
    assert hasattr(pkg, "calculate_area")
    assert hasattr(pkg, "calculate_bbox")


INTERNAL_MODULES = [
    # Core
    # NOTE: 'main' is deliberately omitted — it eagerly imports torch before
    # QApplication is created (ADR-017). In a pytest-qt process Qt is already
    # loaded; importing torch afterward triggers WinError 1114.  main is the
    # entry point and is validated by the CLI smoke tests instead.
    "digitalsreeni_image_annotator.annotator_window",
    "digitalsreeni_image_annotator.utils",
    # Controllers
    "digitalsreeni_image_annotator.controllers.annotation_history",
    # Widgets
    "digitalsreeni_image_annotator.widgets.image_label",
    "digitalsreeni_image_annotator.widgets.video_timeline",
    # Inference
    "digitalsreeni_image_annotator.inference.sam_utils",
    "digitalsreeni_image_annotator.inference.dino_utils",
    # I/O
    "digitalsreeni_image_annotator.io.export_formats",
    "digitalsreeni_image_annotator.io.import_formats",
    # Core helpers
    "digitalsreeni_image_annotator.core.constants",
    "digitalsreeni_image_annotator.core.annotation_utils",
    "digitalsreeni_image_annotator.core.torch_utils",
    "digitalsreeni_image_annotator.core.keypoint_schema",
    # Keypoint / pose tool (issue #35)
    "digitalsreeni_image_annotator.widgets.tools.keypoint_tool",
    "digitalsreeni_image_annotator.dialogs.keypoint_schema_dialog",
    # UI
    "digitalsreeni_image_annotator.ui.default_stylesheet",
    "digitalsreeni_image_annotator.ui.soft_dark_stylesheet",
    # Dialogs
    "digitalsreeni_image_annotator.dialogs.annotation_statistics",
    "digitalsreeni_image_annotator.dialogs.coco_json_combiner",
    "digitalsreeni_image_annotator.dialogs.dataset_splitter",
    "digitalsreeni_image_annotator.dialogs.dicom_converter",
    "digitalsreeni_image_annotator.dialogs.dino_merge_dialog",
    "digitalsreeni_image_annotator.dialogs.dino_phrase_editor",
    "digitalsreeni_image_annotator.dialogs.help_window",
    "digitalsreeni_image_annotator.dialogs.image_augmenter",
    "digitalsreeni_image_annotator.dialogs.image_patcher",
    "digitalsreeni_image_annotator.dialogs.project_details",
    "digitalsreeni_image_annotator.dialogs.project_search",
    "digitalsreeni_image_annotator.dialogs.slice_registration",
    "digitalsreeni_image_annotator.dialogs.snake_game",
    "digitalsreeni_image_annotator.dialogs.stack_interpolator",
    "digitalsreeni_image_annotator.dialogs.stack_to_slices",
    "digitalsreeni_image_annotator.dialogs.yolo_trainer",
]


@pytest.mark.parametrize("module_name", INTERNAL_MODULES)
def test_internal_module_imports(module_name):
    """Every internal module must import without raising."""
    importlib.import_module(module_name)


def test_keypoint_tool_activates(qt_application):
    """Defining a pose schema and activating the keypoint tool must not crash
    (issue #35) — exercises the registration + activation wiring end-to-end."""
    from digitalsreeni_image_annotator.annotator_window import ImageAnnotator

    w = ImageAnnotator()
    w.auto_save = lambda: None  # no project file → auto_save would pop a modal
    try:
        assert "keypoint" in w.image_label._tools
        w.add_class("person")
        w.keypoint_schemas["person"] = {
            "names": ["a", "b"], "skeleton": [[0, 1]], "flip_idx": [0, 1]
        }
        w.current_class = "person"
        w.activate_tool("keypoint")
        assert w.image_label.current_tool == "keypoint"
        assert w.keypoint_button.isChecked()
        w.activate_tool(None)
        assert w.image_label.current_tool is None
    finally:
        w.deleteLater()


def test_annotator_window_inline_imports_are_resolvable():
    """Parse annotator_window.py AST, verify every bare relative import
    (from .module) resolves to a file still in the package root.

    This catches stale inline imports inside function bodies that are
    invisible to test_internal_module_imports because Python defers
    execution until the function is called. Phase 1 moved 25 modules
    into subpackages; four inline imports in annotator_window.py were
    missed and only surfaced at runtime (e.g. from .dino_utils import
    GDINO_MODEL_PATHS which needed to be .inference.dino_utils).
    """
    import ast
    import pathlib

    # Package root — modules that stayed at root live here
    pkg_dir = (
        pathlib.Path(__file__).parents[2]
        / "src"
        / "digitalsreeni_image_annotator"
    )
    source = (pkg_dir / "annotator_window.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    bad = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        # Only bare relative imports at level 1 (from .module)
        if node.level != 1 or not node.module:
            continue
        module = node.module
        # Proper subpackage imports are fine (e.g. .dialogs.foo)
        dots = module.split(".")
        if dots[0] in ("controllers", "dialogs", "inference", "io", "ui", "widgets", "core", "training"):
            continue
        # Root-level modules that stayed behind: utils, annotator_window, main
        root_py = pkg_dir / f"{module}.py"
        root_pkg = pkg_dir / module / "__init__.py"
        if not root_py.exists() and not root_pkg.exists():
            bad.append((node.lineno, f"from .{module} import ..."))

    assert not bad, (
        f"Stale inline imports in annotator_window.py at lines: {bad}. "
        f"The module was likely moved into a subpackage; update the import path."
    )


def test_main_imports_torch_before_qt():
    """Parse main.py's AST: `import torch` must come before any PyQt6 import.

    Two ADRs hang off this single ordering. ADR-017 requires torch to claim its DLL
    slot before Qt loads, or `c10.dll` init fails with WinError 1114; ADR-046 added a
    guarded Qt import immediately below it, so the two Windows DLL workarounds are now
    adjacent and an innocent-looking tidy-up can swap them. The regression reproduces
    only on Windows with torch installed, and `main.py` is deliberately excluded from
    the import list above (importing it here would trigger that very conflict) -- so
    reading the source is the only gate available.
    """
    import ast
    import pathlib

    source = (
        pathlib.Path(__file__).parents[2]
        / "src" / "digitalsreeni_image_annotator" / "main.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)

    # min() over every match, not the first one walk() happens to yield: ast.walk is
    # breadth-first, so "first visited" is not "lowest line" as soon as anyone adds a
    # nested import.
    torch_lines = []
    qt_lines = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "torch":
                    torch_lines.append(node.lineno)
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root == "torch":
                torch_lines.append(node.lineno)
            elif root == "PyQt6":
                qt_lines.append(node.lineno)

    assert torch_lines, "main.py no longer imports torch (ADR-017)"
    assert qt_lines, "main.py no longer imports PyQt6"
    torch_line, qt_line = min(torch_lines), min(qt_lines)
    assert torch_line < qt_line, (
        f"main.py imports PyQt6 at line {qt_line} before torch at line {torch_line}. "
        "ADR-017 requires torch first; this fails only on Windows with torch installed."
    )


def test_main_guards_the_qt_import():
    """The PyQt6 import in main.py must sit inside a try/except (ADR-046).

    Without it a contaminated environment gets a bare `DLL load failed` traceback,
    which is the entire complaint in issue #92.
    """
    import ast
    import pathlib

    source = (
        pathlib.Path(__file__).parents[2]
        / "src" / "digitalsreeni_image_annotator" / "main.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)

    guarded = any(
        isinstance(node, ast.Try)
        and any(
            isinstance(inner, ast.ImportFrom)
            and (inner.module or "").split(".")[0] == "PyQt6"
            for inner in ast.walk(node)
        )
        for node in ast.walk(tree)
    )
    assert guarded, "main.py imports PyQt6 outside a try/except (ADR-046)"
