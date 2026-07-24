# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Central stdlib `logging` framework (`core/logging_config.py`) with a `--debug`
  / `IMAGE_ANNOTATOR_DEBUG` switch; `print()` is banned in `src/`.
- pytest + pytest-qt coverage for `ProjectController` (`.iap` save/load
  roundtrip + `is_loading_project` guard), `ClassController`, `ImageController`
  multi-dim slicing, `SAMController` debounce/in-flight state machine, and
  `DINOController` review workflow (mocked inference).
- `run-app` Claude skill and a `.claude/settings.json` read-only command
  allowlist.

### Changed
- Packaging migrated from `setup.py` to a PEP 621 `pyproject.toml`; dev/test
  dependencies moved to a `dev` extra; `ultralytics` pinned `>=8.3.27,<9`;
  `requirements.txt` removed.
- All `print()` diagnostics migrated to the logging framework.

### Fixed
- Eliminated silent exception swallowing (seven `except: pass` sites now log;
  the lone bare `except:` removed) under a written error-handling convention.
- Out-of-memory on SAM model load now shows an actionable "pick a smaller
  model" dialog instead of a generic error.

## [0.9.0]

### Added
- Grounding-DINO text-prompted detection (single image + batch) with an
  Enter/Escape review-and-accept overlay.
- SAM 2 fine-tuning via a custom Ultralytics loop, with always-on MLflow
  experiment tracking.
- YOLO training + prediction for detection, segmentation, and pose.
- Keypoint / pose annotation: per-class named schema + skeleton (COCO instance
  model, 3-state visibility), with COCO-keypoints and YOLO-pose export/import.
- Undo / redo of annotation edits (Ctrl+Z / Ctrl+Y) via per-image snapshots.
- Canvas selection unified with the annotations table; handle-based resize/move
  and vertex editing for any selected shape; bounds clamping/clipping.
- Annotations table with Area and per-mask Detail % simplification.
- Dark mode, on-the-fly UI font scaling, and image-list filter/sort.

### Changed
- Migrated from PyQt5 to PyQt6.
- Inference moved from subprocess workers to in-process `QThread` execution with
  a re-entrancy guard.
- Reorganised into a thin `ImageAnnotator` orchestrator + per-responsibility
  controllers + per-tool handlers.
