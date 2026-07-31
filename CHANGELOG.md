# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Class and tool hotkeys** — `1`…`9` select the first nine classes, `P`/`R`/`B`/`E`/`K`
  pick the polygon, rectangle, paint, eraser and keypoint tools, `V` returns to
  selection mode. Bare keys go through a gated event filter rather than global
  shortcuts, so they stay inert while typing in any text field.
- **Copy and paste annotations** (`Ctrl+C` / `Ctrl+V`) across images, slices and
  video frames, clamped into the target's bounds.
- **Onion-skinning** for stacks, videos and slice navigation — show neighbouring
  slices' annotations, image, or both, with configurable depth and opacity.
- **Insert and delete polygon vertices**: double-click an edge in vertex-edit
  mode to add one, Alt+click a vertex to remove it (refused at three).
- **Segment Everything** — unprompted SAM mask proposals routed into the existing
  review overlay, with noise filtering, rather than a second review mechanic.
- **Annotation QC audit** — rule-based geometry, redundancy, statistics, hygiene
  and pose checks with unambiguous one-click repairs, applied as a single undo
  entry.
- **Model-vs-ground-truth review ranking** — score every image by how much a
  trained model disagrees with its labels (or how unsure it is where there are
  none), then sort the image list by it. Closes the active-learning loop.
- **Embedding-based dataset curation** — CLIP or DINOv2 image embeddings,
  switchable per dataset because which one suits a given kind of image is not a
  question that has a global answer. Cosine near-duplicate clusters, cluster
  cohesion (so a chain is distinguishable from a blob), coarse appearance modes
  and a diversity report. The cache is keyed by content hash *and* model and
  covers slices and video frames as well as files on disk, so a second run over
  a video project is nearly instant. Clustering is one blocked NumPy pass, which
  moves the supported ceiling from 3000 images to 20 000 and makes the binding
  cost embedding time rather than comparison. Recommends only; it has no delete
  path by design.
- **Near-duplicate clusters seed the train/val split.** A cluster is evidence
  that two images must not land on opposite sides of a split. Structure already
  catches a stack's slices and a video's frames; embeddings catch what structure
  cannot — a folder of frames extracted as ordinary files, where the name says
  "independent image" and the pixels say otherwise. Nothing is computed unless a
  curation run has already happened, and every split path (YOLO export, dataset
  preparation, the Train dialog and SAM fine-tuning) reads the same grouping the
  warning described.
- **One Train Model dialog** for both YOLO and SAM 2, inferring the task from the
  annotations and performing dataset preparation, YAML handling, loading and
  saving implicitly.
- **Post-training lifecycle** — trained models are registered automatically,
  weights copied into the project with a JSON sidecar, results reported, and
  offered for immediate trial.
- **Pascal VOC annotation import** (bbox and bbox+segmentation).
- **Headless CLI** (`sreeni-cli`): `export`, `convert`, `validate` and `predict`,
  behind a Qt-free module boundary enforced by a subprocess test so headless
  operation never comes to require a display.
- Type hints across the Qt-free core, with `ruff` and `mypy` as separate CI
  steps, and coverage for the canvas interaction paths.

### Fixed
- **The train/val split scattered a video across both sides.** The split was
  keyed on the image name, but a multi-dimensional stack contributes one name
  per slice and a video one per frame — so near-identical frames of one
  recording landed in both train and validation by construction. Nothing failed;
  the reported validation metrics simply came back better the more redundant the
  data was. The split key is now the *group*, so a stack's slices and a video's
  frames move together, across YOLO v4/v5+ export, in-app YOLO training and SAM
  fine-tuning alike (where validation loss also drives early stopping). Where no
  leak-free split exists — a project that is a single recording — the app says
  so plainly and offers to back out instead of silently reporting optimistic
  numbers.
- The DINO phrase panel now follows the class list, and class rename/delete
  reaches every name-keyed structure including the undo history, which could
  previously restore annotations under a dead class name.

## [0.9.1] - 2026-07-27

### Added
- Comprehensive `USER_MANUAL.md` (+ print-ready HTML/PDF versions) covering
  every v0.9.0 feature, including video annotation, SAM 3 text-prompted
  detection and video object tracking, keypoint/pose annotation, SAM 2
  fine-tuning, and undo/redo/shape editing.
- pepy.tech downloads badge on the README.

### Changed
- Expanded the in-app Help window (F1) with matching sections and a pointer
  to the full manual; corrected a stale keyboard shortcut.
- README demo video/thumbnail now points to the v0.9.0 walkthrough.

No functional/application code changed in this release.

## [0.9.0] - 2026-07-24

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
- Central stdlib `logging` framework (`core/logging_config.py`) with a `--debug`
  / `IMAGE_ANNOTATOR_DEBUG` switch; `print()` is banned in `src/`.
- pytest + pytest-qt coverage for `ProjectController` (`.iap` save/load
  roundtrip + `is_loading_project` guard), `ClassController`, `ImageController`
  multi-dim slicing, `SAMController` debounce/in-flight state machine, and
  `DINOController` review workflow (mocked inference).
- `run-app` Claude skill and a `.claude/settings.json` read-only command
  allowlist.

### Changed
- Migrated from PyQt5 to PyQt6.
- Inference moved from subprocess workers to in-process `QThread` execution with
  a re-entrancy guard.
- Reorganised into a thin `ImageAnnotator` orchestrator + per-responsibility
  controllers + per-tool handlers.
- Packaging migrated from `setup.py` to a PEP 621 `pyproject.toml`; dev/test
  dependencies moved to a `dev` extra; `ultralytics` pinned `>=8.3.27,<9`;
  `requirements.txt` removed.
- All `print()` diagnostics migrated to the logging framework.

### Fixed
- Eliminated silent exception swallowing (seven `except: pass` sites now log;
  the lone bare `except:` removed) under a written error-handling convention.
- Out-of-memory on SAM model load now shows an actionable "pick a smaller
  model" dialog instead of a generic error.
