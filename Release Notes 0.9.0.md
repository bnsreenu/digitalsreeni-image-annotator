# Release Notes
## Version 0.9.0

The first release in over a year — a major update covering AI-assisted detection, model fine-tuning, pose annotation, and a full editing/undo overhaul. Full technical detail is in [CHANGELOG.md](CHANGELOG.md).

### New Features and Enhancements

1. **Grounding-DINO Text-Prompted Detection**
   - Describe the objects you want in plain English and let Grounding-DINO find them.
   - Works on a single image or across a whole batch, with an Enter/Escape review-and-accept overlay before annotations are committed.

2. **SAM 2 Fine-Tuning**
   - Fine-tune SAM 2 on your own annotated data instead of relying on the generic pre-trained weights.
   - Training runs are tracked automatically via MLflow.

3. **Keypoint / Pose Annotation**
   - Define a named keypoint skeleton per class (COCO instance model) with 3-state point visibility (visible / occluded / not placed).
   - Import and export via COCO-keypoints and YOLO-pose formats.
   - Train and predict with YOLO-pose models directly in the app.

4. **Undo / Redo**
   - Ctrl+Z / Ctrl+Y (or Ctrl+Shift+Z) now undo and redo annotation edits — drawing, deleting, merging, editing, and AI-assisted accepts are all covered.

5. **Improved Selection and Editing**
   - Canvas selection (click, Shift+click, rubber-band) is now unified with the annotations table.
   - Any selected shape gets draggable handles for resize/move, plus a vertex-edit mode (double-click) for polygons.
   - All edits are bounds-clamped so annotations can no longer be pushed outside the image.

6. **YOLO Training Expanded**
   - Training and prediction now cover detection, segmentation, and pose (previously segmentation-only), all from your current annotations.

7. **Interface Improvements**
   - Dark mode.
   - On-the-fly UI font scaling (Ctrl+Shift+=/-/0).
   - Image-list filtering and sorting.
   - Annotations table now shows Area and a per-mask "Detail %" simplification control.

8. **Migrated to PyQt6**
   - The whole application moved from PyQt5 to PyQt6 for continued platform support and native integration improvements.

### Under the Hood

- Inference (SAM, Grounding-DINO) now runs in-process on a background thread instead of a subprocess, with a re-entrancy guard to keep the UI responsive.
- Reorganized into a thin main-window orchestrator plus focused controllers and per-tool handlers, making the codebase considerably easier to extend.
- Packaging migrated to `pyproject.toml`; dependencies pinned for reproducible installs.
- A structured logging framework replaces ad hoc `print()` diagnostics; run with `--debug` (or set `IMAGE_ANNOTATOR_DEBUG`) for verbose output.
- Added an automated pytest + pytest-qt test suite covering project save/load, class management, multi-dimensional image slicing, and the SAM/Grounding-DINO controllers.

### Known Issues

- YOLO training is not currently supported for multi-dimensional images (TIFF stacks / CZI slices) — single images only.
- SAM 2 large may crash the application on systems with limited RAM; smaller SAM2 models are recommended.

### Notes

- This release includes contributions from multiple community members — thank you to everyone who submitted PRs and testing feedback while this release was in progress.
- If you're upgrading from 0.8.12, back up any in-progress `.iap` project files before opening them in 0.9.0 as a precaution, though no breaking changes to the project file format are expected.
