# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

DigitalSreeni Image Annotator - PyQt6 desktop app for image annotation with SAM 2 integration and multi-dimensional image support.

**Fork of**: https://github.com/bnsreenu/digitalsreeni-image-annotator

## Quick Reference

```bash
# Install
pip install -e .

# Run
python -m src.digitalsreeni_image_annotator.main
# or: digitalsreeni-image-annotator
# or: sreeni
```

## Tech Stack

Python 3.10+ | PyQt6 6.7+ | Ultralytics 8.3.27 (SAM 2) | NumPy | OpenCV | Shapely

**Test suite**: `tests/` (pytest + pytest-qt). 94 tests pass on PyQt6.

## Documentation

For detailed architecture and design information, see **[docs/](docs/)**:

- **[Building Block View](docs/05_building_block_view.md)** - Components, data model, class responsibilities
- **[Runtime View](docs/06_runtime_view.md)** - Workflows and key scenarios
- **[Cross-cutting Concepts](docs/08_crosscutting_concepts.md)** - Coordinate systems, conversions, patterns
- **[Architecture Decisions](docs/09_architecture_decisions.md)** - Why we made key choices
- **[Glossary](docs/12_glossary.md)** - Terms, acronyms, data structures

See [docs/README.md](docs/README.md) for full documentation index.

## Project Structure

```
src/digitalsreeni_image_annotator/
├── main.py                       # Entry point
├── annotator_window.py           # ImageAnnotator - thin orchestrator
├── app_settings.py               # QSettings UI prefs: ui_font_pt, dark_mode (ADR-020)
├── utils.py                      # Utility functions (calculate_area, …)
├── __init__.py                   # Public API re-exports
│
├── core/                         # constants, annotation_utils, image_utils
├── controllers/                  # 8 controllers (project, image, sam,
│                                 #   sam_train, dino, yolo, annotation,
│                                 #   class) + io_controller
├── widgets/
│   ├── image_label.py            # ImageLabel canvas widget (dispatcher)
│   ├── canvas_context.py         # CanvasContext read accessor (ADR-018)
│   └── tools/                    # Per-tool handlers (ADR-019): rectangle,
│                                 #   polygon, paint, eraser
├── inference/                    # sam_utils.py, dino_utils.py
├── training/                     # SAM fine-tuning (ADR-021): sam_trainer.py
│                                 #   (SAMFineTuner), sam_dataset.py;
│                                 #   lr_schedule.py + early_stop.py (ADR-028)
├── io/                           # export_formats.py, import_formats.py
├── ui/                           # menu_bar, sidebar, shortcuts, theme, stylesheets
└── dialogs/                      # Standalone tool dialogs (statistics,
                                  #   splitter, augmenter, … 16 files)
```

## Key Classes

| Class | File | Responsibility |
|-------|------|----------------|
| `ImageAnnotator` | annotator_window.py | Thin orchestrator — holds controllers, wires signals, delegates almost everything |
| `ImageLabel` | widgets/image_label.py | Canvas display, zoom/pan, event dispatch to tool handlers |
| `CanvasContext` | widgets/canvas_context.py | Narrow read view of main-window state for ImageLabel (ADR-018) |
| `ToolHandler` (+ 4 subclasses) | widgets/tools/ | Per-tool mouse/key handling (rectangle, polygon, paint, eraser) (ADR-019) |
| `ProjectController` | controllers/project_controller.py | `.iap` save/load, auto-save, `is_loading_project` guard |
| `ImageController` | controllers/image_controller.py | TIFF/CZI loading, multi-dim slicing, image/slice switching |
| `AnnotationController` | controllers/annotation_controller.py | Annotation CRUD, sort, edit-mode, finish_polygon/rectangle |
| `ClassController` | controllers/class_controller.py | Class add/delete/rename/colour/visibility |
| `SAMController` | controllers/sam_controller.py | SAM model picker, debounce, in-flight guard (ADR-013) |
| `DINOController` | controllers/dino_controller.py | DINO single/batch detection, batch review, temp-class workflow |
| `YOLOController` | controllers/yolo_controller.py | YOLO training menu + prediction wiring |
| `SAMUtils` | inference/sam_utils.py | Load SAM models (built-in + fine-tuned), run inference |
| `DINOUtils` | inference/dino_utils.py | Grounding-DINO model load + inference |
| `SAMFineTuner` | training/sam_trainer.py | Fine-tune SAM 2 decoder/encoder via custom loop over Ultralytics SAM2Model (ADR-021) |
| `SAMTrainController` | controllers/sam_train_controller.py | SAM fine-tune menu, GPU gate, training thread, selector registration |

See [Building Block View](docs/05_building_block_view.md) for detailed class documentation.

## Common Development Tasks

### Adding a New Annotation Tool

1. Add button in `ImageAnnotator.create_tool_section()`
2. Set `image_label.current_tool` on click
3. Handle mouse events in `ImageLabel` (mousePressEvent, mouseMoveEvent)
4. Render in `ImageLabel.paintEvent()`
5. Commit via `self.annotationCommitted.emit(annotation_dict)` — the
   orchestrator routes it to `AnnotationController.add_annotation_to_list`
   (see ADR-018)

### Working with Annotations

```python
# Annotation storage: dict[image_filename, list[annotation_dict]]
self.all_annotations[self.image_file_name].append({
    "segmentation": [x1, y1, x2, y2, ...],  # Polygon
    # OR "bbox": [x, y, width, height],     # Rectangle
    "category": class_name
})
```

### SAM Integration

SAM runs in-process; the Ultralytics model object lives on `SAMUtils`
and persists across calls. Inference runs on a background QThread but
the public API is synchronous — see ADR-013 in
`docs/09_architecture_decisions.md`.

```python
# Load model on first selection (downloads weights if missing, ~40-400MB)
self.sam_utils.change_sam_model("SAM 2 tiny")  # blocks UI thread via QEventLoop spin

# Run inference (also runs on worker thread, returns when done)
prediction = self.sam_utils.apply_sam_points(
    qimage,
    positive_points=[(x1, y1)],
    negative_points=[(x2, y2)]
)
# Returns: {"segmentation": [...], "score": float}
```

## Important Notes

### Platform Support
- ✅ Windows, macOS, Linux supported (PyQt6 native integration improved over PyQt5)
- Linux runtime needs libxcb-cursor0 (Qt6 requires this; was optional under Qt5)

### Critical: Project Loading
**Always check `is_loading_project` flag before saving!** Autosave during load corrupts files (v0.8.12 fix).

```python
def save_project(self):
    if self.is_loading_project:
        return  # Skip during load
    # ... save logic
```

### Coordinate Systems
- **Mouse events**: Screen coordinates → must convert to image coordinates
- **Annotations stored**: Image coordinates (unzoomed, absolute pixels)
- Account for `zoom_factor`, `offset_x`, `offset_y`

See [Cross-cutting Concepts](docs/08_crosscutting_concepts.md#coordinate-systems) for details.

### Multi-dimensional Images
- User assigns dimensions (T, Z, C, H, W) via dialog
- Slices extracted with names like `stack.tif_T0_Z5_C0`
- Each slice annotated independently
- Stored in `image_slices` dict
- TIFF axis hint: `load_tiff` reads `tifffile.series[0].axes` and pre-fills the dimension dialog; ndim≥5 had a `[-ndim:]` slice bug that produced 2560 wrong slices on a 5D `TZCYX` file — see arc42 if you touch this

See [Runtime View](docs/06_runtime_view.md#multi-dimensional-image-loading) for workflow.

### Patterns introduced in v0.9.0 (read before touching these areas)

| Area | Pattern | Why |
|------|---------|-----|
| Pan / zoom-to-cursor in scroll area | Use `event.globalPosition()` for pan; derive post-zoom offset from `viewport().width()`, not `self.width()` | Widget-local coords absorb half the pan delta as the widget shifts; `self.width()` is stale during zoom-out before layout settles. See [Pan + Zoom Reference Frames](docs/08_crosscutting_concepts.md#pan--zoom-reference-frames). |
| Dark mode contrast | No hardcoded `background:` / `color:` in widget `setStyleSheet(...)` | Hardcoded greys override `soft_dark_stylesheet.py` and punch bright boxes into the sidebar. Add a global rule first, then write the widget. See [No Hardcoded Colors Rule](docs/08_crosscutting_concepts.md#dark-mode--no-hardcoded-colors-rule). |
| DINO review state | `image_label.temp_annotations` is a single field, **not** per-image — must be re-synced from `dino_batch_results` on every image/slice switch via `_refresh_dino_temp_for_current` | Otherwise the first image's masks bleed onto every subsequent slice during navigation. See [DINO Temp Annotations](docs/08_crosscutting_concepts.md#dino-temp-annotations--single-field-many-images). |
| DINO batch over stacks | Use `_collect_dino_batch_work_items()` to flatten regular images + every loaded slice; don't iterate `self.all_images` directly | Multi-dim images appear in `all_images` as a single entry — slices live in `self.image_slices[base_name]` and were silently skipped. |
| DINO Enter/Escape during review | Application-wide `DINOReviewEventFilter`, gated on pending temp_annotations + no modal + no text input | `QListWidget` consumes Enter for `itemActivated` before `ImageLabel.keyPressEvent` sees it. See [ADR-015](docs/09_architecture_decisions.md#adr-015-application-wide-event-filter-for-dino-review-shortcuts). |
| Auto-accept dropdown | Honored by **both** `run_dino_detection_single` and `run_dino_detection_batch` | Easy to forget in the single path because the combo is labeled "batch". |
| GPU model unload | `model.cpu()` → `gc.collect()` → `torch.cuda.empty_cache()` + `ipc_collect()` + `synchronize()` — full reclaim requires app restart due to per-process CUDA context | Setting refs to None alone leaves circular refs pinned and shows zero Task Manager drop. See [Releasing Model GPU Memory](docs/08_crosscutting_concepts.md#releasing-model-gpu-memory). |
| Export image-path lookup | Exact-key match first, substring fallback only | `"bee.jpg" in "honeybee.jpg"` is True — substring-only matching writes the wrong file. See [Export Format Filename Matching](docs/08_crosscutting_concepts.md#export-format-filename-matching). |
| F2 / global shortcuts | Use `QShortcut` with `Qt.ShortcutContext.ApplicationShortcut`, not `keyPressEvent` | `QTableWidget` consumes F2 for in-cell edit before it bubbles up. |
| Canvas ↔ list selection sync | Canvas selection (idle-mode click/Shift/rubber-band) drives the annotation list via `apply_canvas_selection`; mirror the list with `blockSignals(True/False)` and match annotations by **value-equality**, never identity | PyQt round-trips `UserRole` dicts as copies and `image_label.annotations` is a deepcopy, so identity is never stable; un-blocked `setSelected` recurses through `update_highlighted_annotations`. Multi-select uses **Shift** (Ctrl stays pan). See [ADR-022](docs/09_architecture_decisions.md#adr-022-canvas-mask-selection-unified-with-the-annotation-list). |
| Selection rendering | Don't recolour a selected mask. Keep its class colour; draw a class-colour-independent overlay (dashed `_SELECTION_COLOR` blue bounding box + bright handle squares at corners/edge-midpoints, OGP-style) in a final pass — `_draw_selection_overlay`. For a single selected shape those handles are draggable (resize/move, any shape). Default class colours come from `core/constants.py::default_class_color` (red last, muted) | Red selection was invisible on a red-class mask; a thin dashed outline alone was too faint; the handles carry the visibility. See ADR-022 amendment. |
| Shape editing (#40) | Direct manipulation on the selection handles for **any** single selected shape (`_single_selected_shape()` — most shapes are `"segmentation"`, not `"bbox"`; gating on a bbox key made it unreachable). `_begin_shape_edit` records `kind`: a `"seg"` polygon **scales** its vertices (`_scale_segmentation`) / translates them; a `"bbox"` edits `[x,y,w,h]`. `_draw_selection_overlay` + `_bbox_handle_at` share `_bbox_handle_points` (visual == grab); resize anchors the opposite side (`_resize_bbox`); interior drag moves, **drag-gated**. `_sync_bbox_key` keeps an imported bbox consistent. Clamp + `bboxEditCommitted` → `commit_bbox_edit` on release; Esc reverts | Handles drawn since #75 were visual-only; dispatch sits before the rubber-band branch but stays gated on `_is_select_mode()`. Names keep the `bbox_edit` prefix = "edit via the bounding-box handles". See [ADR-023](docs/09_architecture_decisions.md). |
| Bounds enforcement (#32/#36) | No commit may persist coords outside the image. **Clamp** manual edits (`clamp_segmentation`/`clamp_bbox`, per-coordinate, count-preserving) at edit commit; **clip** augmented polygons (`clip_polygon_to_bounds`, shapely intersection, may drop → `None`, augmenter must `continue`). Drawn shapes already clip in `finish_polygon`/`finish_rectangle` | Clamp keeps vertex correspondence mid-edit; clip is geometrically correct for batch augmentation. See [ADR-024](docs/09_architecture_decisions.md). |
| Annotations table + Detail % (#24) | The Annotations panel is a **`QTableWidget`** (ID \| Class \| Area \| Detail %), not a list — col 0's UserRole holds the annotation (the #75 value-equality marker). Selection-mirror uses **`setRangeSelected`** (additive); `selectRow()` replaces in `ExtendedSelection`. Per-row Detail % spinbox → `on_detail_pct_changed` resolves the live obj (`_live_annotation`), simplifies from a lazy-captured `segmentation_raw` (`simplify_polygon`, 100=raw), refreshes Area+UserRole in place, saves. Connect `valueChanged` **after** the initial `setValue` so building the table doesn't fire it | Re-homing the selection bridge onto a table is the risk; reversibility needs the raw preserved. See [ADR-025](docs/09_architecture_decisions.md). |
| Undo/redo (ADR-026) | **Snapshot** the whole per-image annotation dict, don't replay commands — restoring a deep copy sidesteps value-equality/renumber/`segmentation_raw`. `AnnotationController.record_history()` is the choke-point, called **before** each synchronous mutation (finish poly/rect, delete, merge, change-class, eraser, SAM/DINO accept); **don't** hook `save_current_annotations` (also fires on navigation, runs after mutation). Deferred gestures (bbox drag, paint, **polygon vertex edit**) capture the baseline at **start** via `editBaselineRequested` and push at commit (`commit_edit_baseline` via `commit_bbox_edit`/`commit_polygon_edit`/batch-saved); a deep-equal dedup in `AnnotationHistory.record` drops aborted ones. Vertex edit also gained a save-discipline fix — `commit_polygon_edit` now calls `save_current_annotations` and Esc reverts the in-place drags. Detail-% drags coalesce to one entry. Ctrl+Z/Y are `ApplicationShortcut`s; `_undo_blocked` no-ops during load/modal/text-focus/in-flight gesture | Delete/merge confirmation+success dialogs were **removed** (undo is the net); merge always deletes originals. See [ADR-026](docs/09_architecture_decisions.md#adr-026-snapshot-based-undoredo-for-annotation-edits). |
| Tool activation + Esc | **All six tools (manual + SAM) funnel through `ImageAnnotator.activate_tool(name)`** — the only place `current_tool`, `sam_*_active`, and button checks change, so they can't drift and a SAM tool can't be active with a manual one. Keep `tool_group` non-exclusive (need click-to-toggle-off); `activate_tool` unchecks the others (block-signals around `setChecked`). Esc cancels the in-progress shape **and** emits `selectModeRequested` → `activate_tool(None)`, so Esc always returns to selection mode | SAM toggles used to write state ad-hoc; the group was non-exclusive. See [Tool Activation](docs/08_crosscutting_concepts.md#tool-activation--one-choke-point-mutually-exclusive). |
| Keypoint / pose (#35) | A **pose instance** is `{"keypoints":[x,y,v]*K, "num_keypoints", "bbox", category…}` with **no `segmentation` key** — that absence is the discriminator routing area (→bbox), Detail-% (disabled), and the `draw_annotations` `elif "keypoints"` branch (placed **before** `elif "bbox"`). The per-class **schema** (`{names,skeleton,flip_idx}`) lives in `ImageAnnotator.keypoint_schemas` (one per class, COCO rule), embedded on each `.iap` `classes[]` entry (auto-round-trips via `convert_to_serializable`; malformed dropped on load). `KeypointTool` accepts **both mouse buttons** (right=occluded v1) so it short-circuits the left-only press dispatch like `sam_points`. Point editing reuses the #40 handle path, **not** double-click: a single-point drag (`editing_keypoint`) and a visibility-toggle right-click both commit via `keypointEditCommitted`→`commit_keypoint_edit`; the instance box uses a new edit `kind="kpt"` on the existing `bbox_edit` machinery that transforms the whole pose (`_scale_keypoints`/`_translate_keypoints`, skipping v=0 points so they stay at `(0,0)`) and commits via the regular `bboxEditCommitted`→`commit_bbox_edit`. **Merge and cross-schema change-class are blocked** for instances (merge would silently delete them). K is locked once instances exist. Pure validation in `core/keypoint_schema.py`; clamp via `utils.clamp_keypoints`. See [ADR-029](docs/09_architecture_decisions.md#adr-029-keypoint--pose-annotation--per-class-schema-coco-instance-model-3-state-visibility). **PR-2 (COCO/YOLO-pose export/import) done**: COCO category `skeleton` is 1-based (spec), `flip_idx` is an app-only extension kept 0-based; `create_coco_annotation`/`export_yolo_v5plus`'s writer both check `"keypoints" in ann` **before** segmentation/bbox; YOLO-pose has **one dataset-global `kpt_shape`/`flip_idx`**, so export refuses (`ValueError`→`QMessageBox`) mixed-K or pose+non-pose exports via `_pose_export_check`; all `io.import_formats` entry points uniformly return a 3-tuple with a recovered-schemas dict (`{}` if none); the `io_controller` import rebuild (`_rebuild_imported_annotation`) builds a **fully separate dict shape** for keypoint vs. plain annotations — never a `None`-valued `segmentation`/`type` key, since several existence-only `"segmentation" in annotation` checks elsewhere (not None-guarded) would misfire. **PR-3 (YOLO-pose training + prediction) done**: `train_model` infers the intended task from the prepared dataset yaml (`kpt_shape` present → `"pose"`) and raises `ValueError` pre-flight if the loaded model's `.task` disagrees, rather than failing deep inside Ultralytics; the trained-model schema round-trips two-tier through `_register_trained_model`/`load_prediction_model` — a rich embedded `keypoint_schema` when every trained class shares one identical schema (in-app trained models), else generic `kp0..kpK-1` names reconstructed from bare `kpt_shape`/`flip_idx` (external `.pt` models); `predict()` no longer hardcodes `task='segment'`; `process_yolo_results` branches on `model.task == "pose"` and builds `{"keypoints","num_keypoints","bbox",…}` temp dicts (deliberately no `segmentation` key) with **every point forced to v=2** — Ultralytics gives no true occlusion signal, a documented simplification — seeding `keypoint_schemas["Temp-<class>"]` from `prediction_keypoint_schema`; and `accept_visible_temp_classes`/`reject_visible_temp_classes` carry that `Temp-<class>` schema over to the accepted class name (warn + keep the existing schema on a K mismatch instead of overwriting) or pop the orphaned entry on reject. |

## Development Workflow

**CRITICAL: Always use feature branches — NEVER commit directly to master.**

| Step | Action | Notes |
|------|--------|-------|
| 1 | Create branch: `git checkout -b feature/short-description` | Before any changes |
| 2 | Implement feature | Follow patterns below |
| 3 | Run manual tests | See Testing Checklist |
| 4 | Update arc42 docs if behavior changed | `docs/` — see Documentation section |
| 5 | **Run senior reviewer agent** | `.claude/agents/senior-reviewer.md` — mandatory quality gate before every PR |
| 6 | Commit: `feat: Description` or `fix: Description` | Clear, descriptive messages |
| 7 | Push & create PR | `git push origin feature/branch` |

### Testing Checklist

Before opening a PR, verify at minimum:

1. **Smoke tests pass** — `pytest tests/integration/test_smoke.py -v`. This includes the AST-based `test_annotator_window_inline_imports_are_resolvable` which catches stale relative imports inside function bodies after any module move (see ADR-016). A launch that "looks clean" is NOT sufficient — inline imports fail only when the function is called at runtime.
2. **Launch the app** — no import errors, main window renders
3. **Golden path** — perform the new feature's primary workflow end-to-end
4. **Edge cases** — empty state, cancel/escape, large images, missing model files
5. **Dark mode** — toggle and check rendering of new UI elements
6. **Save/load roundtrip** — if the feature touches `.iap` project files, save, close, reopen, verify state restored
7. **Adjacent features** — verify no regression in SAM, annotation tools, export formats
8. **Inference features** — if touching `sam_utils.py` or `dino_utils.py`, verify the model loads end-to-end (no silent load failure), returns masks/boxes, and the UI stays responsive during inference (timers, redraws, progress dialog cancels keep firing — see ADR-013)

### arc42 Documentation Update Rules

When a change affects behavior documented in `docs/`, update the docs in the same PR:

| Change Type | Update Target |
|-------------|---------------|
| New component/module | `05_building_block_view.md` — add to component table |
| Changed runtime behavior | `06_runtime_view.md` — update workflow description |
| New UI pattern or concept | `08_crosscutting_concepts.md` — add section |
| Architecture decision | `09_architecture_decisions.md` — record ADR |
| New domain term | `12_glossary.md` — add definition |

### Quality Gate — Senior Reviewer

Before every PR, run the senior reviewer agent (`.claude/agents/senior-reviewer.md`).

This is **mandatory** — the agent performs an independent end-of-implementation review:
- Reads the actual diff, not commit messages
- Ranks issues P0 (blocks merge) / P1 (should fix) / P2 (nit)
- Checks CLAUDE.md compliance (feature branches, coordinate systems, `is_loading_project` guards, DINO config persistence, in-process inference re-entrancy guards)

**Run it in the foreground** — never `run_in_background: true`. The review is a blocking quality gate: the next steps (address P0s, push, open PR) depend on its findings. Launch the agent and wait for the result before doing anything else, then iterate until clean.

Address all P0s before merging. Address P1s unless there's explicit justification.

## Known Constraints

- No type hints (gradual addition encouraged)
- Print statements instead of logging (acceptable)
- Absolute paths in projects (not portable)
- SAM 2 large crashes on limited RAM
- YOLO training not supported for multi-dimensional images

See [Risks and Technical Debt](docs/11_risks_and_technical_debt.md) for full list.

## Keyboard Shortcuts

| Global | Action |
|--------|--------|
| Ctrl+N/O/S | New/Open/Save Project |
| Ctrl+Z / Ctrl+Y (or Ctrl+Shift+Z) | Undo / redo annotation edit (ADR-026) |
| Ctrl+Shift+= / Ctrl+Shift+- | UI font bigger/smaller (8-24pt, persisted via QSettings) |
| Ctrl+Shift+0 | Reset UI font size |
| F1 | Help |

| Canvas | Action |
|--------|--------|
| Ctrl+Wheel | Zoom |
| Ctrl+Drag | Pan |
| Click / Shift+Click (no tool) | Select / toggle mask |
| Drag / Shift+Drag (no tool) | Rubber-band select / add |
| Drag handle / inside (one shape selected) | Resize (scale) / move the shape |
| Double-click | Vertex-edit mode |
| Delete | Delete selected mask(s) — instant, undoable (no confirm dialog) |
| Enter | Finish/Accept (keypoint tool: finish pose early, padding unplaced points v=0) |
| Esc | Cancel in-progress shape **and** return to selection mode (deactivates the tool) |
| Left / Right-click (keypoint tool) | Place next keypoint visible / occluded (#35) |
| Backspace (keypoint tool) | Remove the last placed keypoint |
| Right-click a selected pose's point | Toggle its visibility (visible ↔ occluded) |
| Up/Down | Navigate slices |
| -/= | Brush size |

## Quick Tips

- SAM models cache in working directory (Ultralytics)
- Recommend SAM 2 tiny/small (avoid large)
- Polygon area uses shoelace formula (utils.py)
- Export formats copy images to output directory
- Dark mode changes annotation colors for visibility
- Snake game is hidden Easter egg 🐍
