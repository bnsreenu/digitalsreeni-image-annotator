# Building Block View

## Level 1: System Overview

```
┌─────────────────────────────────────────────┐
│   DigitalSreeni Image Annotator            │
│                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐ │
│  │   GUI    │  │  SAM 2   │  │  YOLO    │ │
│  │ (PyQt6)  │  │(Ultraly.)│  │ Trainer  │ │
│  └──────────┘  └──────────┘  └──────────┘ │
│                                             │
│  ┌──────────────────────────────────────┐  │
│  │   Image Processing                   │  │
│  │   (NumPy, OpenCV, Shapely)          │  │
│  └──────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
         │                  │
         ▼                  ▼
   File System        SAM Model Cache
```

## Level 2: Main Components

### Core Application

```
src/digitalsreeni_image_annotator/
├── main.py                        # Entry point, initializes QApplication
	├── annotator_window.py            # ImageAnnotator - main window orchestrator
	├── app_settings.py                # QSettings-backed UI prefs (font size, dark mode) — ADR-020
	├── utils.py                       # Cross-cutting utilities
	├── core/                          # Constants, annotation utils, image utils
	│   ├── constants.py
	│   ├── annotation_utils.py
	│   ├── slice_cache.py             # Lazy multi-dim slice materialisation + bounded LRU (ADR-036, #45)
	│   ├── video_handler.py           # cv2 video decode; frames as lazy slices (ADR-037, #47)
	│   └── torch_utils.py             # Shared torch device resolution + CPU fallback (#57)
	├── widgets/
	│   ├── image_label.py             # ImageLabel - canvas widget; dispatcher
	│   ├── canvas_renderer.py         # CanvasRenderer - painting/overlays (ADR-034)
	│   ├── edit_gestures.py           # EditGestures + pure fns - #40/#35 handles (ADR-034)
	│   ├── canvas_context.py          # CanvasContext - narrow read view (ADR-018)
	│   ├── video_timeline.py          # VideoTimeline scrub bar + frame markers (#48)
	│   └── tools/                     # Per-tool handlers (ADR-019)
	│       ├── base.py                # ToolHandler base
	│       ├── rectangle_tool.py
	│       ├── polygon_tool.py
	│       ├── paint_tool.py
	│       ├── eraser_tool.py
	│       └── keypoint_tool.py       # KeypointTool - pose placement (ADR-029, #35)
	├── controllers/                   # Project/Image/SAM/DINO/YOLO/Annotation/Class
	├── inference/                     # sam_utils.py, dino_utils.py, sam3_utils.py
	│   ├── sam_utils.py
	│   ├── dino_utils.py
	│   └── sam3_utils.py              # SAM3Utils - text-prompt segmentation (ADR-038/039, #50)
	├── io/                            # export_formats.py, import_formats.py
	│   ├── export_formats.py
	│   └── import_formats.py
	├── ui/                            # menu_bar, sidebar, theme, stylesheets
	│   ├── default_stylesheet.py
	│   └── soft_dark_stylesheet.py
	└── dialogs/                       # Standalone tool dialogs
```

### ImageAnnotator (annotator_window.py)

**Responsibility**: Main application window and state management

**Key Attributes**:
```python
all_annotations: dict[str, list]    # {filename: [annotation_dicts]}
all_images: list[str]               # List of loaded image filenames
class_mapping: dict[str, int]       # {class_name: class_id}
keypoint_schemas: dict[str, dict]   # {class_name: {names, skeleton, flip_idx}} pose classes (ADR-029, #35)
image_paths: dict[str, str]         # {filename: absolute_path}
image_dimensions: dict              # Multi-dimensional image metadata
image_slices: dict                  # Extracted slices from stacks
current_slice: str                  # Currently displayed slice
```

**Key Methods**:
- `save_project()`: Save project state to JSON
- `load_project_data()`: Load project from JSON
- `add_annotation()`: Add annotation to current image
- `export_annotations()`: Export to various formats
- `import_annotations()`: Import from COCO/YOLO

### ImageLabel (widgets/image_label.py)

**Responsibility**: Canvas widget — image display, navigation
(zoom/pan), event dispatch, and state ownership. Committed-annotation
rendering, SAM/DINO overlays and the selection overlay are delegated to
`CanvasRenderer` (`widgets/canvas_renderer.py`); the direct-manipulation
edit-gesture state machine (#40 bbox/segmentation handles, #35 keypoint
edits) is delegated to `EditGestures` + pure functions
(`widgets/edit_gestures.py`) — both via thin one-line delegates that keep
every existing `ImageLabel` name working (ADR-034). Per-tool mouse/key
handling lives in `widgets/tools/*` (see ADR-019); ImageLabel dispatches
events to the active handler. `paintEvent` orchestration and polygon
edit mode (modal) stay on ImageLabel.

**Key Attributes**:
```python
current_tool: str                   # Active annotation tool (route via set_active_tool)
zoom_factor: float                  # Current zoom level
annotations: dict                   # Displayed annotations
class_colors: dict                  # Class color mapping
temp_paint_mask: np.ndarray         # In-progress paint stroke (owned by PaintBrushTool)
temp_eraser_mask: np.ndarray        # In-progress eraser stroke (owned by EraserTool)
current_rectangle: list             # In-progress rectangle (owned by RectangleTool)
current_annotation: list            # In-progress polygon points (owned by PolygonTool)
sam_positive_points: list           # SAM positive points
sam_negative_points: list           # SAM negative points
editing_polygon: dict | None        # Polygon being edited (modal sub-state)
_tools: dict[str, ToolHandler]      # Per-tool handlers
_ctx: CanvasContext                 # Narrow read view of main-window state (ADR-018)
```

**Key Methods**:
- `mousePressEvent()` / `mouseMoveEvent()` / `mouseReleaseEvent()` /
  `mouseDoubleClickEvent()`: Ctrl-modifier pan/zoom branches first,
  then SAM/edit-mode branches, then dispatch to
  `active_tool_handler.on_mouse_X()`.
- `keyPressEvent()`: Enter / Escape / Delete / brush-size keys. Modal
  branches (DINO temp, sam_points, sam_box, editing_polygon)
  consume first; otherwise routed to `handler.on_enter()` /
  `on_escape()`.
- `paintEvent()`: image → committed annotations → editing polygon →
  SAM overlays → all tool handlers' `paint_overlay()` → tool-size
  indicator → DINO temp annotations.
- `set_active_tool(name)`: switches `current_tool` and gives the
  previous handler a chance to clean up via `deactivate()`.
- `check_unsaved_changes()`: iterates handlers' `has_unsaved_state()`
  and prompts the user.

**Communication**: emits ~20 Qt signals connected to controller slots
in `ImageAnnotator._connect_image_label_signals` (ADR-018). Reads
main-window state through `CanvasContext`.

### SAMUtils (sam_utils.py)

**Responsibility**: SAM model loading and inference (in-process).

**Key state** (on the `SAMUtils` instance):
- `sam_models: dict` — available SAM model variants (class-level, exposed for the UI dropdown)
- `current_sam_model: str | None` — name of the currently loaded model; `None` if unloaded
- `_model: ultralytics.SAM | None` — the loaded model object (private)

**Key public methods**:
- `change_sam_model(model_name)` — load a SAM model. Blocks the calling thread (with the UI's event loop pumping) until weights are downloaded and the model is in memory. Raises on load failure.
- `apply_sam_points(image, positive_points, negative_points)` — point-prompted segmentation.
- `apply_sam_prediction(image, bbox)` — single bbox-prompted segmentation.
- `apply_sam_predictions_batch(image, bboxes)` — multi-bbox segmentation in one model call (used by the DINO pipeline).
- `unload()` — drop the cached model and free GPU/CPU memory. Wired to the Tools → "Unload AI Models" menu entry.

**Module-level helpers** (not class methods):
- `_qimage_to_numpy(qimage)` — convert a `QImage` to an owned numpy array (always copies; see ADR-013 on lifetime safety).
- `_mask_to_polygon(mask)` — convert a SAM mask tensor into polygon contour vertices.
- `_run_sync(fn, *args, **kwargs)` — run `fn` on a worker `QThread`, pump the calling thread's event loop until done, re-raise any exception. Serialises concurrent calls via the `_inference_in_flight` flag; re-entry raises `InferenceBusyError`.

Inference runs in-process on a background `QThread`. `SAMUtils._run_sync()`
spawns the thread, pumps the caller's event loop until done, and returns
the result — keeping the API synchronous-looking from call sites while
the UI stays responsive. Model objects (Ultralytics `SAM`) live on the
`SAMUtils` singleton and persist across calls. See
[ADR-013](09_architecture_decisions.md#adr-013-in-process-inference-with-qthread-wrapping).
The earlier subprocess approach is documented as
[ADR-011](09_architecture_decisions.md#adr-011-run-torch-based-workers-in-isolated-subprocesses)
(Superseded).

### SAM Fine-Tuning Subsystem (`training/`)

Lets users fine-tune SAM 2 / 2.1 on their own annotations, since
Ultralytics ships no SAM trainer (ADR-021). Distinct from `inference/`
because it is *training*, not inference.

| Module | Responsibility |
|--------|----------------|
| `training/sam_trainer.py` | `SAMFineTuner` — custom decoder (optionally encoder) fine-tuning loop reusing `SAM2Predictor.get_im_features` / `prompt_inference` under autograd, focal+dice loss, AdamW. Per-epoch train/val loss + LR logging, `LambdaLR` warmup→cosine schedule, best-val checkpoint save+reload-verify (ADR-028). Also geometry helpers (`polygon_to_mask`, `mask_to_xyxy`, `mask_to_point`), `make_custom_filename`, `list_custom_models`, and the `SampleGroup` lazy-rasterising dataset item (carries a `name` for the split). |
| `training/sam_dataset.py` | `build_groups_from_project` (live `all_annotations`) and `build_groups_from_folder` (prepared dataset) → `list[SampleGroup]`, mirroring `export_yolo_v5plus` image resolution. `split_groups(groups, train_pct)` deterministically holds out a validation set keyed by **source, not by image** (ADR-044): `SampleGroup.name` is run through `core/dataset_split.derive_groups`, so a stack's slices and a video's frames go to one side together — which matters more here than for YOLO, because val loss drives early stopping. `build_groups_from_folder` ext-strips the manifest path into `name` so the folder path groups like the project path. |
| `training/lr_schedule.py` | Pure `warmup_cosine_lambda(total_steps, warmup_frac, floor)` → `step→multiplier` for SAM's `LambdaLR` (ADR-028); unit-tested without torch. |
| `training/early_stop.py` | Pure `EarlyStopper(patience)` — tracks best/best-epoch and patience-based stop for the SAM loop (ADR-028); unit-tested without torch. |
| `training/mlflow_tracker.py` | Always-on MLflow experiment tracking (ADR-027). `MLflowTracker` (no enable/disable; tracking errors never abort training; on start fires `set_run_url_callback` with the run's UI deep link), `_NullTracker` no-op for trainer calls with no tracker, `resolve_tracking_uri()` (override → `<project>/mlruns` → `<cwd>/mlruns`), `to_mlflow_uri()` (Windows-safe `file://`), `run_ui_url()`, `start_mlflow_ui_server()` / `launch_mlflow_ui()`. SAM logs through it explicitly; YOLO uses Ultralytics' native MLflow callback. |
| `io/export_formats.py::export_sam_dataset` | Writes `images/` + `manifest.json` (authoritative bbox/segmentation specs) for an inspectable, re-trainable on-disk dataset. |

Fine-tuned checkpoints save as `{"model": state_dict}` and reload
through the unchanged `SAM(path)` inference path; `SAMUtils` gains a
`custom_models` registry so they appear in the SAM selector alongside
the eight built-ins.

### DINO Subsystem (Grounding DINO + SAM pipeline)

LLM-assisted detection: the user gives free-form text phrases per class,
Grounding DINO produces bounding boxes, and SAM 2 refines them into
segmentation masks.

| Module | Responsibility |
|--------|----------------|
| `dino_utils.py` | `DINOUtils` — in-process Grounding DINO wrapper. Resolves model paths via `models_base_dir()`, loads `transformers.AutoModelForZeroShotObjectDetection` lazily on first use, caches it across calls, runs inference on a worker `QThread` (same `_run_sync` pattern as `SAMUtils`). |
| `dino_phrase_editor.py` | Two widgets: `ClassThresholdTable` (per-class box/text/NMS thresholds) and `PhraseEditorPanel` (per-class phrase list). These widgets are the **single source of truth** for phrases and thresholds; project save/load reads/writes them via `get_all_phrases()` / `set_phrases()` and `get_thresholds_dict()` / `set_thresholds()`. Selection follows the **top class list** (single source of truth): `ClassController.on_class_selected` calls `ClassThresholdTable.select_class_by_name(...)`, whose `itemSelectionChanged` cascades to the phrase panel — so picking a class up top retargets Add Phrase, not only clicking the threshold-grid row (#63). Both widgets are **keyed by class name**, so every class-roster mutation must sync them — see [Class Name Is a Primary Key](08_crosscutting_concepts.md#class-name-is-a-primary-key--sync-every-registry-on-rename-63) for the full registry list and the rename-collision rule. |
| `dino_merge_dialog.py` | Standalone dialog: merges accumulated DINO+SAM annotations across images into a training-ready COCO JSON. |

**Detection call signature** (in-process):
```python
DINOUtils().detect(
    qimage,                                # PyQt6.QtGui.QImage
    class_configs=[
        {"name": "drone", "phrases": ["drone", "quadcopter"],
         "box_thr": 0.10, "txt_thr": 0.25, "nms_thr": 0.50},
        ...
    ],
    model_name="grounding-dino-base",      # or custom_model_path=...
)
```

**Detection return value**:
```python
[
    {"class_name": "drone", "bbox": [x1, y1, x2, y2], "score": 0.93, "label": "drone"},
    ...
]
# or [] if no boxes survived filtering, or None on error
```

DINO's xyxy boxes feed directly into `SAMUtils.apply_sam_predictions_batch()`,
which returns segmentation polygons (xywh bbox is derived from the polygon at
export time — see [Cross-cutting Concepts](08_crosscutting_concepts.md)).

## Level 3: Controllers

Eight `QObject` controllers plus an `io_controller` helper module
carve `ImageAnnotator` into single-responsibility owners that the
orchestrator delegates to. Each `QObject` controller holds `self.mw
= main_window` and owns one slice of behaviour; the
`io_controller` is a thin module of UI-wrapper functions around the
pure `io/` formatters and does not need to hold state. The
orchestrator keeps pass-through methods so external call sites
(menus, signal wiring, the test harness) don't need to reach into
the controller graph.

| Controller | Responsibility |
|------------|----------------|
| `ProjectController` | `.iap` save/load, auto-save, backup/restore, missing-image prompts, window-title sync. Owns the `is_loading_project` autosave guard (load/save round-trip safety, v0.8.12). |
| `ImageController` | Open / load / switch images and slices. TIFF + CZI loaders (with `imagecodecs` codec-error handling — #56), the multi-dim `DimensionDialog`, the `[-ndim:]` axis-slice bug fix from the v0.9.0 era. Multi-dim slices are now materialised **lazily** via `core/slice_cache.py` (`create_slices` builds names + a `SliceProvider`, QImages decode on demand through a shared bounded LRU — ADR-036 / #45). Videos (`load_video`, `mw.video_handlers`) reuse the same lazy contract: frames are `LazySliceList` slices backed by a `VideoSliceProvider` over `core/video_handler.py::VideoHandler` (ADR-037 / #47). Image-list annotation-status filter (`image_has_annotations`, `apply_image_filter` — #27), alphabetical/grouped sort (`sort_image_list` — #60/#43), per-image named groups (`set_image_group`, `_populate_group_combo` — #43) and derived status badges (`refresh_image_status_icons`, painted-pixmap `QIcon` cache rebuilt on theme flip via `on_theme_changed` — #43). |
| `AnnotationController` | Annotation CRUD, list sorting, highlight, edit-mode entry/exit, `finish_polygon`, `finish_rectangle`, `replace_annotations` (eraser path). Validates writes before mutating `all_annotations`. |
| `ClassController` | Class add / delete / rename / colour / visibility. `update_slice_list_colors`, `is_class_visible`. |
| `SAMController` | SAM box/points tool lifecycle, debounce timer, `_sam_inference_in_flight` re-entrancy guard (ADR-013), model picker. |
| `DINOController` | Single + batch detection, batch review navigation, temp-annotation accept/reject, custom-model browse, `DINOReviewEventFilter` ownership (ADR-015). **Dual-backend (ADR-039):** `_run_text_detection` routes to either the Grounding-DINO two-stage path OR SAM 3's one-stage `SAM3Utils.detect_text` (the "SAM 3 (text prompt)" picker entry); both feed the same review/batch/accept pipeline. |
| `SAM3Utils` | inference/sam3_utils.py — in-process Ultralytics `SAM3SemanticPredictor` wrapper (text→masks) + `track()` video propagation via `SAM3VideoPredictor` (ADR-040). Reuses `SAMUtils`'s `_run_sync`/`_qimage_to_numpy`/`_mask_to_polygon` + shared in-flight flag; gated `sam3.pt` (never auto-downloaded). ADR-038/039/040, #50/#51. |
| `TrackingController` | controllers/tracking_controller.py — SAM 3 video object tracking (#51, ADR-040). `can_track`/`run_tracking`/`_commit_tracked_result` (mirrors `_commit_dino_results`)/`undo_last_track`. Confident frames commit as `source:"sam3-track"` with a `track_run` id; uncertain frames route to `dino_batch_results` for the existing review pipeline. |
| `YOLOController` | Training menu, `TrainingThread`, prediction dialog, result processing. Surfaces the run's MLflow deep link (`_on_mlflow_run_url`, mirrors SAM) and reports the saved `best.pt` path on completion. |
| `ClipboardController` | controllers/clipboard_controller.py — in-app annotation clipboard (#66). App-level, so it survives image / slice / frame / project switches. Deep-copies in and out (value-equality is the only stable identity, ADR-022), clamps into the target's bounds (ADR-024), resolves missing classes once per distinct name, and refuses a pose whose K does not match the target class's schema (ADR-029). One `record_history` per paste. |
| `QCController` | controllers/qc_controller.py — GUI adapter for the annotation audit (#70). Gathers annotations, image sizes and class names, shows `AnnotationQCDialog`, and applies repairs through `record_history` as **one** undo entry. The rules themselves live Qt-free in `core/annotation_qc.py` so the CLI reuses them. |
| `ReviewController` | controllers/review_controller.py — model-vs-ground-truth review scoring (#71). Runs the prediction model across the project and scores each image by disagreement (annotated) or uncertainty (unannotated). Never mutates an annotation: predictions are extracted for scoring **without** going through `process_yolo_results`, which writes into the review overlay as a side effect. |
| `CurationController` | controllers/curation_controller.py — embedding-based near-duplicate detection (#72). Embeds every image *including slices and video frames*, clusters, and selects a cluster in the image list. **Has no delete path at all**, by design. |
| `SegmentEverythingController` | controllers/segment_everything_controller.py — unprompted SAM proposals into the **existing** review overlay (#69). A third producer alongside DINO and SAM 3, not a second review mechanic (ADR-015). Applies the `core/mask_filters` noise limits before anything reaches the canvas. |
| `TrainingController` | controllers/training_controller.py — one entry point for all training (#73, ADR-042). Dispatches the unified `TrainDialog` to the existing trainers and performs the mechanics implicitly (prepare, YAML, load, save, refresh). Orchestration only; the trainers are untouched. |
| `ModelRegistryController` | controllers/model_registry_controller.py — post-training lifecycle (#74). Registers, copies weights into `<project>/models/` with a JSON sidecar, feeds the results panel, and offers "try it now" **for YOLO runs only** (`predict_single_image` routes to the YOLO trainer regardless of what was trained; a fine-tuned SAM checkpoint is used interactively via SAM-box/SAM-points instead). Registers nothing on a failed or stopped run, and nothing at all while `is_loading_project` is set. Drops the review scores (#71), which were computed with the previous model. |
| `SAMTrainController` | SAM fine-tuning menu, GPU gate, `SAMTrainingThread`, config dialog, registers fine-tuned checkpoints into the SAM selector (ADR-021). |
| `io_controller` *(module-level functions, not a class)* | Thin UI wrappers around the pure `io/export_formats.py` and `io/import_formats.py` modules. |

Communication: `ImageLabel` does not import controllers directly —
it emits Qt signals (ADR-018) that the orchestrator connects to
controller slots in `_connect_image_label_signals()`.

## Level 3: The Qt-Free Core (shared with the CLI)

These modules are imported by both the GUI and `sreeni-cli`, and **must not import Qt** at module
level. A subprocess test enforces it (ADR-041); see
[Deployment View](07_deployment_view.md#72-the-qt-free-boundary) for why the guard has to run
out-of-process.

| Module | Responsibility |
|---|---|
| `core/annotation_types.py` | `TypedDict`s for the annotation shapes plus `is_pose` / `is_polygon` / `is_bbox_only` (#78). `PoseAnnotation` declares **no** `segmentation` key — the type expresses the ADR-029 discriminator. |
| `core/annotation_qc.py` | The QC rule engine (#70): geometry, redundancy, statistics, hygiene and pose rules, plus the unambiguous repairs. Powers both the dialog and `sreeni-cli validate`. |
| `core/disagreement.py` | Model-vs-ground-truth scoring (#71). Greedy matching with a swap-improvement pass — no scipy; see the module docstring for why. |
| `core/similarity.py` | Cosine similarity, threshold-based connected-component clustering, medoid representative, outliers (#72). Model-free: it takes plain vectors, so the embedding backend can be swapped without touching it. |
| `core/task_inference.py` | Derives the training task from the annotations and produces the pre-flight blockers (#73). One source of truth shared with `train_model`'s YAML-based inference. |
| `core/model_sidecar.py` | Build / read / locate the trained-model JSON sidecar, and the non-colliding weights filename (#74). |
| `core/project_io.py` | Read an `.iap` without the GUI (#76). **No write path at all** — the CLI must never autosave into a project it was asked to read. |
| `core/mask_filters.py` | Polygon IoU and the noise limits for unprompted mask proposals (#69). |
| `core/onion.py` | Onion-skin neighbour selection, the content choice (annotations / image / both) and the settings clamps (#67). Ends never wrap. |
| `core/image_size.py` | Image dimensions via a Pillow header read (#76) — what replaced `QImage` in the export layer. |
| `core/dataset_split.py` | Group-aware train/val splitting (#81, ADR-044): `derive_groups` (exact from `image_slices`, name-prefix fallback), `plan_split` (whole groups, locally-optimal size, plus the `fell_back` flag), `split_warning` (the text the GUI shows — here rather than on the controller so it stays Qt-free) and `assign_train_val`, which moved here from `io/export_formats.py` and stays re-exported there. Deliberately imports nothing from `core/slice_cache`, which reaches `QImage`. |

## Level 3: CLI

| Module | Responsibility |
|---|---|
| `cli/main.py` | `argparse` subcommands, exit-code constants, and the CLI-slug ↔ internal-format-label map. |
| `cli/commands.py` | `export`, `convert`, `validate`, `predict`. Only `predict` imports torch, lazily. |

## Level 3: Export/Import Subsystem

### Export Formats (export_formats.py)

**Functions**:
- `export_coco_json()`: COCO format with images directory
- `export_yolo_v5plus()`: YOLOv11-compatible structure
- `export_yolo_v4()`: Legacy YOLO format
- `export_labeled_images()`: Colored overlay visualizations
- `export_semantic_labels()`: Single-channel label images
- `export_pascal_voc_bbox()`: Pascal VOC XML (bounding boxes)

**Data Flow**:
```
all_annotations (internal format)
    │
    ├──> convert_to_coco() ──> COCO JSON
    ├──> convert to YOLO ──> YOLO txt files + data.yaml
    ├──> render_annotations() ──> Labeled images (PNG)
    ├──> mask_to_labels() ──> Semantic labels (PNG)
    └──> convert to Pascal VOC ──> XML files
```

### Import Formats (import_formats.py)

**Functions**:
- `import_coco_json()`: Parse COCO format
- `import_yolo_v5plus()`: Parse YOLO v8/v11 format
- `import_yolo_v4()`: Parse legacy YOLO format
- `process_import_format()`: Unified import dispatcher

## Level 4: Tool Dialogs

Each tool is a standalone dialog/window:

| Module | Purpose | Key Features |
|--------|---------|--------------|
| `annotation_statistics.py` | Statistics display | Count, area per class, plotly charts |
| `coco_json_combiner.py` | Merge datasets | Combine multiple COCO JSON files |
| `dataset_splitter.py` | Train/val/test split | Configurable ratios, **unseeded** `random.shuffle` — neither reproducible nor group-aware (ADR-044) |
| `image_patcher.py` | Create patches | Sliding window with overlap |
| `image_augmenter.py` | Data augmentation | Rotation, flip, brightness, preview |
| `slice_registration.py` | Align slices | Multiple registration algorithms (pystackreg) |
| `stack_interpolator.py` | Z-spacing adjustment | Interpolation methods, memory-efficient |
| `dicom_converter.py` | DICOM to TIFF | Preserve metadata, export to JSON |
| `yolo_trainer.py` | Model training | Train YOLO (run → `models/yolo/custom/<project>`, pruned to `best.pt`+`data.yaml` when MLflow has the diagnostics), `list_custom_yolo_models` for the prediction dropdown, `mlflow_run_url` signal, load predictions |

## Data Model

### Project JSON Structure

```json
{
  "images": ["image1.png", "image2.jpg"],
  "image_paths": {
    "image1.png": "/absolute/path/to/image1.png"
  },
  "classes": ["cell", "nucleus"],
  "class_colors": {
    "cell": [255, 0, 0],
    "nucleus": [0, 255, 0]
  },
  "annotations": {
    "image1.png": [
      {
        "segmentation": [x1, y1, x2, y2, ...],
        "category": "cell"
      },
      {
        "bbox": [x, y, width, height],
        "category": "nucleus"
      }
    ]
  },
  "image_dimensions": {
    "stack.tif": "TZCYX"
  },
  "image_shapes": {
    "stack.tif": [10, 50, 3, 512, 512]
  }
}
```

### Annotation Formats

**Polygon** (segmentation):
```python
{
    "segmentation": [x1, y1, x2, y2, x3, y3, ...],  # Flattened coordinates
    "category": "class_name"
}
```

**Rectangle** (bounding box):
```python
{
    "bbox": [x, y, width, height],  # COCO format
    "category": "class_name"
}
```

**Keypoint / pose instance** (ADR-029, #35) — one instance of a pose class's
K keypoints, no `segmentation` key (the discriminator that routes area/detail/
render):
```python
{
    "keypoints": [x1, y1, v1, x2, y2, v2, ...],  # flat [x,y,v]*K, COCO order
    "num_keypoints": <count of points with v > 0>,
    "bbox": [x, y, width, height],               # instance box (resizable)
    "category_name": "person", "category_id": 1, "number": 1,
}
```
The per-class schema (ordered point names, skeleton edges, flip_idx) lives in
`ImageAnnotator.keypoint_schemas` and is embedded on each `classes[]` entry in
the `.iap` file. Pure validation is in `core/keypoint_schema.py`; the editor is
`dialogs/keypoint_schema_dialog.py::KeypointSchemaDialog`. This shape also
round-trips through COCO-keypoints and YOLO-pose export/import (issue #35
PR-2) — see `io/export_formats.py`/`io/import_formats.py` and the ADR-029
addendum. `YOLOTrainer.predict()` and `YOLOController.process_yolo_results()`
are task-aware (detect/segment/pose) rather than hardcoded to
segmentation-only output (issue #35 PR-3).

### Multi-dimensional Image Handling

**Slice Naming Convention**:
```
{ext-stripped base}_T{t+1}_Z{z+1}_C{c+1}_S{s+1}
Example: stack_T1_Z5_C1_S1     (from stack.tif — no extension, 1-based)
Video:   clip_F00042           (core/video_handler.frame_key)
```

The missing extension is what distinguishes a slice name from an image name across the
exporters and `core/dataset_split` (ADR-044) — see
[Slice Naming Convention](08_crosscutting_concepts.md#slice-naming-convention).

**Dimension Labels**: T (Time), Z (Depth), C (Channel), S (Scene), H (Height), W (Width)

## Dependencies Between Components

```
ImageAnnotator (main window)
    ├── uses ──> ImageLabel (display/interaction)
    ├── uses ──> SAMUtils (model inference)
    ├── uses ──> export_formats (export)
    ├── uses ──> import_formats (import)
    ├── uses ──> yolo_trainer (training)
    └── launches ──> Tool Dialogs (utilities)

ImageLabel
    ├── emits signals to ──> ImageAnnotator (writes; see ADR-018)
    ├── reads via ──> CanvasContext (paint/eraser size, current class,
    │                  class_mapping, is_class_visible, scroll_area, …)
    └── uses ──> utils (area, bbox calculations)

SAMUtils
    └── depends on ──> ultralytics.SAM

Tool Dialogs
    └── operate independently (standalone windows)
```
