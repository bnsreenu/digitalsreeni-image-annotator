# Cross-cutting Concepts

## Coordinate Systems

### Screen Coordinates vs Image Coordinates

All mouse events are in screen coordinates and must be converted to image coordinates:

```python
# In ImageLabel
def screen_to_image_coords(self, screen_pos):
    # Account for offset (centering)
    image_x = screen_pos.x() - self.offset_x
    image_y = screen_pos.y() - self.offset_y

    # Account for zoom
    original_x = image_x / self.zoom_factor
    original_y = image_y / self.zoom_factor

    return (original_x, original_y)
```

### Annotation Storage Format

Annotations are stored in image coordinates (unzoomed, absolute pixels):
- **Polygon**: Flattened list `[x1, y1, x2, y2, ...]`
- **Rectangle**: COCO format `[x, y, width, height]`

### Bounds Enforcement — Clamp vs Clip (issues #32 / #36)

No commit path may persist coordinates outside the image rectangle (out-of-bounds
masks silently poison training data). Four pure helpers live in `utils.py`, and
the path picks **clamp** or **clip** by whether vertex correspondence must
survive:

| Helper | Semantics | Used by |
|--------|-----------|---------|
| `clamp_segmentation(seg, w, h)` | per-coordinate snap into `[0,w]×[0,h]`; **count-preserving** | polygon vertex-edit commit (Enter); polygon shape-edit commit (#40) |
| `clamp_bbox(box, w, h)` | snap `[x,y,w,h]` inside (independent corners — trim); keep rectangular & ≥1px | box **resize** commit (#40) |
| `fit_bbox_inside(box, w, h)` | translate `[x,y,w,h]` back inside, **size-preserving** | box / polygon **move** commit (#40) |
| `clip_polygon_to_bounds(seg, w, h)` | shapely intersection (largest part; `buffer(0)` repairs self-intersections); **may split/drop** → `None` | Image Augmenter per transformed polygon (#36) |

**Clamp** for live manual edits (a dragged polygon must not lose or reorder
vertices). **Clip** for batch augmentation (a rotated/zoomed shape genuinely
exits the frame and should be cut at the edge; a fully-outside polygon is dropped).
*Drawn* shapes were already shapely-clipped in `finish_polygon` /
`finish_rectangle`. See ADR-024.

### Polygon simplification — Detail % (issue #24)

SAM/DINO masks are dense (hundreds of vertices). The Annotations table's per-row
**Detail %** spinbox (100 = raw) thins a mask via `utils.simplify_polygon`
(Douglas-Peucker, `cv2.approxPolyDP`, binary-searched to a vertex budget). It is
**reversible**: the dense original is lazy-captured into `segmentation_raw` on
first simplify, and 100 % restores it exactly. The effective (possibly simplified)
`segmentation` is what renders and exports; `segmentation_raw` + `detail_pct` ride
along in `.iap`. See ADR-025.

### Pan + Zoom Reference Frames

Two non-obvious gotchas live in `ImageLabel.mouseMoveEvent` /
`wheelEvent`:

- **Pan must use `event.globalPosition()`, not `event.position()`.**
  Widget-local coords absorb half the cursor delta during a scrollbar
  move (the widget shifts under the cursor mid-drag) → effective
  half-speed pan. The global frame is stable.
- **Zoom-to-cursor must compute the post-zoom `offset_x/y`
  analytically from the viewport, not read `self.offset_x` after the
  zoom call.** `update_scaled_pixmap()` only *relaxes* the minimum
  size on zoom-out; the widget hasn't shrunk by the time
  `update_offset()` runs, so `self.width()` is stale and the offset
  comes out wrong. Use `viewport().width()` + `scaled_pixmap.width()`
  to derive the offset directly. Zoom-in worked by accident because
  the widget grows immediately when `setMinimumSize` enlarges it.

## Image Format Conversions

### QImage ↔ NumPy Array

**QImage to NumPy** (for SAM inference):
```python
def qimage_to_numpy(qimage):
    width = qimage.width()
    height = qimage.height()
    fmt = qimage.format()

    if fmt == QImage.Format_Grayscale16:
        # 16-bit → normalize to 8-bit → RGB
        buffer = qimage.constBits().asarray(height * width * 2)
        image = np.frombuffer(buffer, dtype=np.uint16)
        image_8bit = normalize_16bit_to_8bit(image)
        return np.stack((image_8bit,) * 3, axis=-1)

    elif fmt == QImage.Format_RGB888:
        # Direct conversion
        buffer = qimage.constBits().asarray(height * width * 3)
        return np.frombuffer(buffer, dtype=np.uint8).reshape((height, width, 3))

    # ... handle other formats
```

**16-bit Normalization**:
```python
def normalize_16bit_to_8bit(image):
    # Percentile-based normalization for better contrast
    p2, p98 = np.percentile(image, (2, 98))
    image_clipped = np.clip(image, p2, p98)
    return ((image_clipped - p2) / (p98 - p2) * 255).astype(np.uint8)
```

## Polygon Operations

### Shapely for Geometry

**Merge Annotations**:
```python
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid

# Convert segmentation lists to Shapely Polygons
polygons = []
for ann in selected_annotations:
    coords = [(ann["segmentation"][i], ann["segmentation"][i+1])
              for i in range(0, len(ann["segmentation"]), 2)]
    poly = Polygon(coords)
    poly = make_valid(poly)  # Fix invalid polygons
    polygons.append(poly)

# Merge
merged = unary_union(polygons)

# Convert back to segmentation format
coords = list(merged.exterior.coords)
segmentation = [coord for point in coords for coord in point]
```

### Minimum Area Threshold

Paint brush annotations filter out small artifacts:
```python
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
    if cv2.contourArea(contour) > 10:  # 10 pixels minimum
        # Accept annotation
```

## Project Portability & Unsaved-Project Recovery (#41 / #42)

**Relative image paths (ADR-033).** `build_project_data()` writes `image_paths_rel`
(POSIX separators, project-relative) alongside the absolute `image_paths`, but only when
a project dir exists. `resolve_image_path()` resolves in order relative → absolute →
`<project_dir>/images/` → missing, so a moved project folder opens cleanly and v1 files
still resolve. `core/project_schema.validate_project_data` (pure) runs after `json.load`
and raises `ValueError` on a structurally broken `.iap`.

**Unsaved-project recovery (ADR-032).** With no `current_project_file`, `auto_save()`
never prompts — it writes a silent snapshot (`build_project_data()` → atomic temp +
`os.replace`) to `AppDataLocation/recovery/`, path stored under QSettings
`recovery/pending_path`. `main()` calls `ProjectController.offer_recovery()` after
`window.show()` (never the constructor, so tests don't trigger it); a real save clears it.
The snapshot is exactly the `.iap` shape, so restore reuses `load_project_data`.

## Autosave and Project Corruption Prevention

### Critical: Disable Autosave During Load

**Problem**: Autosave triggered during loading can corrupt project files

**Solution** (v0.8.12):
```python
class ImageAnnotator:
    def load_project_data(self, project_data):
        self.is_loading_project = True  # Disable autosave
        try:
            # ... load all data
        finally:
            self.is_loading_project = False  # Re-enable

    def save_project(self, show_message=True):
        if self.is_loading_project:
            return  # Skip save during load
        # ... normal save logic
```

## SAM Model Management

### Model Caching

First use downloads models, subsequent uses load from cache:
```python
# Ultralytics automatically caches in:
# - Working directory (current implementation)
# - Or ~/.cache/ultralytics/ (default)

sam_model = SAM("sam2_t.pt")  # Downloads if not present
```

### Releasing Model GPU Memory

`SAMUtils.unload()` and `DINOUtils.unload()` must do **three** things,
in order:

1. Drop the cached Python references (`self._model = None`, etc.).
2. **`gc.collect()`** to break circular references inside Ultralytics
   / Transformers model objects (config ↔ model, processor ↔
   tokenizer). Without this, the C++/CUDA backing memory stays pinned
   until Python's cyclic GC runs on its own schedule, which can be
   many seconds or never. Task Manager / `nvidia-smi` will show zero
   drop in GPU memory.
3. **`torch.cuda.empty_cache()`** (plus `torch.cuda.ipc_collect()`) so
   the PyTorch allocator returns the freed blocks to the OS / driver.

Skipping step 2 was the cause of "Tools → Unload AI Models does
nothing visible" in v0.9.0 manual testing.

### Model Size Recommendations

| Model | Size | RAM Usage | Speed | Recommendation |
|-------|------|-----------|-------|----------------|
| SAM 2 tiny | ~40MB | Low | Fast | ✅ Recommended for most users |
| SAM 2 small | ~90MB | Medium | Medium | ✅ Good balance |
| SAM 2 base | ~150MB | Medium-High | Slow | ⚠️ Use with caution |
| SAM 2 large | ~400MB | High | Very Slow | ❌ Not recommended (crashes on limited resources) |

### Device Selection & Compute-Capability Fallback

`torch.cuda.is_available()` is **not** sufficient to decide on GPU
inference: it returns True for any visible CUDA device even when the
installed torch wheels contain no kernels for its compute capability
(torch ≥ 2.8 wheels ship sm_70+ only, so a Pascal GTX 1050 / sm_61
passes the check but every kernel launch fails with
`CUDA error: no kernel image is available for execution on the device`
— upstream issue #57).

All inference paths therefore resolve their device through
`core/torch_utils.resolve_torch_device()`:

- compares `torch.cuda.get_device_capability(0)` against the minimum
  `sm_*` in `torch.cuda.get_arch_list()`; on mismatch returns
  `("cpu", warning)` instead of `"cuda"`,
- caches the decision process-wide (SAM, DINO and YOLO share it),
- prints the warning once; `maybe_warn_cpu_fallback(parent)` shows it
  as a one-time `QMessageBox` from the SAM model picker and the DINO
  detect entry points.

SAM passes `device=` explicitly on every predict call; DINO's
`_resolve_device()` delegates to the helper (the `DINO_DEVICE` env
override still wins); the YOLO trainer passes `device=` to
`model.train()` and prediction. Never call bare
`torch.cuda.is_available()` to pick a device in new code.

## Dark Mode Support

### Stylesheet Switching

```python
# In ImageAnnotator
if dark_mode_enabled:
    self.setStyleSheet(soft_dark_stylesheet)
    self.image_label.set_dark_mode(True)
else:
    self.setStyleSheet(default_stylesheet)
    self.image_label.set_dark_mode(False)
```

**Dark Mode Considerations**:
- Annotation rendering uses inverted colors for visibility
- Text labels use high-contrast colors
- Background grid adjusted for dark backgrounds

### Dark Mode — No Hardcoded Colors Rule

**Do not hardcode `background`, `color`, or other palette-dependent
values in widget `setStyleSheet(...)` calls.** They override both the
default OS look *and* `soft_dark_stylesheet.py`, leaving bright
rectangles on the dark sidebar. Past offenders that bit us:

- `ClassThresholdTable` header had `background: #e0e0e0;` → bright bar
  across the top of the DINO panel in dark mode.
- `lbl_dino_status` had `background: #f5f5f5;` → bright box where the
  "No DINO model loaded" status sat.

Either leave the property out of the inline stylesheet so the global
sheet wins, or use Qt's palette role functions (`palette(base)`,
`palette(mid)`, `palette(text)`, …) which resolve at paint time
against the active palette. Inline hardcoded greys are an anti-pattern.

When introducing a new widget type that doesn't have a rule in
`soft_dark_stylesheet.py` yet — add the rule there *first*, then build
the widget. Otherwise the widget uses the OS default in dark mode,
which on Windows means barely-visible radio-button indicators and
white-on-white headers (the dataset splitter radio buttons hit this
before they were styled).

## Left Sidebar — Scrollable Layout (issue #88)

The left sidebar (`ui/sidebar.py::build_sidebar`) stacks Import, Classes,
the Annotation tools, the DINO panel, the Annotations table and Export in
one `QVBoxLayout`. On short screens or at large UI font sizes an expanded
DINO panel used to squeeze the Annotations table down to just its header
row. The fix is structural, not per-widget:

- The whole sidebar content widget lives inside a `QScrollArea`
  (`window.sidebar_scroll`, `setWidgetResizable(True)`) that is what gets
  added to the main layout — `window.sidebar` itself is the scroll area's
  inner widget. Horizontal scrolling is `ScrollBarAlwaysOff`; because the
  area is width-resizable the content is resized to the viewport width. A
  `setMinimumWidth` on the scroll area preserves the window-width floor the
  sidebar's own `minimumSizeHint` used to impose.
- Each vertically-competing section (`class_list`, `dino_class_table`,
  `dino_phrase_panel`, `annotation_list`) carries a `setMinimumHeight(...)`
  so it keeps a usable size. The sum of those minimums is what makes the
  scroll area scroll when the window is too short — otherwise a section
  could still collapse *inside* the scroll area.
- The scroll area uses `QFrame.Shape.NoFrame` and sets no
  `background:`/`color:` — the active stylesheet still paints the sidebar
  (No Hardcoded Colors Rule above).

## UI Font Zoom (Low-Vision Mode)

### Single Source of Truth: `ui_font_pt`

All UI text size flows from one integer, `ImageAnnotator.ui_font_pt`
(8–24pt, default 10, clamped by `app_settings.clamp_font_pt`). The
Settings → Font Size presets (Small…XXL) jump to fixed values;
Ctrl+Shift+= / Ctrl+Shift+- step ±1pt; Ctrl+Shift+0 resets. Every
change goes through `theme.set_font_pt`, which clamps, re-applies the
theme, persists via QSettings and syncs the preset menu checkmarks
(no preset is checked at an in-between size).

### Appended QSS Overrides, Not Templated Stylesheets

`soft_dark_stylesheet.py` / `default_stylesheet.py` stay static
strings. `apply_theme_and_font` appends scaled rules *after* the
static sheet — later rules of equal specificity win in QSS — for the
body font, `.section-header` and checkbox/radio indicator sizes. The
overrides scale the legacy px values (14px header, 14px indicators,
8px radio radius, 11px/10px compact DINO panel) by `ui_font_pt / 10`
and stay in **px**, so at the default 10pt they reproduce the legacy
look exactly. Widgets that want smaller-than-body text (e.g. the DINO
threshold table / phrase panel) must not set their own `font-size` —
they get a type- or objectName-targeted rule in the appended block
instead, so "compact" still scales. Do not
hardcode `font-size` in widget `setStyleSheet(...)` calls: it overrides
the global rule and the widget stops scaling (same failure mode as the
No Hardcoded Colors rule below; the DINO sidebar captions hit this).

### Canvas Overlay Scaling: `ui_scale`

`apply_theme_and_font` pushes `ui_font_pt / 10.0` to
`ImageLabel.set_ui_scale`. Overlay sizes (annotation label fonts, SAM
point radii, pen widths, edit-point handles, hit-test tolerances) use
the helpers `ImageLabel._pen_w(base)` / `_overlay_font(base)`, which
multiply by `ui_scale` and divide by `zoom_factor` — UI zoom and image
zoom stay orthogonal: overlays grow with the font setting but remain
constant-size on screen across image zoom. At the default 10pt,
`ui_scale == 1.0` and rendering is pixel-identical to the legacy code.
Exception: the SAM point-marker radii are drawn under
`painter.scale(zoom)` without zoom compensation (pre-existing
behaviour) and only multiply by `ui_scale`.

### Persistence via QSettings

`app_settings.py` stores `ui/font_pt` and `ui/dark_mode` in
`QSettings("DigitalSreeni", "ImageAnnotator")` (registry under HKCU on
Windows). These are per-user preferences, deliberately *not* part of
the `.iap` project file. All functions take an optional `QSettings`
instance so tests inject an INI-backed temp file.

## Thread Safety for YOLO Training

### Training Thread

```python
class TrainingThread(QThread):
    progress_update = pyqtSignal(str)
    finished = pyqtSignal(object)

    def run(self):
        try:
            results = self.yolo_trainer.train_model(
                epochs=self.epochs,
                imgsz=self.imgsz
            )
            self.finished.emit(results)
        except Exception as e:
            self.finished.emit(str(e))
```

**UI Update**:
- Training runs in background thread
- Progress updates via Qt signals
- UI remains responsive during training

## MLflow Experiment Tracking (Always On)

MLflow tracking (issue #74, [ADR-027](09_architecture_decisions.md#adr-027-mandatory-mlflow-experiment-tracking-sam-explicit--yolo-native))
records *every* training run so checkpoints stay tied to their hyperparameters and
loss curves. Tracking is mandatory — there is no enable/disable. Several
cross-cutting rules govern it:

**Core dependency, lazy import, crash-safe.** MLflow is in `install_requires` (a
fresh install always has it). `import mlflow` still happens only inside
`training/mlflow_tracker.py` methods (the same lazy idiom as SAM/DINO), so startup
stays fast and a *broken* install can't stop the GUI launching. There is no
"disabled" tracker; the only no-op is `_NullTracker`, used when a trainer is called
without a tracker (direct/programmatic calls, tests). Every live MLflow call is
wrapped so a tracking error logs a status line but **never aborts a run** — pure
crash-safety, not an opt-out.

**Tracking-URI resolution.** `resolve_tracking_uri(main_window)` picks the store by
precedence: a non-empty QSettings override (`tracking/mlflow_uri`) → `<project>/mlruns`
when a project is open (`main_window.current_project_dir`) → `<cwd>/mlruns`.
`to_mlflow_uri()` then converts a local path to a `file://` URI at the MLflow
boundary (a bare Windows `C:\…` path is otherwise read as scheme `c` and rejected),
and `MLFLOW_ALLOW_FILE_STORE=true` is set so mlflow 3.x accepts the local file store.

**Where logging lives — by trainer shape.** SAM has a custom loop, so it logs
*explicitly* through a tracker passed into `SAMFineTuner.train(..., tracker=...)`.
Critically, the MLflow run is started, logged to, and ended **inside `train()` on the
worker thread** — MLflow runs are thread-bound, and the controller only constructs the
(unstarted) tracker. The tracker's status `log` is wired to the trainer's
thread-safe `progress_signal`, so tracker messages reach the GUI via a queued
connection (never a direct cross-thread QTextEdit write). YOLO instead arms
Ultralytics' built-in MLflow callback (`MLFLOW_TRACKING_URI` /
`MLFLOW_EXPERIMENT_NAME` env + `ultralytics.settings.update({"mlflow": True})`)
on every run.

## Error Handling

### Error-Handling Convention (issue #34)

Where an exception is caught and what happens to it is a decision, not an
accident. The rules, enforced in review:

1. **Core / inference / io / training modules raise.** They never show dialogs
   and never return `None` to signal failure of a user-initiated action.
2. **Controllers / dialogs (the UI boundary) catch**, `logger.exception(...)`,
   and surface via `QMessageBox` when a user action failed.
3. **Catch the narrowest exception type** that models the expected failure.
   `except Exception` is allowed only at a UI boundary or a documented
   crash-safety barrier (e.g. `mlflow_tracker` "never let tracking abort
   training").
4. **Never `pass` silently.** Every swallowed exception must either log
   (`logger.exception(...)` / `logger.warning(..., exc_info=True)`) or carry a
   narrow type plus a `# reason` comment.
5. **Bare `except:` is banned** (it also traps `KeyboardInterrupt` / `SystemExit`).

```python
# Good — narrow, documented no-op
except TypeError:
    pass  # signal already disconnected

# Good — swallow but stay visible
except Exception:
    logger.warning("CUDA cache cleanup failed during unload", exc_info=True)

# Bad — silent, untyped
except Exception:
    pass
```

The OOM-on-model-load path is a worked example: `core/torch_utils._is_oom`
(torch-free) lets `SAMController.change_sam_model` show a tailored "pick a
smaller model" dialog for out-of-memory failures while keeping the generic
download/torch message for everything else. See ADR-031.

### YOLO Model/Data Mismatch

**Problem**: Loading YOLO model trained on different classes

**Solution**:
```python
try:
    model = YOLO(model_path)
    model_classes = model.names
    yaml_classes = data_yaml['names']

    if model_classes != yaml_classes:
        QMessageBox.warning(
            self,
            "Class Mismatch",
            f"Model classes: {model_classes}\n"
            f"Data classes: {yaml_classes}"
        )
        return
except Exception as e:
    # Handle gracefully instead of crashing
```

### YOLO-Pose Single-Schema-Per-Dataset (issue #35 PR-2)

**Problem**: A YOLO-pose dataset's `data.yaml` has ONE global `kpt_shape`/
`flip_idx`, not one per class — exporting a mix of pose classes with
different K, or a pose class alongside a non-pose class, would silently
produce inconsistent label-line token counts with no matching schema.

**Solution** — validate before writing anything to disk, same
`ValueError` → `QMessageBox.warning` surfacing as the YOLO model/data
mismatch above:
```python
# io/export_formats.py
def _pose_export_check(all_annotations, class_mapping, keypoint_schemas):
    ...
    if inconsistent or len(distinct_k) > 1 or non_pose_classes:
        raise ValueError("YOLO-pose export requires every exported class "
                          "to share exactly one keypoint schema (K) ...")
    return k, flip_idx

# controllers/io_controller.py
try:
    output_dir, yaml_path = export_yolo_v5plus(..., keypoint_schemas=mw.keypoint_schemas)
except ValueError as e:
    QMessageBox.warning(mw, "Export Error", str(e))
    return
```
`_pose_export_check` runs first, before `os.makedirs` — a rejected export
leaves zero output on disk.

## Multi-dimensional Image Slicing

### Dimension Assignment

User assigns meaning to each dimension:
```
TIFF shape: (10, 50, 3, 512, 512)
User assigns: T   Z   C   H    W

Result: 10 timepoints × 50 Z-slices × 3 channels = 1500 slices
Each slice: 512×512 pixels
```

### Slice Naming Convention

The authority is `SliceProvider._build_index` in `core/slice_cache.py`. Two details are
load-bearing and were documented backwards until #81 — the base is **ext-stripped** and the
indices are **1-based**:

```python
# core/slice_cache.py, paraphrased
slice_name = f"{base}_" + "_".join(f"{dims[i]}{val + 1}" for ...)

# base = os.path.splitext(os.path.basename(path))[0]   -> "stack", not "stack.tif"
# val + 1                                              -> Z1, not Z0
#
# Example: "stack_T1_Z5_C1"
# Video frames: "clip_F00042"  (core/video_handler.frame_key, ADR-037)
```

**The absence of a dot is the discriminator.** A slice name has no extension; a regular image
name keeps one. Three places rely on that to tell them apart: both YOLO exporters
(`'_' in name and '.' not in name`), `io/export_formats._is_exportable`, and
`core/dataset_split._slice_base`, which derives the group a slice belongs to for the train/val
split (ADR-044). `add_images_to_list` enforces the ext-stripping so `video.mp4` and `video.tif`
cannot clobber each other's slices.

Writing `stack.tif_T0_Z5_C0` into a fixture therefore produces a name the grouping treats as an
ordinary image — which is not a bug in the grouping.

## The Shortcut Registry and Its Gates (issue #65, ADR-043)

The app has **two** keyboard mechanisms, and which one a binding belongs to is a real decision:

| Mechanism | For | Why |
|---|---|---|
| `QShortcut` with `ApplicationShortcut` | **unconditional** bindings — Ctrl+Z, Ctrl+Y, F2 | `QTableWidget` and other focusable children consume keys before they bubble to the main window, so `keyPressEvent` never sees them. No text field wants a modified key. |
| `ShortcutEventFilter` (`ui/shortcuts.py`) | **conditional** bindings — bare digits and letters, plus Ctrl+C/Ctrl+V | An `ApplicationShortcut` on `3` swallows that keystroke in *every* `QLineEdit`: renaming a class to "Layer 3" or typing a DINO phrase would silently drop characters. An event filter is the only mechanism that can be conditional on where focus is. |

The filter is a **registry**, not another bespoke filter: `{(key, modifiers): callable}` plus a
list of gate predicates that must all pass. Adding a global-key feature means registering a
binding. This is what ADR-015's follow-up asked for.

### The gates

Both live in `ui/input_gates.py` and are **shared with `DINOReviewEventFilter`**, which used to
duplicate them inline. Two filters drifting apart on a *safety* predicate is exactly the failure
that note warned about.

- `no_modal_open()` — a modal owns the keyboard; firing a canvas binding underneath it would act
  on state the user cannot see.
- `focus_is_text_entry()` — covers `QLineEdit`, `QTextEdit`, `QPlainTextEdit`, `QAbstractSpinBox`
  and **editable** combo boxes. A non-editable combo takes no typed text, so gating on it would
  needlessly disable the bindings whenever a dropdown has focus.

`widget_is_text_entry(widget)` is split out from `focus_is_text_entry()` deliberately: which
widget holds focus is a property of the windowing system and unreliable under the offscreen
platform used in CI, whereas "is a `QSpinBox` a text entry" is a decision this module owns and
must get right. The split is what makes the classification testable.

### Rules for new bindings

- A binding returning `False` does **not** consume the event — that is what makes an out-of-range
  digit a silent no-op rather than an error.
- Tool bindings must call `ImageAnnotator.activate_tool` and nothing else. Writing `current_tool`
  or the SAM flags directly reintroduces the drift the choke-point exists to prevent.
- Digits address the **first nine** classes only. Classes 10+ stay mouse-reachable; a
  modifier-based second bank was rejected because two-key class selection is slower than clicking.

### Discoverability without breaking the item text

`ClassShortcutDelegate` *paints* the 1-9 badge. It cannot go into the item text, because **the
item text IS the class name** across the app — `findItems(MatchExactly)` re-selects on it, the
`Temp-` prefix checks drive the whole review workflow, `text()[5:]` derives the permanent name on
accept, and rename reads it back. The same rule applies to the review-score badge on the image
list (`ImageScoreDelegate`, issue #71), where the text additionally backs the
`all_images[i]` ↔ `image_list.item(i)` positional invariant (ADR-035).

## Redundancy and Uncertainty: a Precedence Rule (issues #71 + #82)

Two features rank the same images by different criteria, and they meet on the cluster list.

- **Redundancy** (#72/#82) answers *what can be skipped*. Forty near-identical frames teach
  roughly what five would.
- **Uncertainty** (#71) answers *what is worth the effort*. A high score means the model is unsure
  or disagrees with the labels.

The rule is **precedence, never a combined score**: redundancy filters, uncertainty ranks what is
left. So the suggested member of a near-duplicate cluster is the medoid, labelled `most typical` —
the right pick when the question is *which one to keep* — unless review scores are available, in
which case it becomes the most uncertain member, labelled `most uncertain` — the right pick when
the question is *which one to annotate*.

Multiplying the two into one number would need a weight nobody can justify, and would hide which
of the two drove the answer.

Two conditions gate the switch, both in `CurationController.suggested`:

1. **Every member must be scored.** Otherwise the members that happen to have been measured
   outrank the unmeasured ones for no reason.
2. **Every score must be an uncertainty score.** `ReviewController` scores an annotated image by
   *disagreement* (against its labels) and an unannotated one by *uncertainty* (against nothing).
   These are different quantities on different scales; ranking a cluster that mixes them compares
   two different measurements, and does so entirely plausibly.

Worth knowing before reading anything into an empty uncertainty column: the two features overlap
on **plain-image projects and nowhere else**. Review scoring runs from a file path and skips
stacks and videos entirely, while curation exists mainly *for* slices. That is why the column is
hidden rather than shown empty when no scores exist — an empty column reads as "nothing is
uncertain here", not "nothing measured it".

## The Annotation Clipboard (issue #66)

App-level, not per-image: it survives switching image, slice, frame and project, because "copy
here, paste 40 slices later" is the entire point. Deliberately **not** the system clipboard —
annotations are rich dicts with class bindings and schema constraints, and round-tripping them
through text would lose the class-mapping decision.

Three rules, each present because of a specific way the naive version breaks:

1. **Deep copy in and out.** `image_label.annotations` is itself a deep copy and PyQt round-trips
   `UserRole` dicts as copies, so value-equality is the only stable identity (ADR-022). A
   reference would make the pasted shape an alias of the source.
2. **Clamp, don't clip.** The target may be smaller. Per-coordinate clamping preserves vertex
   count and ordering (ADR-024); a shapely clip is geometrically prettier but changes the vertex
   count out from under a shape the user is about to nudge. `segmentation_raw` is clamped the
   same way, or restoring Detail-% to 100 would push coordinates back out of bounds.
3. **A pose travels only to a class with the same K.** K is locked once instances exist
   (ADR-029); a 17-point pose on a 5-point class is not slightly wrong, it is corrupt. Mismatches
   are reported and skipped — and a skipped pose does not cost the user the polygons pasted
   alongside it.

One `record_history()` before any mutation, so a five-annotation paste is one Ctrl+Z. A cancel at
the class-mapping dialog pastes **nothing at all**: a partial paste after a cancel is worse than
none.

## Onion-Skinning and the Slice LRU (issue #67)

### Ghost the annotations, not the pixels

There are **two** ghosts, chosen by the `content` setting, and which one is the default matters
more than any other decision here.

The raster ghost — the animator's onion skin, the neighbouring slice blended over the current one
— shipped first and was the first thing the feature got complained about. On an opaque
photographic slice it does not read as "here is where that object was"; it reads as *this image is
out of focus*. Two cels differ only where the drawing moved, so a blend is legible; two adjacent
microscopy slices or video frames differ **everywhere**, so the blend is just noise.

The question actually worth answering while stepping through a stack is *what did I label on the
neighbouring slice, and does this one line up with it?* So `CONTENT_ANNOTATIONS` is the default:
the neighbour's committed shapes, dashed and unfilled in their class colour. The current slice's
own masks are filled, so an unfilled ghost stays distinguishable from exactly the thing it exists
to be compared against. Hidden classes are skipped — a ghost that ignores the visibility checkbox
is a ghost you cannot turn off.

The annotation ghost is also the **cheap** one: a dict lookup per neighbour in `all_annotations`,
no decode and no LRU traffic. `onion.wants_image()` gates the expensive half, so the default
costs nothing beyond the lookup.

### Layering and resolution

The ghost is drawn **after** the current image and **before** every annotation layer. The issue
asked for "underneath the current image", which cannot work: unlike an animation cel, an image
slice is a fully opaque raster, so anything painted beneath it is invisible. The chosen order
delivers what the requirement actually wanted — a visible ghost with the current slice's
annotations legible on top.

Neighbours are resolved in `display_image`, the single funnel where the canvas image changes,
**never in `paintEvent`**: a cache lookup (and on a miss, a full decode) per repaint would put
decoding in the pan and zoom path. Pixels go through `LazySliceList.get` like every other consumer,
so the shared bounded LRU (ADR-036) stays the only owner of decoded pixels. The default
single-neighbour mode costs one extra live decode; "both" costs two, which is why the capacity of
8 was left alone rather than quietly raised.

The ghost is **decorative and stays that way**: it never becomes `original_pixmap` (what SAM,
export and every measurement read), contributes nothing to hit-testing, and is dropped on
`clear()`. Opacity is restored to 1.0 after the pass so it cannot wash out later layers.

## Vertex Count Changes and the Detail-% Baseline (issue #68)

`segmentation_raw` (ADR-025) is captured lazily the first time a mask is thinned and is the source
the Detail-% spinbox re-simplifies **from**. Once the live polygon gains or loses a vertex, that
copy describes a different shape — so the next Detail-% drag would re-derive the outline from
pre-edit geometry and **silently revert the user's work**.

Resolved by **invalidating** rather than mirroring. Mirroring only means something if the raw and
the live polygon share a vertex correspondence, and after simplification they do not: the raw is
denser, so "the vertex the user just deleted" has no single counterpart there. Invalidation makes
the edited shape the new baseline — correct, and explainable as "Detail-% restarts from what you
now see". Esc restores the raw/detail pair too, so cancelling costs nothing either.

The undo model stays session-scoped: the baseline is captured at edit-mode entry, insert/remove
save and refresh the table via `polygonGeometryChanged` **without** pushing history, and Enter
commits the session as one undo step. Esc reverts everything including insertions, leaving no
entry (ADR-026).

## Keyboard Shortcuts

### Global Shortcuts

| Shortcut | Action |
|----------|--------|
| Ctrl+N | New Project |
| Ctrl+O | Open Project |
| Ctrl+S | Save Project |
| Ctrl+W | Close Project |
| Ctrl+Shift+S | Save Project As |
| Ctrl+Alt+S | Annotation Statistics |
| Ctrl+Shift+= (or Ctrl++) | Increase UI font size |
| Ctrl+Shift+- (or Ctrl+-) | Decrease UI font size |
| Ctrl+Shift+0 | Reset UI font size |
| F1 | Help Window |

### Canvas Shortcuts

| Shortcut | Action |
|----------|--------|
| Ctrl+Wheel | Zoom In/Out |
| Ctrl+Drag | Pan |
| Click (no tool) | Select mask under cursor |
| Shift+Click (no tool) | Toggle mask in selection |
| Drag (no tool) | Rubber-band box-select; Shift+Drag adds |
| Drag handle (one shape selected) | Resize — scales a polygon, edits a box |
| Drag inside (one shape selected) | Move the whole shape |
| Delete | Delete selected mask(s) |
| Double-click | Enter vertex-edit mode |
| Esc | Cancel Current Annotation |
| Enter | Finish/Accept Annotation |
| Up/Down | Navigate Slices (multi-dimensional) |
| -/= | Adjust Brush/Eraser Size |

## Logging and Debug Output

The application logs through the stdlib `logging` module, rooted at the
package logger `digitalsreeni_image_annotator`. `print()` is **banned in
`src/`** (see ADR-030). Configuration lives in
[`core/logging_config.py`](../src/digitalsreeni_image_annotator/core/logging_config.py):

- `configure(level=None)` — called once from `main.py:main()` **before**
  `QApplication` is created. It installs a single stderr `StreamHandler` on
  the package logger, sets the level, and is **idempotent** (a second call
  adds no second handler — important for tests and re-entry).
- `get_logger(__name__)` — every module's logger. `configure()` derives the
  package root from its own `__name__`, so all loggers share that root and
  inherit its handler/level automatically — this holds whether the app is
  imported as `digitalsreeni_image_annotator` (installed / `sreeni`) or
  `src.digitalsreeni_image_annotator` (the `python -m src...` launcher).

### Level policy

| Level | Use for | Examples |
|-------|---------|----------|
| `debug` | Diagnostic chatter: array shapes/dtypes, metadata dumps, coordinate/point dumps, per-slice / per-file loop progress | "input points: …", "slice 5/12 written" |
| `info` | State changes a user might care about | "SAM model loaded: …", "Project auto-saved.", "Created N slices for …" |
| `warning` | Soft failures / ignored conditions **not** in an `except` ("Skipped …", "… not found", failed user action that returns) | "No SAM model selected." |
| `exception` / `error(exc_info=True)` | Inside an `except` block — appends the traceback | "Error applying SAM points" |

When in doubt between debug and info, choose **debug**: the default INFO level
must stay quiet enough for daily use, and a per-slice INFO line on a 2560-slice
stack is unusable. `logger.exception(...)` may only be called inside an
`except` block; outside one use `logger.error(..., exc_info=True)`.

### The debug switch

The default level is INFO. DEBUG is enabled by either:
- `--debug` on the command line (`python -m src.digitalsreeni_image_annotator.main --debug`), or
- `IMAGE_ANNOTATOR_DEBUG=1` in the environment.

### Rule for new code

New code uses `logger = get_logger(__name__)`, never `print()`. User-facing
messaging still goes through `QMessageBox` / dialogs — logging is the
diagnostic channel, dialogs are the user channel; the two are independent
(see the Error-Handling Convention above).

## Class Name Is a Primary Key — Sync Every Registry on Rename (#63)

There is no class-id object. A class is identified by its **name string**, and
that name is the key into several independent registries:

| Registry | Owner | Notes |
|----------|-------|-------|
| `class_mapping` | main window | name → numeric id (export) |
| `image_label.class_colors` | canvas | name → `QColor` |
| `image_label.class_visibility` | canvas | read via `.get(name, True)` |
| `keypoint_schemas` | main window | pose classes only (ADR-029) |
| `all_annotations[img]` / `image_label.annotations` | per-image buckets | plus each annotation's `category_name` |
| `dino_class_table` rows | DINO widget | keyed by row text |
| `dino_phrase_panel._phrases` | DINO widget | name → phrase list |
| `dino_batch_results` | main window | pending review results, `category_name` captured at **detection** time |
| `image_label.temp_annotations` | canvas | the on-screen half of the same review state |
| `AnnotationHistory` snapshots | annotation controller | each undo entry is a whole `{class_name: [...]}` dict (ADR-026) |

**Every roster mutation must touch all of them.** `add_class` and
`delete_class` do; `rename_class` silently skipped the DINO pair and
`class_visibility` until #63 — so a renamed class kept detecting under its dead
name, lost its phrases *and* thresholds on the next project load (both are
filtered against the live class list), and quietly became visible again.

The last three rows were added after #63, and each failed *later* and quietly.
A rename mid-review left pending proposals tagged with a name the project no
longer had — drawn in the fallback colour, then discarded on accept. And the
history snapshots made **Ctrl+Z after a rename restore the annotations under
the old name**: unmapped, uncoloured, absent from the class list, so invisible
on the canvas and still written into the `.iap`. A rename re-keys both; a
delete drops them (a prompted proposal for a deleted class can only ever be
discarded, and an undo that resurrects the class brings it back unusable).

The onion-skin ghosts (#67) are a **cache**, not a registry: they hold
`(class_name, annotations)` pairs resolved at navigation time, so rename and
delete call `refresh_onion_skin()` rather than re-keying them.

**Renaming onto an existing name is rejected, not merged.** `rename_class`
runs a **pre-flight** block that validates every precondition *before* the
first mutation, then commits. The collision check is against
`image_label.class_colors`, **not** `class_mapping` — `Temp-*` review classes
live in `class_colors` + `class_list` only (`add_temp_classes` never writes
`class_mapping`), so a `class_mapping` check would let a rename onto a pending
temp class through: the real class's annotation bucket merges into the temp
one, and the next Reject sweeps `Temp-*` and deletes it.
Without that guard the rename half-clobbered every registry — the old class's
id and colour overwrote the target's, annotation buckets merged, and the DINO
table ended up with two identically-named rows, which makes
`get_thresholds_dict()` (a name-keyed dict comprehension) drop one row's
thresholds and `_build_dino_class_configs()` emit the class twice.
`ClassThresholdTable.rename_class` carries the same uniqueness guard as a
backstop, mirroring `add_class`.

**`Temp-` is a reserved prefix, not just a naming convention.** It is a
stringly-typed discriminator with *destructive* semantics:
`accept_visible_temp_classes` / `reject_visible_temp_classes` sweep `Temp-*`
wholesale, and reject **deletes** the matching annotations. `rename_class`
therefore refuses to rename *into* the namespace, and refuses to rename a
pending `Temp-*` class *out* of it (accept or reject it first).

Known debt: the prefix is spelled as an inline `startswith("Temp-")` /
`"Temp-*"` literal in ~18 places across five modules rather than one
`is_temp_class()` helper. Note **why the guard lives on the rename path and
not on creation**: `load_project_data` calls `add_class` for every saved
class, so a naive rejection in `add_class` would make any legacy project
containing a hand-made `Temp-foo` class fail to load. A follow-up needs to
handle that case (migrate on load, or gate the guard on
`not is_loading_project`) before guarding creation.

**Rename must not re-derive visibility.** `item.setText()` on a `class_list`
row emits `itemChanged`, which is wired to `toggle_class_visibility` — that
would re-derive `class_visibility` from the checkbox and become a hidden
co-author of the rename. `rename_class` wraps the `setText` in
`blockSignals(True/False)` so the explicit re-key is the sole writer.

When adding a new name-keyed registry, add it to the table above **and** to
every mutation site. The registries are hand-synced at each call site rather
than projected from one roster — that coupling is known debt, and the reason
this table keeps growing one production bug at a time.

## DINO Temp Annotations — Single Field, Many Images

> **Four producers** now feed the temp-annotation / `dino_batch_results`
> pipeline. Adding one means widening `REVIEW_SOURCES` in `dino_controller.py`
> and parking results under the image key — never installing a parallel review
> mechanic (ADR-015).
>
> | Producer | `source` tag | Notes |
> |---|---|---|
> | Grounding-DINO two-stage | `"dino"` | the original |
> | SAM 3 text prompt | `"sam3"` | ADR-039, one-stage `SAM3Utils.detect_text` |
> | SAM 3 video tracking | `"sam3"` | ADR-040, **uncertain** frames only; confident frames bypass review and commit directly as `source: "sam3-track"` |
> | Segment Everything | `"sam-everything"` | issue #69, unprompted; additionally carries `assigned_class`, which is `None` until the user clicks the proposal |
>
> The **review gate** (`DINOReviewEventFilter`) reads the shared
> `REVIEW_SOURCES` tuple, so a producer cannot be half-registered *there*. Note
> that `source` carries other meanings elsewhere with their own literals —
> slice-list colouring in `image_controller` and the scoring exclusion in
> `core/disagreement` both test `"sam3-track"` directly — so a new producer
> still has to be checked against those by hand. All four share
> the SAME `dino_batch_results` dict, so the single-field re-sync rule below
> applies to all of them unchanged — which is precisely why Segment Everything
> parks its proposals there rather than living only in `temp_annotations`:
> without that, one click in the image list would silently discard a batch of
> class assignments.
>
> **Known limit**: one image key holds one producer's entries. Two producers
> parking for the same image means the last writer wins, silently. In practice
> the runs are sequential and `SegmentEverythingController.run()` refuses to
> start while a review is pending — but `stage_proposals`, the entry point the
> tests use, has no such guard, and neither does anything structural.

`ImageLabel.temp_annotations` is a **single list on the image_label**,
not a per-image cache. It holds the pending DINO+SAM masks shown as
an overlay while the user decides accept/reject. The per-image batch
cache is `ImageAnnotator.dino_batch_results` (a dict keyed by image
name) — `image_label.temp_annotations` is only ever set to one image's
slice of that dict at a time.

Consequences this codebase has tripped over:

- **Image/slice switches must re-sync** `temp_annotations` from
  `dino_batch_results` for the new image (load if pending, clear if
  not). Otherwise masks from the previously-viewed image visually
  bleed onto every slice the user navigates to. See
  `_refresh_dino_temp_for_current()`.
- **Enter / Escape during review** must work even when the focus is on
  slice_list / image_list / a button — `QListWidget` consumes
  Enter for itemActivated before `ImageLabel.keyPressEvent` ever sees
  it. Solved with an application-wide event filter
  (`DINOReviewEventFilter`) that fires only while
  `temp_annotations` has DINO items and skips modal dialogs and text
  inputs. Setting `image_label.setFocus()` synchronously inside
  `_show_dino_batch_review` was not enough — Qt's focus handling
  raced the click event that opened the review and the canvas
  often didn't end up focused. `QTimer.singleShot(0, …)` defers until
  the current event chain settles.
- **Auto-accept dropdown applies to both paths.** The batch-mode
  combo ("Review before accepting" / "Auto-accept all detections")
  controls **both** "Detect Current Image" and "Detect All Images".
  Only checking it in `run_dino_detection_batch` and not
  `run_dino_detection_single` produced a confusing "auto-accept
  doesn't actually auto-accept for single image" bug.
- **Batch detection must enumerate slices, not just `all_images`.**
  Multi-dim images live in `all_images` as a single entry with
  `is_multi_slice=True`, and their actual slice QImages live under
  `self.image_slices[base_name]`. The first cut of
  `run_dino_detection_batch` iterated `all_images` and skipped the
  multi-slice entries with a console log — leaving stack-based
  projects unable to use "Detect All Images" at all. Batch jobs go
  through `_collect_dino_batch_work_items()` which flattens regular
  images + every loaded slice into a `(name, QImage)` list.
- **Review navigation must handle slice names.** Slice names like
  `stack_T1_Z1_C1` are not in `image_list`. After collecting batch
  results for slices, `_navigate_to_image_or_slice()` finds the
  parent image via `os.path.splitext` matching and then activates
  the specific row in `slice_list`. Without this, batch review on
  slices either silently no-op'd or showed the first regular
  image's masks on a slice.

### Two Temp-Annotation Mechanisms

The name "temp annotation" refers to **two unrelated mechanisms** in
this codebase. They look similar (both are pending, unsaved
detections shown for accept/reject review) and are easy to confuse,
but they live in different data structures, render through different
code paths, and are reviewed through different UI. When touching
prediction-review code, check which one you're actually in.

**Mechanism A — `ImageLabel.temp_annotations` (a list).** Populated
*only* by DINO's own single/batch detect-then-SAM flow (see above).
Rendered by a dedicated `ImageLabel.draw_temp_annotations` method,
which has no keypoints branch — it only understands `"segmentation"`
and `"bbox"`. Reviewed via the application-wide
`DINOReviewEventFilter` (ADR-015), keyed on Enter/Escape. Nothing in
YOLO uses this path.

**Mechanism B — `"Temp-{class}"` entries inside `image_label.annotations`
(a dict).** These are ordinary-looking annotation-dict entries, just
filed under a class name prefixed `"Temp-"`. Populated by
`add_temp_classes()` and used by **both** DINO's promote-to-permanent-
class flow **and** all of YOLO's box/mask/pose prediction review
(`YOLOController.process_yolo_results` builds the same `{class_name:
[annotation, ...]}` shape DINO does and calls `self.mw.add_temp_classes(...)`
directly). Because these entries sit in the real `annotations` dict,
they render through the ordinary `draw_annotations` — already
keypoint-aware since PR-1 (the `elif "keypoints"` branch) — so pose
predictions display correctly without any special-casing. Accept/reject
go through `accept_visible_temp_classes()` / `reject_visible_temp_classes()`,
which only move dict entries and rename keys — they do **not** call
`calculate_area`, `simplify_polygon`, or `create_coco_annotation`, and
have no existence-only `"segmentation" in annotation` checks, so
keypoint-shaped (segmentation-less) candidates pass through safely.
`accept_visible_temp_classes()` additionally carries over any
`keypoint_schemas["Temp-{class}"]` entry to the permanent class name
(warning and keeping the existing schema on a K mismatch instead of
overwriting); `reject_visible_temp_classes()` pops the orphaned
`"Temp-{class}"` schema entry.

**YOLO-pose prediction (issue #35 PR-3) uses only Mechanism B** — it
never touches `ImageLabel.temp_annotations` or
`DINOReviewEventFilter`. See [ADR-029](09_architecture_decisions.md#adr-029-keypoint--pose-annotation--per-class-schema-coco-instance-model-3-state-visibility)
for the pose instance schema itself.

## Multi-dimensional TIFF Axis Defaults

`load_tiff` extracts `tif.series[0].axes` (e.g. `"TZCYX"`) and maps
it through `{T:T, Z:Z, C:C, S:S, Y:H, X:W}` to populate the
`DimensionDialog` combo boxes. This is what lets a user open an
ImageJ-style 5D TIFF and just click OK.

When the metadata is missing or unfamiliar, fall back to the
hand-crafted defaults keyed on `ndim`:

| ndim | default labels |
|------|---------------|
| 3 | `Z H W` |
| 4 | `T Z H W` |
| 5 | `T Z C H W` |
| 6 | `T Z C S H W` |

**Do not** use `default_dimensions[-ndim:]` of a shorter list to
"extend" defaults — that silently degrades for `ndim ≥ 5`: the final
combo gets no default and inherits the first item ("T"), which is
the wrong axis. The 5D TZCYX bug that produced 2560 one-row slices
on a `(2,5,2,256,256)` file came from exactly this.

## Export Format Filename Matching

`export_formats.py` historically looked up image paths via substring
match:

```python
image_path = next(
    (path for name, path in image_paths.items() if image_name in name),
    None,
)
```

That is fragile — `"bee.jpg" in "honeybee.jpg"` returns True and you
write the wrong file. The COCO, YOLO v4, and YOLO v5+ exports all
share this code path.

**Always try the exact key first; fall back to substring only if no
exact key matches.** Pattern:

```python
image_path = image_paths.get(image_name)
if image_path is None:
    image_path = next(
        (path for name, path in image_paths.items() if image_name in name),
        None,
    )
```

The substring fallback is kept for backward compatibility with old
projects that may have stored normalised image names (e.g. without
extension); new code should prefer the exact-key path.

## Image List Filter — Hide Rows, Never Remove Them

The image list can be filtered by annotation status (combo above the
list; upstream issue #27, `ImageController.apply_image_filter`). The
filter uses `setRowHidden(i, True)` and must **never** remove items:

- `DINOController._navigate_to_image_or_slice` and the COCO importer
  iterate `image_list` rows by index; removing rows would shift
  indices under them.
- Removing the current item fires `currentRowChanged`, which is wired
  to `switch_image` — a filter change could silently switch the
  displayed image. Hiding fires nothing.

A non-matching row is hidden **even when it is the current selection**.
`setRowHidden` does not change `current_image` or fire
`currentRowChanged`, so the canvas keeps showing the worked-on image
while its row leaves the list — e.g. the current image gains its first
annotation under the "Without annotations" filter and disappears from
the list, but stays on screen until the user navigates away. Keyboard
nav skips hidden rows. (Guaranteed by
`test_hiding_current_row_keeps_canvas_and_fires_no_switch`.)

Re-apply runs from `ClassController.update_slice_list_colors()`. The
contract: every annotation-mutation site either calls that method
directly **or** emits `annotationsBatchSaved` (whose handler
`_on_annotations_batch_saved` calls it). All `annotationCommitted`
emitters follow up with `annotationsBatchSaved`
(image_label.py / paint_tool.py), so both commit paths are covered.
New mutation paths must keep one of those two routes — don't add
bespoke `apply_image_filter()` call sites.

## Lazy Slice Materialisation — Name-Only vs Pixel Consumers (issue #45, ADR-036)

Multi-dim `image_slices[base]` is a `LazySliceList` (`core/slice_cache.py`), not a
list of `(name, qimage)` tuples — QImages decode on demand through a shared bounded
LRU (`LRU_CAPACITY = 8`). It stays interface-compatible (`__iter__`, `__getitem__`,
`__len__`, `__bool__`, `.names`, `.get(name)`, `prefetch_around`), but there is a
critical distinction for anyone touching slice code:

- **Name-only consumers must never iterate for pixels.** Saving a project, computing
  annotation status, rebuilding the slice list, and navigation membership checks need
  only slice *names*. Use `slice_cache.slice_names(slices)` (or `.names`) — iterating
  `for name, _ in slices` would materialise every QImage. `save_project` is
  regression-tested to call `SliceProvider.extract` **zero** times.
- **Pixel consumers materialise one at a time.** `switch_slice`/`activate_slice` use
  `slices.get(name)` + `prefetch_around(name)`; exporters, the SAM-dataset build and
  DINO batch iterate via `__iter__` (which feeds/evicts the LRU). A batch export still
  transiently collects a stack's QImages into its own `slice_map` — that is an accepted
  per-operation cost; the SESSION memory is what the LRU bounds.
- **Every "drop a stack" path must release.** `remove_image`/`delete_selected_image`/
  `redefine_dimensions` call `slice_cache.release_slices(...)` before dropping the stack;
  `open_images` (which replaces the whole dataset) and `clear_all` clear the entire shared
  LRU **and** `image_slices` — because a `LazySliceList` pins its whole decoded source
  array under Strategy A, merely rebinding `mw.slices = []` would leak the outgoing stack
  for the session.
- `mw.slices` and `image_slices[base]` are the **same** `LazySliceList` object, and
  slice naming is byte-identical to the old eager path (it is the annotation key +
  export filename). Extraction returns a fresh QImage each call — never mutate a cached
  one (the SAM worker may be reading it, ADR-013).

## Video Frames as Lazy Slices (issue #47, ADR-037)

A video is a stack whose slices are frames, and it **reuses** the #45 lazy machinery
rather than a parallel cache: `image_slices[base]` is a `LazySliceList` backed by a
`core/video_handler.py::VideoSliceProvider` (duck-compatible with `SliceProvider`), so
every slice consumer above works for video unchanged. Practical rules:

- **Never read a frame's pixels directly.** Go through the `LazySliceList`
  (`slices.get(frame_key)` / `__iter__`), which decodes via `VideoHandler.get_frame`
  and caches in the shared `SliceLRU`. Frame keys are `f"{base}_F{idx:05d}"`
  (`video_handler.frame_key` / `parse_frame_index`, regex anchored at end so a
  `stack_T1_Z5` key never parses as a frame).
- **Handlers are a separate resource that must be released.** `mw.video_handlers[base]`
  holds the open `cv2.VideoCapture`; release it on every drop path
  (`remove_image`/`delete_selected_image`/`redefine_dimensions`/`open_images`/
  `clear_all`) alongside `release_slices`.
- **GUI-thread only.** `cv2.VideoCapture` is not thread-safe; `get_frame` seeks +
  decodes synchronously on the main thread (same discipline as `SliceProvider.extract`),
  and always returns a fresh `.copy()` QImage (the numpy buffer dies at return; the SAM
  worker may read a cached one, ADR-013).
- **Base-name collision** (`video.mp4` vs `video.tif` → same `image_slices` key) is
  refused in `add_images_to_list`.

## Video Timeline — A Pure View (issue #48)

`widgets/video_timeline.py::VideoTimeline` is a scrub slider + annotated-frame
marker strip + `F i/N • MM:SS / MM:SS` label shown under the canvas for videos
only. It is a **view**, never a second source of truth for the current frame:

- **All frame changes route through `switch_slice`.** User scrubbing emits
  `frameSelected(idx)`; the orchestrator's `on_timeline_frame_selected` maps
  idx→slice-list row and calls `switch_slice` (keeping unsaved-changes checks,
  annotation save, and the DINO temp re-sync intact). The timeline never writes
  `current_image`.
- **No signal feedback loop.** `set_current_frame` (called from `switch_slice`
  via `update_video_timeline`) wraps `slider.setValue` in an `_updating` guard
  so the programmatic sync can't re-emit `frameSelected` and re-enter
  `switch_slice`. `set_video` is guarded the same way, so reconfiguring the
  timeline on every annotation mutation never spuriously jumps to frame 0.
- **Marks derive from `all_annotations` keys**, computed by
  `ImageController.annotated_frame_indices(base)` (prefix `base + "_F"` +
  `parse_frame_index`, exact) and refreshed at the `update_slice_list_colors`
  choke point — the same hook the slice list uses — so they stay correct after
  every save/accept/undo/delete without a frame switch.
- **Palette-derived colours** (`highlight`/`mid`/`text`), no literals. The app
  themes via QSS (no `setPalette`), so `highlight`/`mid` are effectively the
  static default palette — chosen because a saturated accent + mid-grey both
  read on the light and soft-dark backgrounds — while the `text` role (the
  high-contrast current-frame tick) does follow the stylesheet. The slider has
  `NoFocus` so it never swallows arrow-key slice nav.

## Image List Groups & Status Badges (issue #43)

The image list gained two at-a-glance affordances, both layered on the
existing filter/sort wiring — **no tree widget, no thumbnails, no header
rows**:

- **Status badges**: `ImageController.refresh_image_status_icons()` sets a
  small painted-pixmap `QIcon` per row — a filled green dot when
  `image_has_annotations(info)` (any slice of a stack counts), a hollow gray
  outline otherwise. Nothing is stored; both states are derived. Icons are
  painted once per `(annotated, dark_mode)` into a cache; because they are
  **painted pixmaps, not stylesheet colours**, the No Hardcoded Colors Rule
  isn't violated — but the cache is cleared and rebuilt when the theme flips
  (`on_theme_changed`, called from `ui/theme.py::toggle_dark_mode`). The
  refresh runs at the end of `apply_image_filter()` (so every annotation
  mutation repaints badges via the `update_slice_list_colors →
  apply_image_filter` contract) and at the end of `sort_image_list()`. The
  dot colours are theme-tuned (a brighter green / lighter gray on the dark
  sidebar), which is what makes the `(annotated, dark_mode)` cache dimension
  and the `on_theme_changed` rebuild do real work.
- **Named groups**: an optional `"group"` string key on each `all_images`
  entry (no registry; the group set is derived). `set_image_group` sets/clears
  it, re-sorts, and auto-saves (guarded by `is_loading_project`). The
  context menu ("Move to group…" / "Remove from group") drives it. Grouped
  images cluster via the `sort_image_list` key
  `(group.casefold(), name.casefold(), name)`; the group shows only in the
  row **tooltip** — the item TEXT stays the bare file name, because
  `item(i).text()` is read as a filename by DINO batch navigation and COCO
  import reconciliation. A second combo (`image_group_combo`) filters by
  group; `apply_image_filter` hides a row when **either** the status filter
  or the group filter excludes it (OR of the two hide flags), keeping index 0
  of both combos as the cheap "hide nothing" default. Groups persist in the
  `.iap`; on load no restoration step is needed — `load_project_data` aliases
  `all_images` to the parsed `project_data["images"]` and the load loop does
  not rebuild it (`add_images_to_list` no-ops because `image_paths` is set
  first), so the `"group"` keys survive as-is.

## Image List Sorting — Rebuild, Don't `setSortingEnabled`

The image list is kept alphabetical (upstream issue #60,
`ImageController.sort_image_list`). Two constraints shape the
implementation:

- `currentRowChanged` is wired to `switch_image`, so `setSortingEnabled(True)`
  is forbidden — a live re-sort would reorder rows and fire spurious
  image switches.
- COCO import (and other positional lookups) assume `all_images[i]`
  matches `image_list.item(i)`. So the model and the view are sorted
  **together**: `all_images` is sorted, then the list is cleared and
  repopulated from it with `blockSignals(True)` around the rebuild, and
  the prior (or newly added) selection is restored explicitly. The #27
  filter is re-applied at the end of the rebuild.

`update_image_list` routes through `sort_image_list`; `add_images_to_list`
calls it with the first added file selected. It is skipped per-image
during project load (the list is rebuilt once via `update_ui`) to avoid
an O(n²) re-sort.

## TIFF Compression Codecs

Reading an LZW- (or otherwise) compressed TIFF requires the optional
`imagecodecs` package; without it `tifffile` raises `ValueError` mid-read
(upstream issue #56). `imagecodecs` is now a hard dependency, but
`ImageController.add_images_to_list` also catches the codec `ValueError`
(`_is_missing_codec_error`) and shows an actionable "pip install
imagecodecs" dialog, skipping the file instead of crashing or leaving a
half-added entry. Non-codec `ValueError`s still propagate.

## Canvas Decoupling — Signals + CanvasContext

`ImageLabel` (the canvas widget) does **not** hold a reference to
`ImageAnnotator`. Communication is split:

- **Writes** (committing an annotation, requesting a SAM prediction,
  asking for tools to be re-enabled, etc.) leave the widget as Qt
  `pyqtSignal` emissions. The signal block at the top of `ImageLabel`
  documents every outbound interaction. `ImageAnnotator` connects
  each signal to the right controller slot once, in
  `_connect_image_label_signals` (called at the end of
  `ImageAnnotator.__init__`).
- **Reads** (`paint_brush_size`, `current_class`, `class_mapping`,
  `is_class_visible`, `scroll_area`, etc.) go through a
  `CanvasContext` object passed in via
  `image_label.set_context(CanvasContext(self))`.
  `CanvasContext` wraps the main window rather than copying state,
  so updates made by controllers are visible on the next read.

**Why both mechanisms.** Signals are inherently one-way (fire and
forget); a synchronous read like "is this class visible" needs a
return value, which signals don't provide. Trying to express reads
as request/response signals adds latency and ordering bugs. The
`CanvasContext` accessor list is small (~10 methods) and stable.

**Rules for adding traffic in either direction**:

- New write from canvas → orchestrator: declare a `pyqtSignal` on
  `ImageLabel`, add a slot on a controller, wire it in
  `_connect_image_label_signals`. Do not add a back-reference to
  `ImageAnnotator`.
- New read from canvas → orchestrator: add a method on
  `CanvasContext`. Do not expose `_ctx._mw` directly.

**Synchronous-emit ordering**. Qt's default `AutoConnection` runs the
slot synchronously when the sender and receiver share a thread (true
for everything on the GUI thread). Code that emits a signal and then
reads state expected to be updated by it is correct — the slot has
already run by the time `.emit()` returns. This is load-bearing for
`accept_temp_annotations`, where `classRequested` must complete
before the subsequent class lookup.

**Batch save signal**. Paint commits and accept-temp commits emit
`annotationCommitted` per annotation but `annotationsBatchSaved` only
once at the end. The single batch save preserves O(1) `.iap` writes
per user action; replacing it with a per-annotation save would turn
paint commits into O(N). See ADR-018.

See ADR-018 in `09_architecture_decisions.md` for the rationale and
the full pattern.

## Canvas Selection ↔ List Selection

When no drawing tool is active (`ImageLabel._is_select_mode()`), the canvas
behaves like a pointer: a single click selects the smallest mask under the
cursor, a drag draws a rubber band that box-selects, and **Shift** toggles /
adds. This is wired so there is **one** selection shared by the canvas overlay
(`highlighted_annotations`, blue selection outline + handles) and the bottom-left
annotation list — so
`Delete` / `Merge` / `Change Class` (which read `annotation_list.selectedItems()`)
work identically whether you selected on the image or in the list. See ADR-022.

Flow: `ImageLabel` emits `canvasSelectionChanged(annotations, mode)` (mode =
`replace` | `add` | `toggle`) → `AnnotationController.apply_canvas_selection`
computes the new set, assigns `image_label.highlighted_annotations`, and mirrors
it onto the list.

Two non-obvious rules make this correct:

- **Match by value-equality, never identity.** `image_label.annotations` is a
  `deepcopy` of `all_annotations`, and PyQt round-trips dicts stored in a list
  item's `UserRole` as *copies* — so the "same" annotation has different object
  identity on the canvas, in `all_annotations`, and in a list item. Every
  selection comparison therefore uses dict `==` (`a == b`), the same convention
  as `select_annotation_in_list`, `delete_selected_annotations`, and the
  `annotation in highlighted_annotations` test in `draw_annotations`. A
  consequence: two value-equal duplicate masks select together — accepted, and
  pre-existing.
- **Block list signals while mirroring.** `apply_canvas_selection` wraps the
  programmatic list selection in `annotation_list.blockSignals(True/False)`.
  Without it, `setSelected` fires `itemSelectionChanged` →
  `update_highlighted_annotations`, which would overwrite the freshly-computed
  set with the list items' own objects (and clobber a `toggle`).

**The Annotations panel is a `QTableWidget`** (ID | Class | Area | Detail %), not
a `QListWidget` (issue #24, ADR-025). The bridge above is unchanged in spirit but
the API maps to table calls: the annotation dict lives in **column 0's UserRole**;
`count()/item(i)/selectedItems()` become `rowCount()/item(r, ANNOT_COL_ID)/row-
deduped `selectedIndexes()`; and the mirror uses **`setRangeSelected` (additive)**,
because `selectRow()` *replaces* the selection in `ExtendedSelection` mode and
would drop all but the last row. `blockSignals` + value-equality are preserved.

**Ctrl is reserved for pan.** Multi-select uses **Shift**, not Ctrl, because
Ctrl+drag is the pan gesture (whose reference-frame handling is deliberately
delicate — see [Pan + Zoom Reference Frames](#pan--zoom-reference-frames)).
Leaving Ctrl untouched keeps that gesture intact.

**Selection is drawn independent of class colour.** A selected mask is *not*
recoloured (the first version turned it red, which vanished on a red-class mask,
and red was the default first class colour). Instead `draw_annotations` keeps the
class colour and, in a final pass on top of every mask, draws a dashed
selection-blue **bounding-box marquee plus bright handle squares** at the 4
corners + 4 edge midpoints (`_SELECTION_COLOR`, `_draw_selection_overlay`) —
modelled on the sibling open-garden-planner app's CAD selection. The handles
carry the visibility (a single thin dashed outline was too faint). For a **single
selected shape** those same handle squares are resize grab targets (and the
interior is a move target) — `_draw_selection_overlay` and the `_bbox_handle_at`
hit-test share `_bbox_handle_points`, so the squares you see *are* the targets;
see ADR-023. Resizing scales a polygon's vertices (a box edits `[x,y,w,h]`);
reshaping a polygon vertex-by-vertex is still double-click vertex edit.
This never collides with any class colour. Relatedly, the default class palette
(`core/constants.py`) was reordered so red is last and the fill opacity lowered
to keep the image legible — see the No Hardcoded Colors Rule for the broader
"don't fight the theme/colours" theme.

## Tool Activation — One Choke-Point, Mutually Exclusive

**Pose classes admit only the keypoint tool (#44).** Activating a shape tool
(`toggle_tool`) or a SAM tool (`SAMController.toggle_sam_*`) while a pose class is
selected is refused (button unchecked); selecting a pose class while a shape/SAM tool is
active deactivates it (`ClassController.on_class_selected`). Editing a schema is still
allowed on a legacy-mixed class. See ADR-029 Guards.

All six canvas tools (Polygon, Rectangle, Paint, Eraser, SAM-box, SAM-points)
go through a **single** activation method, `ImageAnnotator.activate_tool(name)`
(`name=None` = selection mode). It is the only place `current_tool`, the SAM
flags (`sam_box_active` / `sam_points_active`), and the toolbar button checks
change, so they can never drift apart. `activate_tool`:

1. checks exactly one tool button (block-signals around `setChecked`, so the
   programmatic check doesn't re-enter `toggle_tool` / `toggle_sam_*`),
2. clears SAM transient state unless entering that SAM tool (sets the flag if it is),
3. calls `image_label.set_active_tool(name)` (deactivates the previous handler),
4. updates cursor + `update_ui_for_current_tool`.

`toggle_tool` (manual buttons) and `toggle_sam_box` / `toggle_sam_points` (SAM,
in `SAMController`) all delegate here. Before this, the `QButtonGroup` was
non-exclusive and the SAM toggles had their own ad-hoc state writes, so a SAM
tool could be active **at the same time** as a manual tool (both buttons checked,
both overlays live). The group stays `setExclusive(False)` because we need
click-to-toggle-off; exclusivity is enforced by `activate_tool` unchecking the
others. `_is_select_mode()` keys off `current_tool is None` + cleared SAM flags,
so it is correct once those are always in sync.

## Esc Returns to Selection Mode

Selection mode (no tool, click-to-select masks) is the canvas default. Pressing
**Esc** now both cancels any in-progress state **and** deactivates the active
tool, so a single Esc always lands you back in selection mode (previously the
tool stayed selected and you had to click its button off). `ImageLabel`'s Escape
handler clears the gesture (SAM points/box, in-progress polygon/paint/eraser) and
then emits `selectModeRequested` when a tool is active; the window's
`_on_select_mode_requested` calls `activate_tool(None)`. Exceptions that stay put:
polygon vertex-edit exit (already selection mode, uses `enableToolsRequested`),
DINO temp-review reject, and cancelling a rubber-band drag.
