# Architecture Decisions

## ADR-001: GUI Framework Choice

**Status**: Superseded by [ADR-014](#adr-014-migrate-from-pyqt5-to-pyqt6)

**Original decision (historical)**: Use PyQt5 5.15.11. Chosen because the upstream project used PyQt5, PyQt5's ecosystem was more mature at the time, and migration carried risk.

**Superseding decision**: The project migrated to PyQt6 6.7+ in the same PR that introduced in-process AI inference. See [ADR-014](#adr-014-migrate-from-pyqt5-to-pyqt6) for the rationale. The migration unblocked the subprocess removal in [ADR-013](#adr-013-in-process-inference-with-qthread-wrapping); note, however, that PyQt6 did **not** by itself eliminate the WinError 1114 DLL load-order conflict — that conflict persists and is handled by importing torch before Qt in `main.py` (see [ADR-017](#adr-017-eager-torch-import-in-mainpy-before-qapplication-creation)).

---

## ADR-002: Use Ultralytics for SAM Integration

**Status**: Accepted

**Context**: Need to integrate Segment Anything Model 2 for semi-automated annotation

**Decision**: Use Ultralytics library instead of direct SAM2 installation

**Version bound** (reconciled in the pyproject migration, #38): pinned as
`ultralytics>=8.3.27,<9` in `pyproject.toml` — the lowest version the code is
documented against, capped below the next major. The SAM fine-tuning loop
(ADR-021) was verified on 8.4.51, which remains satisfiable.

**Rationale**:
- Simplifies SAM model loading (single line)
- Includes PyTorch dependencies
- Automatic model caching
- No manual model download required
- Supports both SAM 2.0 and SAM 2.1 variants

**Consequences**:
- ✅ Simplified installation (no separate SAM2 setup)
- ✅ Automatic model management
- ✅ Consistent API
- ⚠️ Dependency on Ultralytics release cycle

---

## ADR-003: Store Absolute Paths in Project Files

**Status**: Superseded by ADR-033 (dual absolute + relative paths for portability, #42)

**Context**: Project files need to reference image locations

**Decision**: Store absolute paths to images in project JSON

**Rationale**:
- Images can be anywhere on filesystem
- No requirement to keep images with project file
- Simplifies project structure

**Consequences**:
- ✅ Flexible image locations
- ❌ Projects not portable between machines
- ❌ Moving images breaks projects

**Mitigation**: Export functions copy images to output directory

---

## ADR-004: No Automated Testing Framework

**Status**: Superseded — the project now has a pytest + pytest-qt suite

**Superseding note**: This decision no longer holds. The repository has a real
automated test suite under `tests/` (`unit`, `integration`, `ui`) — boot smoke,
coordinate conversions, export/import round-trips, controller state machines,
project save/load, multi-dim slicing, and the DINO/SAM/YOLO wiring — run in CI
on 3 OS × Python 3.10-3.14 (`.github/workflows/tests.yml`). An AST-based
inline-import gate guards refactors (see
[ADR-016](#adr-016-static-ast-inspection-of-inline-imports-as-quality-gate-for-refactor-prs)).
Run headless with `QT_QPA_PLATFORM=offscreen pytest tests/ -v`.

### Original decision (historical)

**Status**: Accepted (Technical Debt)

**Context**: Application is GUI-heavy with complex interactions

**Decision**: Rely on manual testing only

**Rationale**:
- PyQt testing requires significant setup (pytest-qt, fixtures)
- Visual nature of tool makes automated testing difficult
- Small development team
- Rapid iteration on features

**Consequences**:
- ❌ Risk of regressions
- ❌ Manual testing required for all changes
- ❌ Slower development velocity for large refactors
- ✅ Lower initial development overhead

**Future Consideration**: Add unit tests for utility functions (calculate_area, conversions)

---

## ADR-005: Disable Autosave During Project Loading

**Status**: Accepted

**Context**: Projects were getting corrupted when application terminated during loading (v0.8.9 bug)

**Decision**: Set `is_loading_project` flag to disable autosave during load

**Rationale**:
- Autosave triggered with partially loaded state corrupts file
- Loading large projects is slow, increases risk
- Simple flag prevents the issue

**Consequences**:
- ✅ Prevents project corruption
- ✅ Minimal code change
- ⚠️ Users lose autosave protection during load window

---

## ADR-006: Use Shapely for Polygon Operations

**Status**: Accepted

**Context**: Need to merge, validate, and manipulate polygon geometries

**Decision**: Use Shapely library for all polygon operations

**Rationale**:
- Industry-standard computational geometry library
- Handles invalid polygons gracefully
- Efficient union/intersection operations
- Well-tested algorithms

**Consequences**:
- ✅ Robust polygon handling
- ✅ Easy merge operations
- ✅ Automatic polygon validation
- ⚠️ Additional dependency

---

## ADR-007: Flatten Polygon Coordinates in Storage

**Status**: Accepted

**Context**: Need to store polygon annotations

**Decision**: Store as flattened list `[x1, y1, x2, y2, ...]` instead of nested `[[x1, y1], [x2, y2], ...]`

**Rationale**:
- Compatible with COCO JSON format
- Smaller file size
- Standard in annotation tools

**Consequences**:
- ✅ COCO compatibility
- ✅ Compact representation
- ⚠️ Must convert to/from paired format for some operations

---

## ADR-008: Support Multiple Export Formats

**Status**: Accepted

**Context**: Users need annotations in different formats for various ML frameworks

**Decision**: Implement exporters for COCO, YOLO, Pascal VOC, labeled images, semantic labels

**Rationale**:
- Different frameworks have different input requirements
- YOLO and COCO are most common
- Labeled images useful for visual verification
- Semantic labels needed for segmentation models

**Consequences**:
- ✅ Wide compatibility
- ✅ Flexible workflow
- ⚠️ More code to maintain
- ⚠️ Must keep up with format changes (e.g., YOLOv11)

---

## ADR-009: Use Per-Slice Annotation Storage for Multi-dimensional Images

**Status**: Accepted

**Context**: TIFF stacks and CZI files have multiple slices that need individual annotations

**Decision**: Store annotations per slice with naming convention `{filename}_T{t}_Z{z}_C{c}`

**Rationale**:
- Each slice is effectively a separate 2D image
- Simple extension of existing single-image annotation
- User can navigate and annotate independently

**Consequences**:
- ✅ Simple mental model (each slice = image)
- ✅ Reuses existing annotation code
- ⚠️ Large stacks create many entries in annotations dict
- ⚠️ No 3D annotation support

---

## ADR-010: Normalize 16-bit Images to 8-bit for Display

**Status**: Accepted

**Context**: SAM and display require 8-bit images, but microscopy often uses 16-bit

**Decision**: Normalize 16-bit to 8-bit using percentile clipping

**Rationale**:
- SAM models trained on 8-bit RGB images
- Displays only show 8-bit effectively
- Percentile clipping (2nd-98th) provides better contrast than linear

**Consequences**:
- ✅ Better visual contrast
- ✅ SAM compatibility
- ⚠️ Information loss (quantization)
- ⚠️ Different normalization per image/slice

---

## ADR-011: Run Torch-based Workers in Isolated Subprocesses

**Status**: Superseded by [ADR-013](#adr-013-in-process-inference-with-qthread-wrapping)

**Context**: Both SAM 2 (via Ultralytics) and Grounding DINO (via transformers) load PyTorch into the process. On Windows + Python 3.14, importing PyQt5 first and then loading PyTorch causes `WinError 1114` (DLL load order conflict between Qt and Torch native dependencies). The application is fundamentally PyQt5-based, so we cannot reorder these imports.

**Decision**: Run each ML model in its own subprocess script that has no PyQt5 imports — `sam_worker.py` for SAM and `dino_worker.py` for DINO. The parent GUI process speaks to each worker over stdin/stdout with JSON requests and responses.

**Rationale**:
- The DLL conflict only manifests when both libraries are loaded in the same process. Splitting them across processes avoids the issue entirely.
- Keeps the GUI responsive: heavy model loading doesn't block PyQt's event loop in the same address space.
- Lets us swap or upgrade torch/transformers/ultralytics versions without worrying about Qt interactions.
- The JSON-over-stdio protocol is simple, language-agnostic, and easy to debug — just inspect what the worker prints.

**Consequences**:
- ✅ Works reliably on Windows + Python 3.14 (the original motivating bug)
- ✅ Worker scripts are PyQt-free; they can be tested independently
- ⚠️ Per-inference subprocess spawn cost (~1-2 s startup + first model load)
- ⚠️ Need UTF-8 forced on both ends of the pipe (`PYTHONIOENCODING=utf-8` in env, `encoding="utf-8", errors="replace"` on parent) — Windows cp1252 default crashes on non-ASCII bytes in torch warnings
- ⚠️ Two near-identical worker scripts to maintain (`sam_worker.py` mirrors the pattern from `dino_worker.py`)

**Superseded by**: Migrating to PyQt6 (ADR-013) eliminated the underlying DLL conflict. The subprocess hop, JSON marshalling, and `check_worker_isolation.py` tooling were removed in the same PR.

**Related**:
- Implementation (historical): `sam_utils.py` / `sam_worker.py`, `dino_utils.py` / `dino_worker.py`
- Original SAM-only version landed in #65 (Python 3.14 support)
- DINO subprocess pattern landed alongside the DINO feature

---

## ADR-012: Lazy Model Load on Dropdown Selection

**Status**: Accepted

**Context**: Both SAM and DINO model weights are large (SAM 2 tiny ~80 MB up to large ~400 MB; Grounding DINO base ~1.9 GB) and may not exist on first run. An earlier DINO flow required an explicit "Load" button click that did the resolve-or-download dance synchronously before the user could detect anything.

**Decision**: Selecting a model from the dropdown only updates state. Actual downloads happen on first use (first Detect call). UI feedback in the status label distinguishes "Ready: <model>" (weights present) from "<model> — will download on first detection".

**Rationale**:
- Matches the existing SAM behaviour (`change_sam_model` just stores the name; download happens in the worker).
- Removes a redundant click — one fewer thing for users to discover.
- Selecting a model the user picked by mistake is now free; only confirmed Detect triggers the (potentially heavy) download.

**Consequences**:
- ✅ Consistent UX between the SAM and DINO panels
- ✅ Faster perceived startup; no spurious downloads from idle browsing
- ⚠️ First Detect after selection blocks the UI while download runs (~1 min for DINO base); the status label shows progress but the dialog is otherwise unresponsive
- ⚠️ No async download progress dialog — `huggingface_hub` prints to stdout

---

## ADR-013: In-process Inference with QThread Wrapping

**Status**: Accepted

**Context**: ADR-011 introduced a subprocess hop for every SAM and DINO inference call to work around a PyQt5 + Torch DLL load-order conflict on Windows + Python 3.14. The workaround cost a fresh `python sam_worker.py` / `dino_worker.py` spawn per inference (~1-2 s warm latency, model reloaded from disk on every call) plus a temp-PNG marshal of the image.

Migrating the GUI from PyQt5 to PyQt6 (same PR) was expected to eliminate the DLL conflict — initially verified by `tools/check_pyqt6_torch_coexistence.py` importing PyQt6 packages → torch cleanly. However, further testing (see [ADR-017](#adr-017-eager-torch-import-in-mainpy-before-qapplication-creation)) discovered that the conflict resurfaces when Qt's **platform plugin** is loaded before torch, which happens inside `QApplication()`. The practical workaround is to import torch eagerly before creating the QApplication.

**Decision**: Run SAM and DINO inference directly inside the main Python process. Keep the model objects on the `SAMUtils` / `DINOUtils` singletons so they persist across calls. Wrap each inference in a short-lived `QThread` to keep the UI thread responsive; the public API blocks the caller via a nested `QEventLoop` so call sites in `annotator_window.py` stay synchronous-looking.

**Rationale**:
- The latency win is the whole point. Subprocess spawn + Python startup + model reload was ~1-2 s every call; in-process with a cached model is ~50-500 ms.
- Threading via a nested `QEventLoop` (the `_run_sync` helper in `sam_utils.py`) lets the calling thread keep pumping events — timers, repaints, progress dialog cancels still work — while inference runs on the QThread. Existing call sites need no refactor.
- Torch and transformers are imported lazily on first inference, so app startup stays fast for users who never touch SAM/DINO.
- `_qimage_to_numpy` already exists; converting the QImage on the calling thread (not on the worker) keeps Qt objects single-threaded as required.

**Consequences**:
- ✅ Each inference is ~1-2 s faster on Windows; less dramatic on macOS/Linux but still smoother.
- ✅ Cached model survives between calls — opening a DINO model once costs once. The DINO model stays on its compute device (CPU or CUDA) for its full lifetime; the old worker shuffled CPU↔GPU per call, defeating the caching gain on PCIe. Call `DINOUtils.unload()` / `SAMUtils.unload()` to free GPU memory explicitly.
- ✅ UI stays responsive during batch DINO+SAM runs (the calling thread's `QEventLoop` still processes events).
- ✅ One source of truth per model — no more keeping `sam_utils.py` and `sam_worker.py` aligned.
- ✅ Exceptions from the inference worker (model load failures, CUDA errors) propagate out of `_run_sync` rather than being printed and silently turned into `None`. The `change_sam_model` error path in `annotator_window.py` actually catches now.
- ⚠️ A crash in torch (CUDA OOM, segfault) now takes the app down where the subprocess used to absorb it. Mitigation: inference is wrapped in `try/except` at the `_run_sync` boundary; the user sees an error dialog instead of a frozen UI.
- ⚠️ Model RAM stays resident until the user closes the app (or invokes the `unload()` method).
- ⚠️ Re-entrancy is a real hazard, addressed with belt-and-braces:
   - `_run_sync` sets a module-level `_inference_in_flight` flag and raises `InferenceBusyError` if re-entered. Same-thread re-entry can happen because the calling thread pumps its event loop while waiting (a timer fire, a click on an un-disabled widget, etc.). A `QMutex` would not help — same-thread re-acquisition deadlocks on a non-recursive mutex and is meaningless on a recursive one.
   - The known re-entry vector — the SAM debounce timer firing during an in-flight inference — is guarded at the call site: `apply_sam_prediction` in `annotator_window.py` carries its own `_sam_inference_in_flight` flag and skips. Batch DINO already disables its trigger buttons.
   - The two-layer design is intentional: the call-site flag handles the common case quietly; the `_run_sync` flag is the safety net that surfaces unknown re-entry vectors as a real exception rather than corrupting the model with concurrent `.forward()` calls (torch / ultralytics / transformers model objects are not thread-safe).

**Related**:
- Implementation: `sam_utils.py`, `dino_utils.py` (both refactored in the same PR that retires ADR-011).
- Smoke test: `tools/check_pyqt6_torch_coexistence.py` (gate that gated this whole change).
- Supersedes: [ADR-011](#adr-011-run-torch-based-workers-in-isolated-subprocesses).

---

## ADR-014: Migrate from PyQt5 to PyQt6

**Status**: Accepted

**Context**: The project shipped on PyQt5 5.15+ (ADR-001) from inception. Two pressures combined to motivate a migration:
1. The PyQt5 + Torch DLL load-order conflict on Windows + Python 3.14 (ADR-011) forced an entire subprocess isolation layer. It was hypothesised that Qt6's packaging would eliminate the conflict entirely, but real-world testing (see [ADR-017](#adr-017-eager-torch-import-in-mainpy-before-qapplication-creation)) showed the conflict persists when Qt's platform plugin is loaded before torch, regardless of whether PyQt5 or PyQt6 is the binding. The migration still removes PyQt5-specific issues (XCB plugin paths, enum namespacing drift).
2. PyQt5 is in maintenance mode. PyQt6 is the actively developed line, gets new Qt6.x features, and has better Linux native integration (XCB plugin paths in particular).

**Decision**: Migrate the GUI binding from PyQt5 (`>=5.15.0`) to PyQt6 (`>=6.7.0`). Land in a single PR alongside the subprocess-removal work (ADR-013), gated behind `tools/check_pyqt6_torch_coexistence.py` to confirm the DLL conflict is actually gone on Windows + Python 3.14.

**Rationale**:
- Two coupled changes share most of their cost (touching every file that imports PyQt5) so doing them in one PR avoids paying the migration tax twice.
- Most PyQt5→PyQt6 differences are enum namespacing (`Qt.AlignCenter` → `Qt.AlignmentFlag.AlignCenter`) and module relocations (`QAction` moves from `QtWidgets` to `QtGui`) — mechanical, codemod-able. The behavioural risk is in event APIs (`event.pos()` → `event.position()`, returning `QPointF` not `QPoint`) and a handful of removed widgets (`QDesktopWidget` → `QGuiApplication.primaryScreen()`).
- The pytest-qt suite (coordinate transforms, export/import round-trips, controller state machines, and boot smoke) serves as the regression safety net.

**Consequences**:
- ✅ Subprocess workers retired; inference is in-process with cached models (see [ADR-013](#adr-013-in-process-inference-with-qthread-wrapping)).
- ✅ Cleaner Linux story — `libxcb-cursor0` is required by Qt 6 (was optional under Qt 5), but the platform plugin path mess is gone.
- ✅ Long support runway: PyQt6 is the maintained binding.
- ⚠️ One-time migration cost: ~30 files touched, enum namespacing across `annotator_window.py` (300+ references), `event.pos()` → `event.position()` rewrite in `image_label.py`.
- ⚠️ PyQt6 is GPLv3 / commercial like PyQt5. Switching to PySide6 (LGPL) was considered and rejected to stay close to the existing `pyqtSignal`/`pyqtSlot` API.
- ✅ All `.exec_()` call sites in `src/` migrated to `.exec()` in the v0.9.0 fix-pack — the PyQt5 alias is gone from this codebase.

**Verification**:
- `tools/check_pyqt6_torch_coexistence.py` tests both import orders. The production order (torch first, then `QApplication`) must pass. The Qt-first order is the known-failing case and is checked only to document the environment. Run before merging on the Windows + Python 3.14 target.
- 65 tests pass on the new binding under `QT_QPA_PLATFORM=offscreen`.
- Full app constructs and renders headlessly; snake-game easter egg validates the `QDesktopWidget` → `QGuiApplication.primaryScreen()` replacement.

**Related**:
- Supersedes: [ADR-001](#adr-001-gui-framework-choice).
- Unblocks: [ADR-013](#adr-013-in-process-inference-with-qthread-wrapping).

---

## ADR-015: Application-wide Event Filter for DINO Review Shortcuts

**Status**: Accepted (v0.9.0)

**Context**: During DINO batch / single-image review, the user has
to accept (Enter) or reject (Escape) pending masks. The keyboard
handling was originally in `ImageLabel.keyPressEvent`, which only
fires when the canvas has focus. In practice the user clicks slice
entries, image entries, or buttons during review — focus moves to
those widgets and Enter is consumed locally (e.g. `QListWidget`
emits `itemActivated`), never reaching the canvas. The result: Enter
and Escape silently failed during the most common review workflow.

Three options were considered:

1. **Force focus back to the canvas on every UI interaction** —
   intrusive, breaks normal navigation (Tab/Arrow keys on lists), and
   fragile because Qt's focus chain is not always predictable.
2. **Global `QShortcut` with ApplicationShortcut context** — fires
   regardless of focus but unconditionally hijacks Enter / Escape,
   breaking modal dialogs (Enter activates default button) and inline
   editing in `QLineEdit` / `QInputDialog`.
3. **Application-wide `QObject` event filter** that intercepts only
   when DINO temp_annotations are pending, and only when the focused
   widget is not a text input and no modal dialog is active.

**Decision**: Option 3. Implement `DINOReviewEventFilter`, install it
on `QApplication.instance()` once at startup, and gate the
interception on three conditions: pending DINO temp_annotations,
no active modal widget, focus not on `QLineEdit`/`QTextEdit`.

**Consequences**:
- ✅ Enter/Escape works regardless of which widget holds focus during
  DINO review.
- ✅ Modal dialogs and text-input fields are unaffected.
- ✅ Pattern is reusable for any future "review pending state" feature.
- ⚠️ Adds a per-key-press function call cost to the entire app. The
  filter short-circuits in three cheap checks before any work, so the
  overhead is negligible (≤ a few μs per keystroke).
- ⚠️ Single global filter means future review-state features must
  share it or layer additional filters; if more review modes appear,
  collapse them into a strategy registry rather than installing
  multiple top-level filters.

**Related**:
- Implementation: `DINOReviewEventFilter` class in
  `controllers/dino_controller.py` (moved there in Phase 4b);
  `installEventFilter` call in `ui/shortcuts.py:install_event_filters`,
  invoked from `ImageAnnotator.__init__` (moved there in Phase 8).
- Cross-cuts: documented in
  [Cross-cutting Concepts → DINO Temp Annotations](08_crosscutting_concepts.md#dino-temp-annotations--single-field-many-images).

---

## ADR-018: Decouple ImageLabel from ImageAnnotator via Signals + CanvasContext

**Status**: Accepted (Phase 6 of the modular refactor)

**Context**: Before Phase 6, `ImageLabel.set_main_window(main_window)`
injected the orchestrator into the canvas widget, and the widget poked
~50 sites on `main_window` directly — both reading state
(`paint_brush_size`, `class_mapping`, `current_class`, `scroll_area`,
`current_slice`, `image_file_name`) and mutating it
(`all_annotations[name] = …`, `add_class(…)`,
`update_annotation_list()`, `save_current_annotations()`,
`update_slice_list_colors()`, `schedule_sam_prediction()`,
`zoom_in()`, `enable_tools()`, etc.). The coupling made:
- ImageLabel impossible to test in isolation without a
  whole-`ImageAnnotator` fixture.
- Every controller extraction (Phases 3–5) leak through `main_window`
  delegation pass-throughs, because deleting them would break the
  widget.
- The Phase 7 per-tool split (paint / eraser / polygon / rectangle
  handler classes) impractical, because each handler would need the
  same `main_window` reference and would multiply the coupling.

Three options were considered:

1. **Protocol / duck-typed callback object** — pass a small protocol
   with the methods ImageLabel needs. Strict, type-safe, but writes
   are still synchronous direct calls; the widget still knows the
   exact method names on the orchestrator.
2. **Defer the fix** — leave `main_window` for one more phase, accept
   the debt. Cheapest, but each subsequent refactor pays the cost.
3. **Qt signals for every write + a narrow read accessor object** —
   ImageLabel emits typed signals; the orchestrator connects each to
   a controller slot during `__init__`. Reads go through a
   `CanvasContext` object with method-style accessors.

**Decision**: Option 3. ImageLabel declares ~20 `pyqtSignal`s covering
annotation lifecycle, SAM, class, tool/UI state, navigation, and
batch finalisation. Reads go via a `CanvasContext` instance passed in
through `set_context(ctx)`. The previous `set_main_window` /
`self.main_window` field is removed entirely.

The connection block lives in `ImageAnnotator._connect_image_label_signals`,
called once at the end of `__init__` after every controller exists.
`CanvasContext` wraps the main window rather than copying state, so
the source of truth stays on `ImageAnnotator` and controllers see
their writes reflected on the next read.

**Consequences**:
- ✅ ImageLabel has zero `main_window` references; signals form the
  documented public write surface at the top of the class.
- ✅ ImageLabel is now testable in isolation by connecting signals
  to stub slots; no controller fixture needed.
- ✅ Phase 7 (per-tool handlers) can carve `mousePressEvent` /
  `mouseMoveEvent` etc. without each handler needing the orchestrator.
- ✅ Signal connections are explicit and grep-able — searching for
  `il.annotationCommitted.connect` finds the single wiring site.
- ⚠️ Two parallel mechanisms (signals for writes, `CanvasContext` for
  reads) need to be kept in step. The widget's signal block and
  `_connect_image_label_signals` must stay in sync; a missing
  connection is a silent no-op write.
- ⚠️ Signal connections rely on Qt's default `AutoConnection` semantics,
  which is synchronous within a single thread. Consumers that depend
  on a write taking effect before the next read (e.g. `classRequested`
  emit followed by `_ctx.class_id(name)` read) must stay on the GUI
  thread.
- ⚠️ The synchronous batch-save signal (`annotationsBatchSaved`)
  preserves the original O(1)-save-per-batch behaviour. Replacing it
  with per-annotation save would silently turn paint commits into
  O(N) saves. Future refactors must keep the batch boundary.

**Pattern for adding a new ImageLabel → orchestrator interaction**:

1. Add a `pyqtSignal(<args>)` to `ImageLabel`.
2. Add a slot method on a controller (or main window) with matching
   signature.
3. Wire it in `_connect_image_label_signals`.
4. Replace the previous direct call site in ImageLabel with
   `self.<signal>.emit(<args>)`.

**Pattern for adding a new read accessor**:

1. Add a method on `CanvasContext` returning the value.
2. Use `self._ctx.<accessor>()` at the read site in ImageLabel.

**Related**:
- Implementation: `widgets/canvas_context.py`,
  `widgets/image_label.py` (signal block lines 42–70),
  `annotator_window.py:_connect_image_label_signals`.
- Cross-cuts: documented in
  [Cross-cutting Concepts → Canvas Decoupling](08_crosscutting_concepts.md#canvas-decoupling--signals--canvascontext).
- Predecessor pattern: ADR-015 (DINO event filter) showed that
  ImageLabel can't reliably observe global keyboard state without
  help; ADR-018 generalises "explicit interaction surface, narrow
  read surface" to all canvas ↔ orchestrator traffic.

---

## ADR-019: Per-Tool Handler Classes inside ImageLabel

**Status**: Accepted (Phase 7 of the modular refactor)

**Context**: After Phase 6, `ImageLabel` no longer held a back-reference
to `ImageAnnotator`, but it still embedded four distinct annotation
tools (polygon, rectangle, paint_brush, eraser) as if/elif branches
spread across six event methods (`mousePressEvent`, `mouseMoveEvent`,
`mouseReleaseEvent`, `mouseDoubleClickEvent`, `keyPressEvent`,
`paintEvent`). Each tool also owned helper methods on the widget
(`start_painting`, `commit_paint_annotation`, `commit_eraser_changes`,
`finish_polygon`, `cancel_current_annotation`, …). Adding a new tool
meant touching all six event methods plus the widget's helper layer,
and the file had reached ~1,240 LOC.

Three options were considered:

1. **Keep tools as if/elif branches** — cheapest, but the widget keeps
   accruing every new tool's behaviour.
2. **Per-tool widget subclass** (one `QWidget` per tool, swap on tool
   change) — too heavy: tool switches would require teardown of the
   pixmap, scroll context, zoom factor, and the SAM/DINO/edit-mode
   sub-states that cut across tool selection.
3. **Per-tool handler classes** with a thin dispatcher on the widget.
   Plain Python objects (not QObjects); the widget keeps a
   `_tools: dict[str, ToolHandler]` and routes events to
   `active_tool_handler`. Tools emit through the widget's existing
   Phase 6 signals.

**Decision**: Option 3. Each tool becomes a subclass of `ToolHandler`
in `widgets/tools/`. The contract:

- Event hooks return `True` when consumed: `on_mouse_press`,
  `on_mouse_move`, `on_mouse_release`, `on_double_click`, `on_enter`,
  `on_escape`.
- `paint_overlay(painter)` renders in-progress state (paint mask,
  eraser mask, polygon-in-progress, rectangle preview).
- `has_unsaved_state()` / `commit()` / `discard()` participate in
  the widget's `check_unsaved_changes()` dialog.
- `deactivate()` runs when the user switches away from this tool;
  default is no-op (matches the pre-Phase-7 "silently drop temp state
  mid-stroke" behaviour).

**Deliberate non-decision: state ownership.** Tool handlers contain
only *behaviour*; the temp-state fields (`current_rectangle`,
`current_annotation`, `temp_paint_mask`, `temp_eraser_mask`,
`drawing_polygon`, `drawing_rectangle`, `is_painting`, `is_erasing`)
remain on `ImageLabel`. Reason: `AnnotationController.finish_rectangle`
and `finish_polygon` (Phase 5a) read `mw.image_label.current_rectangle`
and `mw.image_label.current_annotation` directly. Moving the state
onto the handlers would have required a parallel controller refactor.
Handlers mutate `self.label.X` for those fields; pure-tool state
(e.g. future tool-internal counters) can live on the handler. See
the architectural-smell note below.

**What stays on `ImageLabel` (intentional non-extraction)**:

- Navigation (zoom, pan, offset, scaled pixmap) — cross-cutting.
- SAM bbox / points state — activates from any tool via the SAM-box /
  SAM-points toggles, cuts across the main tools.
- Polygon edit mode (`editing_polygon`, `handle_editing_click`,
  `handle_editing_move`, `draw_editing_polygon`) — modal state
  orthogonal to tool selection; sets `current_tool = None` while
  active. Promoting this to a handler would tangle the modal flow.
- DINO `temp_annotations` + `accept_temp_annotations` —
  cross-cutting; already touched by ADR-015's event filter.
- `draw_tool_size_indicator` — small enough that splitting it across
  paint/eraser handlers buys nothing.

**`paintEvent` overlay pass**. Iterates **all** handlers'
`paint_overlay()`, not just the active one. Reason: pre-Phase-7 the
temp paint mask, temp eraser mask, and polygon-in-progress rendered
whenever their state was populated, regardless of `current_tool`.
Each handler's `paint_overlay` short-circuits when its state is empty,
so the iteration is cheap and the user can switch tools mid-stroke
without losing visual feedback.

**Consequences**:
- ✅ `image_label.py` shrinks from 1,239 to ~960 LOC. Adding a new
  tool now means: create one file in `widgets/tools/`, register it
  in `_tools`, wire a button in `annotator_window.py`. No event-method
  edits.
- ✅ Each tool can be unit-tested by instantiating the handler with
  a stub `label` carrying signals and `_ctx` — no controller fixture
  needed.
- ✅ Phase 6's signal contract (ADR-018) is unchanged: handlers emit
  via `self.label.<signal>.emit(...)`.
- ⚠️ **State leak across the widget boundary.** Handlers reach into
  `self.label.X` for state. The contract drifts toward "handler is a
  namespaced function bag." Mitigation: revisit if/when controllers
  are updated to ask the handler (e.g. `polygon_tool.points()`)
  instead of reading the widget's field.
- ⚠️ `deactivate()` is no-op by default. If you make it
  `discard()` later, audit the three call sites that still write
  `current_tool = None` directly (`ImageLabel.clear()`,
  `ImageLabel.start_polygon_edit`, three locations in
  `SAMController`) — they bypass `set_active_tool` and therefore the
  hook.
- ⚠️ `check_unsaved_changes` now iterates all handlers, not just
  paint/eraser. Polygon participates via `has_unsaved_state() = len > 2`
  (sub-3-point polygons are silently discarded on switch — they
  can't be saved anyway).

**Pattern for adding a new mouse-driven tool**:

1. Create `widgets/tools/foo_tool.py` with `class FooTool(ToolHandler):`.
2. Override the event hooks you need; emit via
   `self.label.<signal>.emit(...)` and read via `self.label._ctx.X()`.
3. Register in `ImageLabel.__init__`'s `_tools = {…, "foo": FooTool(self)}`.
4. Add a button in `ui/sidebar.py:build_sidebar` next to the existing
   tool buttons, register it in `window.tool_group`, and connect
   `clicked` to `window.toggle_tool`. Then add a branch in
   `ImageAnnotator.toggle_tool` that calls
   `self.image_label.set_active_tool("foo")` for that button (since
   Phase 8 the UI building lives in `ui/sidebar.py`, not on the
   orchestrator).

**Related**:
- Implementation: `widgets/tools/base.py`,
  `widgets/tools/{rectangle,polygon,paint,eraser}_tool.py`,
  `widgets/image_label.py:set_active_tool`,
  `widgets/image_label.py:paintEvent` overlay-iteration block.
- Predecessor: ADR-018 (Phase 6 signal decoupling) made this safe by
  removing the `main_window` reference; handlers don't need an
  orchestrator handle.
- Cross-cuts: documented in
  [Cross-cutting Concepts → Canvas Decoupling](08_crosscutting_concepts.md#canvas-decoupling--signals--canvascontext)
  (extended to describe the tool dispatcher).

---

## ADR-016: Static AST Inspection of Inline Imports as Quality Gate for Refactor PRs

**Status**: Accepted

**Context**: During Phase 1 of the modular refactoring (2025-06-10), 25 modules were moved into `core/`, `dialogs/`, `inference/`, `io/`, `ui/`, `widgets/` subpackages. The smoke tests (`test_smoke.py`) verified that every module could be imported at top-level. All 30 smoke tests passed. However, four stale **inline imports** inside method bodies were missed:

```python
# annotator_window.py — inside function bodies, NOT top-level
from .dino_utils import GDINO_MODEL_PATHS        # moved to .inference.dino_utils
from .annotation_statistics import ...           # moved to .dialogs.annotation_statistics
from .project_details import ...                 # moved to .dialogs.project_details
from .project_search import ...                  # moved to .dialogs.project_search
```

These imports were deferred until the specific UI action triggered the function (e.g. picking a DINO model from the dropdown). The smoke tests, which only import modules, never execute function bodies and therefore never resolved the inline `from .dino_utils` reference. The bug surfaced only in manual QA when selecting a DINO model.

**Decision**: Add a static AST analysis test (`test_annotator_window_inline_imports_are_resolvable`) that parses `annotator_window.py`, extracts every bare relative import (`from .module`), and asserts the module still exists in the package root. The test fails with the exact line number for any stale import, preventing silent runtime-only regressions from reaching CI.

**Rationale**:
- Top-level import rewrites are mechanical and easy to verify via module import.
- Inline imports inside method bodies are invisible to module-level import tests.
- Manual QA is the fallback for behaviour, not for mechanical import correctness.
- AST inspection is cheap (~1 ms), zero false positives for this codebase, and runs in every CI build along with smoke tests.

**Consequences**:
- 🛑 Regression now impossible: the 30th smoke test would have failed the PR before merge.
- 🔧 No runtime cost — purely static analysis.
- ⚠️ Only covers `annotator_window.py`. If other files use the same inline-import pattern, the test should be generalized (or each file that contains inline imports gets its own AST check). In this codebase, `annotator_window.py` is the only file with significant inline imports.
- ⚠️ Doesn't catch dynamic imports (`__import__`, `importlib.import_module`), but we don't use those.

**Related**:
- Implementation: `tests/integration/test_smoke.py` (`test_annotator_window_inline_imports_are_resolvable`).
- Cross-cuts: `CLAUDE.md` "Testing Checklist" updated to reference this test as a mandatory CI gate.

---

## ADR-017: Eager Torch Import in `main.py` before `QApplication` Creation

**Status**: Accepted

**Context**: ADR-011 and ADR-014 both discussed a DLL load-order conflict on Windows when PyQt and PyTorch share a process. The conflict was first observed with PyQt5 (ADR-011) and later claimed to be resolved by migrating to PyQt6 (ADR-014):

> "Qt6's packaging reshuffle eliminates it." — ADR-014
>
> "...verified by `tools/check_pyqt6_torch_coexistence.py` importing PyQt6 → torch → transformers → ultralytics cleanly in one process..." — ADR-013

This claim was based on testing at the time, but it tested the **wrong order**: importing PyQt6 *packages* before torch works even in Qt5. The actual failure mode is triggered only when Qt's **native platform plugin** is loaded, which happens inside `QApplication.__init__()`, not at `import PyQt6`. The earlier verification script did not call `QApplication()`, so it never exercised the real failure path.

Real-world testing with `torch 2.11.0+cu126 + PyQt6 6.10.2 + Python 3.14.2` on Windows 11 shows the conflict **still surfaces** when Qt's platform DLLs (e.g. `qwindows.dll`) are loaded BEFORE torch's `c10.dll`. The error is `OSError: [WinError 1114] A dynamic link library (DLL) initialization routine failed`.

**Root cause analysis**: Qt and torch both ship native DLLs that load into the same process. On Windows the DLL load order and address-space layout matter. When Qt's platform plugin claims certain memory slots or loads conflicting CRT libraries before torch does, torch's `c10.dll` init fails. The conflict is NOT between PyQt5 and torch per se — it is between Qt platform plugins and torch, regardless of whether the binding is PyQt5 or PyQt6.

**Decision**: Two complementary changes:

1. In `main.py`, eagerly `import torch` (with an `ImportError` fallback) **before** importing `QApplication` and creating the app. This ensures torch's DLLs claim their slot first.
2. In `__init__.py`, replace eager toplevel imports of `annotator_window`, `image_label`, and `sam_utils` with a `__getattr__`-based lazy loader. The package init runs before `main.py` when launched via the `sreeni` console script (`digitalsreeni_image_annotator.main:main`). If `__init__.py` eagerly imports modules that transitively import PyQt6 (e.g. `annotator_window`), Qt loads first and the `import torch` in `main.py` crashes with the same WinError 1114. Lazy loading defers the Qt import until someone actually accesses `pkg.ImageAnnotator`, which only happens after the torch-first guard has run.

**Verification**:
- `tools/check_pyqt6_torch_coexistence.py` now tests both orders:
  1. `torch` → `QApplication` (production order) — **PASS**.
  2. `QApplication` → `torch` (the claimed-safe order) — **FAIL** on Windows with torch 2.11.0.
- Exit code 0 means production order works; exit code 1 means even torch-first fails and subprocess isolation (ADR-011) must be restored.
- Smoke test `test_public_api_exports` passes: `__getattr__` correctly resolves all five public names.

**Consequences**:
- ✅ SAM and DINO model loading works on Windows + Python 3.14 + PyQt6 without subprocess overhead.
- ✅ App startup cost is negligible — torch import adds ~0.5-1 s before the splash window appears, which is acceptable for a desktop annotation tool.
- ⚠️ `tests/integration/test_smoke.py` cannot import `main.py` because the pytest-qt test process already has Qt loaded; importing torch afterward triggers the same WinError 1114. `main.py` is therefore **excluded** from the module-import list and is validated by CLI smoke tests instead.
- ⚠️ Future Qt upgrades may change DLL packaging and make this unnecessary, but `check_pyqt6_torch_coexistence.py` will detect that automatically.
- ⚠️ Any new public name added to `__init__.py` must also be wired through `__getattr__` or it will transitively pull in PyQt6 and break the torch-first guard.

**Related**:
- Supersedes (in spirit): ADR-014's claim that PyQt6 eliminates the conflict.
- Unblocks: ADR-013 in-process inference on the affected Windows environment.
- Implementation: `src/digitalsreeni_image_annotator/main.py`.
- Gate: `tools/check_pyqt6_torch_coexistence.py`.

## ADR-020: App-Global UI Preferences via QSettings; Canvas Overlays Scale with `ui_font_pt`

**Status**: Accepted

**Context**: The low-vision accessibility feature (continuous UI font
zoom, 8–24pt) needed (a) the chosen size to survive app restarts and
(b) canvas overlay elements — annotation labels, SAM point markers,
pen widths — to grow with the setting. UI preferences were previously
reset on every launch, and the `.iap` project file was the only
persistence mechanism in the app.

**Decision**:
1. Introduce the app's first QSettings usage
   (`QSettings("DigitalSreeni", "ImageAnnotator")`, module
   `app_settings.py`) for `ui/font_pt` and `ui/dark_mode`. These are
   per-user preferences, so they do **not** go into the `.iap` file —
   a project opened by a different user must not impose a font size.
2. A single integer `ImageAnnotator.ui_font_pt` is the source of
   truth; the named presets and the step shortcuts both funnel
   through `theme.set_font_pt` (clamp → apply → persist → menu sync).
3. Canvas overlay sizes derive from `ui_scale = ui_font_pt / 10.0`
   (10 = the legacy default, so the default renders pixel-identical
   to the pre-feature code). `ImageLabel` receives the value via a
   plain setter from `apply_theme_and_font`, not via CanvasContext —
   consistent with the existing direct `image_label.setFont` call,
   and avoids a paint-before-context-set window.

**Alternatives considered**:
- Storing prefs in the `.iap` file — rejected: project files are
  shared artifacts; accessibility settings are personal.
- Templating the static stylesheets per font size — rejected:
  appended QSS override rules (later rules win at equal specificity)
  achieve the same with zero churn in the two stylesheet strings.

**Consequences**:
- ✅ Font size and dark mode persist across restarts.
- ✅ Tests stay hermetic: every `app_settings` function accepts an
  injectable `QSettings` (INI temp file) instance.
- ⚠️ Any new scalable UI metric should use `ImageLabel._pen_w` /
  `_overlay_font` or the appended-override block in
  `theme.apply_theme_and_font` — hardcoded px values won't follow the
  setting (see "UI Font Zoom" in `08_crosscutting_concepts.md`).
- ⚠️ Deliberately-compact widgets (DINO threshold table / phrase
  panel) don't hardcode their small font inline; the appended block
  owns it via type/objectName selectors (`ClassThresholdTable`,
  `PhraseEditorPanel …`, `#dino_phrase_hint`) so compact ≠ unscaled.
  Follow that pattern for new compact widgets.
- ⚠️ Known debt: `dino_merge_dialog.py` still carries hardcoded
  `font-size:Npx` tokens and a `color:#444` dark-mode contrast issue,
  so it doesn't scale. Tracked, not an oversight; fix when that
  dialog is next touched.

---

## ADR-021: SAM Fine-Tuning via a Custom Loop over the Ultralytics SAM2 Module

**Status**: Accepted

**Context**: Users annotating domain-specific imagery (microscopy,
medical, materials) get generic SAM masks that need heavy correction.
We want to let them fine-tune SAM 2 / 2.1 on their own annotations and
reuse the result in the existing SAM-box / SAM-points workflow
(upstream issue bnsreenu#73).

The obvious approach — mirror the YOLO trainer's `model.train(...)` —
**does not work**: Ultralytics registers only a *predictor* for SAM's
`segment` task (`SAM.task_map`), so `SAM(...).train()` raises
`NotImplementedError` (verified on ultralytics 8.4.51).

**Decision**: Fine-tune with a custom PyTorch loop that **reuses
Ultralytics' own forward path**. `SAM(...).model` is a plain
`SAM2Model` `nn.Module`; its `SAM2Predictor` exposes the forward in
reusable pieces — `get_im_features` (image encoder) and
`prompt_inference` / `_inference_features` (prompt encoder + mask
decoder). These are *not* wrapped in `inference_mode` unless reached
via the public `__call__`, so calling them directly under
`torch.enable_grad()` yields differentiable mask logits. The engine
(`training/sam_trainer.py`) adds focal+dice loss (≈20:1) + AdamW +
backward. Default freeze policy: train only `sam_mask_decoder`
(image + prompt encoders frozen); an optional flag also unfreezes the
image encoder.

Checkpoints are saved as `{"model": state_dict}` — the exact shape
Ultralytics' `_load_checkpoint` reads (it rebuilds the architecture
from the filename suffix and `load_state_dict`s the nested `model`
key). Consequently a fine-tuned file **must keep its base token in the
name** (e.g. `myrun_sam2_t.pt`), enforced by `make_custom_filename`;
`build_sam` selects the architecture by `ckpt.endswith(token)`. Every
save is round-trip-verified by reloading through `SAM(out_path)` and
running one forward — failing loudly rather than producing a file that
won't reload (cf. facebookresearch/sam2#337 key-mismatch failures).

**Alternatives considered**:
- *facebookresearch/sam2 training code* — rejected: heavy extra
  dependency overlapping Ultralytics' bundled SAM2, and its checkpoints
  need state-dict conversion to reload into our `SAM()` inference path.
- *Export dataset + train externally* — rejected as the default (less
  "integrated"), though `Prepare SAM Dataset` + folder training give a
  similar offline path for users who want it.

**Consequences**:
- ✅ No new runtime dependency; fine-tuned models drop straight into
  the existing SAM selector and inference path.
- ✅ Exposure to Ultralytics internals is confined to a few
  already-exercised predictor methods, guarded by
  `test_sam_finetuning.py::TestUltralyticsAPI` (fails on an upgrade
  that renames them).
- ⚠️ The trainer loads its **own** `SAM` instance on its `QThread`
  (it does not touch `SAMUtils._model`), and must **not** use
  `sam_utils._run_sync` (its re-entry guard is GUI-thread-local). The
  real hazard is two SAM models (resident inference + training) on one
  CUDA context, so `SAMTrainController` locks the SAM inference UI
  (tools + model selector + the fine-tune menu) for the duration —
  re-enabled in `training_finished` on both the success and error
  paths.
- ⚠️ Decoder fine-tuning is realistically GPU-only; a CPU-only box is
  hard-warned before a run (`resolve_torch_device`), and the device is
  pinned so an incompatible GPU is honoured as CPU instead of crashing.
- ⚠️ Encoder features are recomputed per epoch (bounded memory) rather
  than cached across epochs; revisit if large datasets need the speedup.
- ⚠️ **Loss must use the inference coordinate frame.** SAM2 letterboxes
  the image (`LetterBox(1024, center=False)`, pad bottom/right) and
  inference maps masks back with `ops.scale_masks(..., padding=False)`,
  which crops that padding before upsampling. The training loss therefore
  runs the decoder logits through the *same* `ops.scale_masks` before
  comparing to the GT mask — a naive `F.interpolate` over the full
  low-res mask bakes the padding into the target and the decoder learns
  masks shifted by the pad (a downward shift on non-square images, caught
  only during GUI testing because the e2e tests used square images). The
  landscape regression test (`test_landscape_no_mask_shift`) and the
  `ops.scale_masks` API-drift guard protect this.
- ℹ️ The custom loop later gained a train/val split, a no-grad validation pass
  (`val_loss`), a warmup→cosine LR schedule, and early stopping with best-checkpoint
  selection — see **ADR-028**.

---

## ADR-022: Canvas Mask Selection Unified with the Annotation List

**Status**: Accepted (issue bnsreenu#75)

**Context**: Selecting an existing annotation was only possible through the
bottom-left annotation list (already `ExtendedSelection`) or by *double*-clicking
a mask on the canvas — which immediately enters vertex-edit mode. There was no
single-click select, no box/multi-select on the image, and canvas `Delete` worked
only while in vertex-edit mode. Issue #75 asked for single-click select (without
entering edit), rubber-band box select, modifier multi-select, and multi-delete —
all directly on the canvas.

**Decision**: Add an **idle-mode selection layer** to `ImageLabel` and route it
through the *existing* annotation-list selection so delete/merge/change-class are
reused unchanged:

- **Idle activation.** Selection is live only in `_is_select_mode()` — no drawing
  tool, not editing, not SAM, no temp review. Picking any tool restores drawing.
  No new tool button (matches the user's "a single click should select" ask).
- **Gestures.** Plain click selects the smallest mask under the cursor (covers
  segmentation *and* bbox); click on empty space clears; drag draws a rubber band
  and selects every annotation whose bounds intersect it; **Shift** makes a click
  toggle and a drag additive. Double-click is unchanged (still vertex edit).
- **Ctrl stays pan.** Ctrl+drag pan (with its carefully tuned reference frame) is
  left untouched; multi-select uses Shift instead of Ctrl.
- **One selection, two surfaces.** The canvas emits
  `canvasSelectionChanged(annotations, mode)`; `AnnotationController.apply_canvas_selection`
  computes the new set (replace/add/toggle), sets `image_label.highlighted_annotations`,
  and **mirrors it onto the list** with signals blocked. `Delete` on the canvas
  reuses `delete_selected_annotations` (which reads the list selection).

**Consequences**:
- Delete / Merge / Change-Class need no new logic — they already operate on the
  list selection, which the canvas now drives.
- ⚠️ Matching between the canvas and list relies on **dict value-equality**, like
  the rest of the selection code (`image_label.annotations` is a deepcopy of
  `all_annotations`, and PyQt round-trips `UserRole` dicts as copies, so identity
  is never stable). Value-equal duplicate masks would select together — a
  pre-existing, accepted limitation. See the crosscutting "Canvas selection ↔
  list selection" section.
- ⚠️ The list mirror must block `itemSelectionChanged` while selecting, or it
  recurses back through `update_highlighted_annotations` and overwrites the set.

**Selection is rendered class-colour-independent (amendment).** The first cut
drew the selected mask in solid **red** — invisible on a red-class mask, and the
default palette assigned red as the *first* class colour. Selection is now an
overlay drawn in a final pass on top of every mask, independent of class colour
and modelled on the sibling open-garden-planner app's CAD selection: a dashed
selection-blue **bounding-box marquee** (`_SELECTION_COLOR = QColor(0, 120, 215,
220)`) plus bright opaque-blue **handle squares** at the 4 corners + 4 edge
midpoints, white-cased and fixed on-screen size (`_draw_selection_overlay` in
`widgets/image_label.py`). The handles are what make selection unmistakable
regardless of mask colour (a single thin dashed outline was too faint; an earlier
marching-ants + marquee was too busy). The handles are now grab targets for
resize/move of any selected shape (see ADR-023). The mask keeps its
class colour; the rubber-band rect uses the same blue dashed style. Separately,
the default class palette
(`core/constants.py::DEFAULT_CLASS_COLORS` / `default_class_color`) was reordered
so red is **last** (no fresh project starts on red) and muted, and the default
fill opacity dropped to `0.2` (`DEFAULT_FILL_OPACITY`) so masks don't bury the
image. Existing projects keep their persisted class colours.

---

## ADR-023: Direct-Manipulation Shape Editing on the Selection Handles

**Status**: Accepted (issue bnsreenu#40)

**Context**: `"bbox"`-keyed annotations (from COCO/YOLO import and detectors)
were **not editable at all** — `start_polygon_edit` only matches `"segmentation"`,
so double-click vertex edit skipped them. ADR-022 draws 8 handle squares around
*any* selected annotation, but they were visual-only. A first cut wired them up
for `"bbox"`-typed annotations only — but almost everything in this app is stored
as `"segmentation"` (drawn rectangles, polygons, SAM/DINO masks all are), so the
handles looked grabbable on every shape yet did nothing on the shapes users
actually have. The handles must act on **any** selected shape.

**Decision**: Wire the handles up as **direct-manipulation** resize/move of the
single selected shape, modelled on the sibling open-garden-planner app's
`ResizeHandle`. No new mode, no double-click — it works off the existing idle-mode
selection:

- **Single-shape, any kind.** Handles are draggable when exactly one annotation
  with a bounding box is selected (`_single_selected_shape()`); a multi-select
  leaves them visual. The press handler resolves to the live object
  (`_live_annotation`) and records `kind` — `"seg"` (polygon/mask) or `"bbox"`
  (box-only import) — which picks the geometry the handles drive
  (`_begin_shape_edit`).
- **Anchor-from-handle.** A corner/edge drag computes the new bounding box
  (`_resize_bbox`: replaces the dragged coordinate, opposite side fixed,
  normalised, ≥ 1px). A `"bbox"` shape sets `[x, y, w, h]` directly; a polygon
  **scales every vertex** from the old box to the new one
  (`_scale_segmentation`), so the outline resizes proportionally. Per-handle
  resize cursors match OGP (`_BBOX_HANDLE_CURSORS`; `SizeAll` over the interior).
- **Move is drag-gated.** A press inside the shape starts a *pending* move that
  promotes only once the drag clears the `3px/zoom` threshold — so a plain click
  still falls through to selection (preserving nested-mask click-through). Move
  translates the box (`[x,y,w,h]`) or all vertices (`_translate_segmentation`).
  The geometry mutates **in place** so the canvas + overlay redraw live.
- **Bbox key stays in sync.** Imported annotations carry both `segmentation` and
  `bbox`; editing the polygon recomputes the `bbox` key (`_sync_bbox_key`) so
  export/training stay consistent. Drawn shapes have no bbox key and gain none.
- **Commit / cancel.** Release clamps into the image (ADR-024 — move slides the
  intact shape back inside, resize trims/clamps) and emits `bboxEditCommitted` →
  `AnnotationController.commit_bbox_edit` (save + list rebuild + re-mirror the
  selection). Escape restores the original geometry.

**Consequences**:
- The handles you see are exactly the grab targets — `_draw_selection_overlay`
  and `_bbox_handle_at` share `_bbox_handle_points`, so visual and hit geometry
  can't drift — and now they work on every selected shape, not just imported boxes.
- Resizing a polygon **scales** it (handles drive the bounding box); reshaping a
  polygon vertex-by-vertex is still double-click vertex edit. A `"bbox"` shape
  stays rectangular by construction.
- ⚠️ The shape-drag branches sit **before** the rubber-band branch in the
  idle-mode mouse dispatch; both are gated on `_is_select_mode()` so a
  tool/edit/SAM state still wins. (Internal names keep the `bbox_edit` /
  `bboxEditCommitted` prefix — they denote editing via the bounding-box handles,
  whatever the underlying geometry.)

---

## ADR-024: Bounds Enforcement — Clamp Manual Edits, Clip Augmented Data

**Status**: Accepted (issues bnsreenu#32, bnsreenu#36)

**Context**: Annotation coordinates could be persisted outside the image
rectangle and silently poison training data. *Drawn* shapes were already safe
(`finish_polygon`/`finish_rectangle` shapely-intersect with the image boundary),
but two paths weren't: **manual edits** (polygon vertex drag; the new bbox drag)
clamped nothing, and the **Image Augmenter** wrote rotated/zoomed/flipped polygons
verbatim.

**Decision**: Add three pure helpers in `utils.py` and apply the right one per
path:

- **Clamp manual edits** with `clamp_segmentation` / `clamp_bbox` — per-coordinate
  snap into `[0, w] × [0, h]`. Per-coordinate (not a shapely cut) is deliberate:
  it **preserves the vertex count and ordering**, so a polygon being dragged never
  loses or splits points mid-edit. Applied in place at edit commit (polygon Enter;
  bbox release), persisting through the existing save-by-reference path.
- **Clip augmented data** with `clip_polygon_to_bounds` — a shapely intersection
  (largest resulting polygon; `buffer(0)` first to repair self-intersections an
  affine augmentation can introduce). Geometric trimming is correct here because an
  augmented shape genuinely extends past the frame and should be cut at the edge,
  not have stray vertices snapped onto it. A polygon left fully outside returns
  `None` and is **dropped** by the augmenter loop.

**Consequences**:
- One vocabulary, two semantics: *clamp* (cheap, count-preserving, for live edits)
  vs *clip* (exact, may drop/split, for batch augmentation). The choice is about
  whether vertex correspondence must survive, not about which is "more correct".
- ⚠️ `clip_polygon_to_bounds` can return fewer/more vertices than the input and may
  return `None`; callers must handle the drop (the augmenter `continue`s).
- The existing `finish_polygon`/`finish_rectangle` inline clips were left as-is to
  keep the diff contained; they could later delegate to `clip_polygon_to_bounds`.

---

## ADR-025: Reversible Per-Annotation Polygon Simplification (Detail %)

**Status**: Accepted (issue bnsreenu#24)

**Context**: SAM/DINO masks are stored as raw dense polygons — `_mask_to_polygon`
returns the flattened `cv2.findContours` boundary with no simplification, so a
single mask can carry hundreds of vertices, bloating label files. Issue #24 asked
for a "mask complexity — less ↔ more points" control. The point add/remove half of
#24 was already covered by the SAM-points tool; this is the remaining piece.

**Decision**: A **per-annotation, reversible Detail %** control, surfaced as a
column in the Annotations panel:

- **Detail % (1–100, 100 = raw).** `utils.simplify_polygon(raw, pct)` thins via
  Douglas-Peucker (`cv2.approxPolyDP`), binary-searching the epsilon for the
  richest polygon whose vertex count is still ≤ `round(raw_count × pct/100)`.
- **Reversible via a preserved raw.** The dense original is **lazy-captured** into
  `segmentation_raw` the first time a mask is thinned (nothing simplifies it
  before that, so the live `segmentation` *is* the raw at capture). 100 % copies
  `segmentation_raw` back into `segmentation` exactly. No edits to the SAM/DINO/
  manual accept paths were needed.
- **Two new annotation keys** (`segmentation_raw`, `detail_pct`) ride along: they
  round-trip through `.iap` for free (project save does `ann.copy()` →
  `convert_to_serializable` → JSON), and exports read only the effective
  `segmentation`, so the *simplified* polygon is what's exported. Imported/old
  annotations have neither key → handled by lazy-init.
- **Live + in place.** The change handler resolves the selected row to the live
  drawn object by value-equality (`image_label._live_annotation`, reused from
  #40), mutates `segmentation` in place, refreshes the Area cell + the row's
  UserRole, redraws, and saves. The `bbox` key (if present) is recomputed.

**The Annotations panel became a `QTableWidget`** (ID | Class | Area | Detail %),
mirroring `dialogs/dino_phrase_editor.ClassThresholdTable` (per-row spinbox via
`setCellWidget`, `SelectRows`, `NoEditTriggers`, stylesheet-only header). This
re-homes the #75 canvas↔list selection bridge onto a table:

- The annotation dict lives in **column 0's UserRole** (the value-equality marker).
- `count()/item(i)/selectedItems()` → `rowCount()/item(r, 0)/row-deduped
  selectedIndexes()`; the mirror uses **`setRangeSelected` (additive)** because
  `selectRow()` *replaces* the selection in ExtendedSelection mode and would drop
  all but the last row. `blockSignals` + value-equality are preserved verbatim.

**Consequences**:
- Closing #24 with a small, contained change: the feature is the table UI + one
  controller handler + one pure util; the accept paths are untouched.
- ✅ Fully reversible per annotation: Detail %=100 restores `segmentation_raw`
  exactly. **Exception:** reshaping a polygon with the #40 handles invalidates the
  baseline — `_clamp_edited_shape` drops `segmentation_raw` and resets
  `detail_pct=100`, so the *edited* geometry becomes the new raw (the old dense
  outline no longer describes the reshaped polygon, and a later 100 % must not
  silently revert the edit). The detail handler also re-points
  `highlighted_annotations` at the mutated object so the overlay + a subsequent
  handle drag stay coherent.
- ⚠️ The spinbox `valueChanged` is connected **after** the initial `setValue`, so
  building/rebuilding the table never fires the simplification handler.
- ⚠️ The dead `core/annotation_utils.py` still references the old QListWidget API
  but is unimported (confirmed) — left as-is to keep the diff contained.

---

## ADR-026: Snapshot-Based Undo/Redo for Annotation Edits

**Status**: Accepted

**Context**: Annotation edits (create, delete, merge, move/scale, change class,
detail %, paint, eraser, SAM/DINO accept) were all irreversible. The only safety
net was a confirmation dialog on delete and a keep/delete prompt on merge — both
of which broke flow (delete also popped a success dialog). The justification for
those dialogs was "you can't undo," so removing them required a real undo/redo.

The mutation surface is wide and subtle: every operation writes both
`image_label.annotations` (the live working copy) and `all_annotations[key]`, and
the two share inner list objects via the shallow-copy save. Annotations are
matched by **value-equality**, not identity (ADR-022/025), numbers are reassigned
on most edits (`renumber_annotations`), and Detail % carries a lazily-captured
`segmentation_raw` (ADR-025). A fine-grained command-per-operation design would
have to reproduce every one of these invariants in its undo path.

**Decision**: **Snapshot the whole per-image annotation dict** before each edit;
undo restores a snapshot wholesale. Restoring the entire dict sidesteps all the
value-equality / renumbering / selection-rehoming / `segmentation_raw`
subtleties — there is nothing to reconcile, only a deep copy to install.

- `controllers/annotation_history.AnnotationHistory` holds **per-image-key**
  undo/redo stacks (key = `current_slice or image_file_name`), so Ctrl+Z acts on
  the image on screen and never reaches an image you can't see. Stacks are
  retained across navigation and cleared on clear-all / new-project / project
  open. Depth is capped (50) and the symmetric model needs no separate baseline:
  `record(before)` pushes the pre-edit state, `undo(current)` swaps current onto
  redo and returns the popped before-state, `redo(current)` is the mirror.
- **One choke-point**, `AnnotationController.record_history()`, called *before*
  each synchronous mutation (finish polygon/rectangle, delete, merge, change
  class, eraser replace, SAM accept, DINO accept). It is **not** hooked onto
  `save_current_annotations()` — that also fires on navigation and runs *after*
  mutation, so it can neither be filtered to real edits nor capture a clean
  "before."
- **Deferred gestures** (bbox move/scale, paint stroke, polygon vertex edit)
  notify the controller only *after* mutating in place. They capture the baseline
  at gesture **start** via a new `ImageLabel.editBaselineRequested` signal →
  `capture_edit_baseline`, and push it at commit (`commit_edit_baseline`, called
  from `commit_bbox_edit`, `commit_polygon_edit`, and the `annotationsBatchSaved`
  handler). A **deep-equality dedup** in `record()` drops aborted gestures (Esc'd
  drag, empty stroke) so they leave no entry.
  - *Vertex edit also got a save-discipline fix.* Its Enter-commit historically
    only refreshed the list and relied on a later save to persist (and **Esc did
    not revert** the in-place drags). `commit_polygon_edit` now calls
    `save_current_annotations`, and Esc restores the segmentation from a snapshot
    taken at edit-mode entry — so the commit is both persisted and undoable, and
    Esc truly cancels.
- **Detail-% coalescing.** The spinbox fires `valueChanged` per step; a whole
  drag on one annotation records once (token = key + number + class), so one
  Ctrl+Z reverts the entire drag including `detail_pct` and `segmentation_raw`.
- **Shortcuts** are `QShortcut`s with `ApplicationShortcut` context (Ctrl+Z; Ctrl+Y
  and Ctrl+Shift+Z for redo) — the annotation-list `QTableWidget` would otherwise
  consume Ctrl+Z. Undo/redo are no-ops during project load, while a modal is open,
  while a text field has focus, or while a draw/edit gesture is in flight
  (`_undo_blocked`). Undo persists via `auto_save` — the net must survive reopen.

**Delete and merge dialogs removed.** With undo as the net, `delete_selected_annotations`
drops both the confirmation and the success dialog; `merge_annotations` drops the
keep/delete prompt (originals are always replaced by the union) and the success
dialog. Validation warnings stay.

**Consequences**:
- ✅ Every annotation edit is reversible; destructive ops are instant and flow-friendly.
- ✅ Robust against the value-equality/renumber/raw subtleties because it restores
  whole dicts rather than replaying operations.
- ⚠️ Memory is a bounded deep copy per edit per image (annotations are small;
  depth-capped at 50). ⚠️ Undo clears the current selection rather than trying to
  re-resolve it by value across a list rebuild — the safe, predictable choice.

---

## ADR-027: Mandatory MLflow Experiment Tracking (SAM-explicit / YOLO-native)

**Status**: Accepted (issue bnsreenu#74); amended — tracking is now a core,
always-on feature rather than an opt-in extra (see "Amendment" below).

**Context**: Training output (SAM fine-tuning and YOLO training) vanished after a
session — no record linked a saved checkpoint to its hyperparameters, no run-to-run
comparison, no persisted loss curves. The only feedback was live strings in
`TrainingInfoDialog`. Issue #74 asked for MLflow tracking. This fork's owner
decided every training run must be tracked — the app already hard-requires torch /
ultralytics / transformers, so MLflow's footprint is negligible and an opt-out only
invites "why is there no run?" confusion.

**Decision**:

- **Core dependency.** MLflow is in the `pyproject.toml` runtime `dependencies`
  list, not an optional extra — a fresh `pip install` always has it. `import
  mlflow` still happens only inside the methods of `training/mlflow_tracker.py`
  (never at module top), so app startup stays fast and a *broken* install can't stop
  the GUI from launching — but tracking is never *expected* to be absent.
- **One small wrapper, `MLflowTracker`, always on.** There is no "disabled" mode and
  no user toggle. Every live mlflow call is wrapped so a tracking error logs a status
  line but **never aborts training** — pure crash-safety, not an opt-out. A separate
  `_NullTracker` no-op stands in only when a trainer is called *without* a tracker
  (direct/programmatic calls, tests); the GUI always supplies a real tracker.
- **Two integration styles, by trainer shape:**
  - **SAM** has a *custom* training loop, so it logs **explicitly** through a
    tracker passed into `SAMFineTuner.train(..., tracker=...)`. The run is
    started/logged/ended **inside `train()` on the worker thread** because MLflow
    runs are thread-bound. The controller builds the (unstarted) tracker; the
    trainer wires the tracker's status `log` to its own thread-safe
    `progress_signal` (never a direct cross-thread QTextEdit write).
  - **YOLO** uses Ultralytics' *built-in* MLflow callback — armed every run by
    setting `MLFLOW_TRACKING_URI` / `MLFLOW_EXPERIMENT_NAME` and
    `ultralytics.settings.update({"mlflow": True})`. It logs richer metrics
    (box/cls/dfl loss, mAP, the model) for free.
- **Local file store by default.** `resolve_tracking_uri()` precedence: a non-empty
  QSettings override → `<project>/mlruns` when a project is open → `<cwd>/mlruns`.
  Two cross-version/cross-platform hazards are handled at the mlflow boundary
  (`to_mlflow_uri()` + an env flag), not in the resolver (which keeps returning a
  plain path for display and directory use):
  - **Windows path → `file://` URI.** mlflow validates the URI *scheme*, so a bare
    `C:\…\mlruns` is read as scheme `c` and rejected — local tracking would silently
    degrade to untracked. `to_mlflow_uri()` converts local paths to `file://` URIs
    (genuine `http`/`sqlite`/`databricks` URIs pass through) at every mlflow call
    site: `MLflowTracker.start()`, the YOLO `MLFLOW_TRACKING_URI` env var, and the
    `mlflow ui` launch.
  - **mlflow 3.x file-store opt-out.** mlflow ≥3 raises on the local file store
    unless `MLFLOW_ALLOW_FILE_STORE=true`; we `setdefault` it before touching mlflow
    so the documented file-store default keeps working on both 2.x and 3.x without
    overriding an explicit user setting.
- **Config surface — destination only, no on/off.** A dedicated **Settings →
  Experiment Tracking** dialog (`MLflowSettingsDialog`) edits the tracking-store
  URI and experiment name (the only knobs), and an **Open MLflow UI** action shells
  out to `<python> -m mlflow ui`. The training dialogs have **no** "track this run"
  checkbox — every run is tracked.
- **Live run link + auto-open.** When a SAM run opens, `MLflowTracker.start()`
  captures `run_id`/`experiment_id` and fires a `set_run_url_callback` with the UI
  deep link (`run_ui_url()` → `http://localhost:5000/#/experiments/<id>/runs/<id>`).
  The trainer relays it via its `mlflow_run_url` signal (worker → GUI thread); the
  controller posts a **clickable link** into the Fine-Tuning Progress dialog (now a
  `QTextBrowser` with `setOpenExternalLinks`), starts the `mlflow ui` server **once**
  (`start_mlflow_ui_server`, split out of `launch_mlflow_ui`), and **opens the run
  in the browser** (deferred ~2.5 s via `QTimer` on first launch so the cold server
  is ready). All of this is best-effort and never disturbs the run.

**Amendment (always-on)**: The feature originally shipped as an opt-in extra with a
per-dialog checkbox and an `enabled` QSettings flag (matching issue #74's "optional"
framing). That was reversed: MLflow moved to `install_requires`, the checkboxes and
the `enabled` pref were removed, `MLflowTracker` lost its `enabled` parameter, and
`load/save_mlflow_prefs` now carry only `(uri, experiment)`. The lazy import and the
never-abort-training error handling remain — as robustness, not as an opt-out.

**Consequences**:
- Every training run produces an MLflow run with no user action; there is no path
  that trains untracked except a genuinely broken MLflow install (which degrades
  safely rather than crashing the run).
- ⚠️ Two logging styles (explicit for SAM, native for YOLO) means runs from the two
  trainers are organized by their respective conventions; both honor the same
  tracking URI / experiment name, but their param/metric keys differ.
- ⚠️ MLflow run thread-affinity is load-bearing for SAM: the run **must** open and
  close on the worker thread that trains, not the GUI thread that builds the tracker.

---

## ADR-028: Train/Val Split, Val-Loss, Warmup→Cosine LR & Early Stopping for Training

**Status**: Accepted (issue bnsreenu#85)

**Context**: The two trainers were asymmetric and neither reported *generalization*.
SAM fine-tuning (ADR-021) logged only a per-epoch `train_loss` over all annotated
instances — no held-out set, so the curve always trended down and couldn't reveal
overfitting (the main reason to track experiments). Its LR was fixed for the whole
run and only the last epoch's weights were saved. YOLO already got val metrics + mAP
from Ultralytics natively, but the app surfaced only a single `trainer.loss` line and
exposed none of Ultralytics' LR-schedule / early-stop knobs.

**Decision**: Give both trainers a configurable train/val split, both-loss tracking,
a linear-warmup → cosine-to-floor LR schedule, and patience-based early stopping with
best-checkpoint selection. The schedule shape is a **fixed smart default** (warmup =
first 10% of steps, cosine floor = 10% of peak); only the *peak* LR, train %, and
patience are user-editable (the literature says the peak LR matters more than the
shape).

- **Deterministic split.** A new `sam_dataset.split_groups(groups, train_pct)`
  reuses the YOLO export's stable-MD5 `assign_train_val` (ADR for #83) so SAM and
  YOLO split identically and reproducibly. `SampleGroup` gained a `name` used only as
  the split key. *(Superseded in part by ADR-044: the key is now the source group
  derived from that name, not the name itself — otherwise the val loss that drives
  early stopping below is measured on near-copies of trained frames.)* At 100% train (or a single image) the val set is
  empty and the val pass / early stopping are skipped (the UI says so; the SAM dialog
  also disables OK at 0% train). **YOLO's split stays at "Prepare Dataset" time** —
  it's baked into `images/train` vs `images/val` folders at export, so the Train
  dialog only adds schedule/early-stop knobs, never a re-export.
- **SAM val pass + both losses.** The per-image loss body was extracted into
  `_image_instance_losses(..., train=bool)` so the same forward serves training
  (`enable_grad`, backprops once per image) and a no-grad validation pass
  (`_validation_loss`, run under `net.eval()` then encoder train-mode restored). Each
  epoch logs `train_loss`, `val_loss`, and the current `lr` to MLflow and the progress
  window. YOLO surfaces its native val loss + mAP via a new `on_fit_epoch_end`
  callback (fires *after* validation, so `trainer.metrics` is populated; keys read
  defensively) — MLflow already gets them natively.
- **LR schedule.** SAM uses `torch.optim.lr_scheduler.LambdaLR` driven by a pure
  `lr_schedule.warmup_cosine_lambda(total_steps)`, stepped once per optimizer step
  (`~ceil(train_images / batch_size)` per epoch; the lambda clamps if the real count
  drifts). A checkbox reverts to constant LR. When on, YOLO forwards `cos_lr=True`,
  `lr0`, `lrf=0.1`, `warmup_epochs=round(0.1·epochs)` to Ultralytics' `train()`; when
  off it forwards `cos_lr=False`, `lrf=1.0`, `warmup_epochs=0` so both toggles' "off"
  state means a genuinely constant LR (not a linear-decay-with-warmup).
- **Early stopping + best checkpoint.** SAM uses a tiny pure `EarlyStopper(patience)`
  (default 20; `0` disables). On each val improvement the trainer snapshots the
  weights (CPU clone); `_save_and_verify(..., state=best_state)` saves that snapshot
  rather than the last epoch (falls back to the live net when there's no val /
  improvement, preserving the original behaviour). YOLO forwards `patience` and keeps
  Ultralytics' `best.pt`.

**Why pure helpers** (`lr_schedule.py`, `early_stop.py`, `split_groups`): the tricky
math/bookkeeping is unit-tested without torch, Qt, or a real run
(`test_lr_schedule.py`, `test_early_stop.py`, `test_sam_split.py`); the val/best/early
-stop wiring is covered by driving `_run_epochs` with light stubs
(`test_sam_finetuning.py::TestValPassAndBestCheckpoint`), and the YOLO passthrough by
`test_yolo_training_args.py`.

**Alternatives considered**:
- *A shared trainer base class for both paths* — rejected: the two loops are too
  different (custom PyTorch vs Ultralytics-native). Only the pure pieces +
  `assign_train_val` are shared; a unification refactor isn't warranted.
- *Exposing every schedule knob* (warmup fraction, floor, lr0 for SAM) — deferred:
  the simplified surface (peak LR + train % + patience + a schedule toggle) keeps the
  dialogs legible; the fixed 10%/10% recipe is the modern default.
- *Stepping the SAM schedule per epoch* — rejected in favour of per-optimizer-step so
  short runs still get a real warmup ramp.

**Consequences**:
- ✅ Both trainers now show generalization (val_loss / mAP) and a tracked LR curve;
  SAM saves its best-val checkpoint instead of the last epoch.
- ⚠️ SAM's saved checkpoint changing from "last" to "best-val" is a behaviour change
  (only when a val set exists); 100% train preserves the old last-epoch save.
- ⚠️ The SAM val pass recomputes encoder features for the held-out images each epoch
  (same bounded-memory tradeoff as training, ADR-021); a larger split costs more time.
- ⚠️ `on_fit_epoch_end` reads Ultralytics metric keys (`val/box_loss`,
  `metrics/mAP50(B)`, …) which vary by task/version, so every read is guarded and a
  miss just omits that field rather than breaking the run.

---

## ADR-029: Keypoint / Pose Annotation — Per-Class Schema, COCO Instance Model, 3-State Visibility

**Status**: Accepted (issue bnsreenu#35, PR-1 + PR-2 + PR-3 — complete)

**Context**: The app annotated polygons, rectangles, and paint masks (+ SAM/DINO) but
had no way to place **keypoints** — the primitive for pose estimation and landmark
detection. "Keypoint annotation" can mean standalone points or full COCO/YOLO-pose
instances; the maximal target was chosen: per-class ordered named keypoints + a
skeleton, one annotation = one K-point instance tied to a bounding box, COCO 3-state
visibility, and (later PRs) COCO/YOLO-pose export-import + YOLO-pose training. This PR
(PR-1) covers annotate + persist + render only.

**Decision**:

- **One instance = one annotation, flat `[x,y,v]*K` + stored `bbox`.** A pose instance
  is stored like any other annotation in `all_annotations[image][class][]`:
  `{"keypoints": [x1,y1,v1, …], "num_keypoints": <v>0 count>, "bbox": [x,y,w,h],
  "category_id", "category_name", "number"}`. The flat triple list mirrors COCO
  exactly, so it round-trips through `.iap` via `image_utils.convert_to_serializable`
  with **zero** save/load code. **Absence of a `segmentation` key is the load-bearing
  discriminator**: `calculate_area` falls to the bbox branch, the Detail-% spin
  auto-disables, and `draw_annotations` routes to the keypoint branch (added *before*
  the `bbox` branch, since an instance also carries a bbox).
- **`v` is the COCO 3-state enum** (0 = not labelled, 1 = labelled+occluded,
  2 = labelled+visible) — identical to YOLO-pose, so no remap on export. The bbox is
  **stored, not derived**, so `_annotation_bbox`, `calculate_area`, click-selection,
  and the #40 resize handles all work unchanged; `_keypoint_bounds` is a fallback only
  for imports that omit a box. `num_keypoints` is recomputed on every edit.
- **Per-class schema, not per-instance.** COCO requires all instances of a category to
  share one keypoint set, so the schema lives in `main_window.keypoint_schemas`
  (`{class_name: {"names", "skeleton", "flip_idx"}}`), **not** on the annotation. A
  class is a "pose class" iff it has a schema. The schema is embedded on each `classes[]`
  entry in `.iap` (mirroring `dino_config`'s validate-on-load robustness — malformed
  schemas are dropped with a print, never crash; old projects load unchanged with an
  empty store). `flip_idx` (h-flip partner per point) is app-only for COCO but required
  by YOLO-pose; pure validation lives in `core/keypoint_schema.py` so it's unit-testable.
- **Guided in-order placement tool.** `KeypointTool(ToolHandler)` places points in
  schema order: left-click = visible (v2), right-click / Shift+left = occluded (v1),
  auto-finish at K, Enter finishes early (remaining points padded v0), Backspace goes
  back, Esc discards. Because the manual-tool dispatch is left-button-only, the tool
  short-circuits `mousePressEvent` to accept both buttons (mirroring `sam_points`). The
  in-progress overlay renders for free (paintEvent iterates all handlers).
- **Editing reuses the #40 selection-handle path, not double-click** (double-click is
  segmentation-specific). A new edit `kind="kpt"` makes the instance **box transform the
  whole pose** (scale/translate points + box together, like a polygon's box) via the
  existing `bbox_edit` machinery — it commits via `bboxEditCommitted` → `commit_bbox_edit`,
  same as a bbox/segmentation resize. A **separate** single-point drag (`editing_keypoint`)
  moves one keypoint, and a right-click on a committed point toggles its visibility
  (2↔1) — both of *these* push an undo baseline at gesture start and commit via
  `keypointEditCommitted` → `commit_keypoint_edit` (ADR-026). Not-labelled (v=0) points
  are skipped by `_scale_keypoints`/`_translate_keypoints` and stay at `(0,0)`, since
  COCO/YOLO-pose require that invariant.
- **Guards.** Keypoint instances are rejected from **merge** (no mergeable geometry —
  they'd be silently deleted) and from cross-schema **change-class** (a normal
  annotation can't become a pose instance and vice versa; a keypoint instance only
  moves to a pose class with an identical `names` list). The schema-definition dialog
  locks the keypoint count K once instances exist (only K can corrupt them).
  **A class is pose OR regular, not both, enforced at the UI (#44):** defining a schema
  on a class that already holds plain annotations is blocked
  (`ClassController.define_keypoint_schema`, only for a *new* conversion — editing an
  existing/legacy-mixed schema stays allowed so names/skeleton can still be fixed);
  activating a shape tool (`toggle_tool`) or a SAM tool (`SAMController.toggle_sam_*`)
  while a pose class is selected is refused (button unchecked); selecting a pose class
  while a shape/SAM tool is active deactivates it (`on_class_selected`); and DINO
  detection skips pose classes at the one config builder both paths share
  (`_build_dino_class_configs`). Legacy-mixed classes from older projects still load,
  render, and save — the guards only stop *new* mixing; `_pose_export_check` remains the
  backstop for imported/legacy data.
- **Area = bbox area (not 0).** `calculate_area` returns the stored box's `w*h` for a
  keypoint instance — deliberate, so sort-by-area behaves consistently with imported
  bbox annotations rather than dumping all poses to the end.
- **Export/import (PR-2).** COCO categories gain `keypoints`/`skeleton` (**1-based**,
  per spec) plus an app-only `flip_idx` extension key (kept **0-based** — no COCO
  precedent, and it's consumed only by our own importer / the PR-3 trainer, both
  0-based; converting it would just add a pointless round-trip). `create_coco_annotation`
  and `export_yolo_v5plus`'s per-annotation writer both check `"keypoints" in ann`
  *before* segmentation/bbox, mirroring the rendering dispatch order — a pose instance
  also carries a `bbox`, so checking bbox first would make the keypoints branch
  unreachable. YOLO-pose (`data.yaml`: `kpt_shape: [K, 3]`, `flip_idx`) is
  **dataset-global**, so `export_yolo_v5plus` refuses (`ValueError` → `QMessageBox`, via
  `_pose_export_check`) a mix of >1 distinct `(K, flip_idx)` schema or a mix of pose and
  non-pose classes among the annotations actually being written; detection is
  data-driven (based on which annotations carry `keypoints`), not solely on
  `keypoint_schemas`, so a caller that doesn't thread schemas through still gets a
  correct K (PR-3's `prepare_dataset` does thread `keypoint_schemas` through, for the
  richer `flip_idx`/skeleton data — see the PR-3 addendum below). All four
  `io.import_formats`
  entry points (`import_coco_json`, `import_yolo_v4`, `import_yolo_v5plus`, and
  `process_import_format`'s pass-through) now uniformly return
  `(annotations, image_info, keypoint_schemas)` — `{}` where nothing was recovered — so
  `io_controller.py` has one contract regardless of format. YOLO-pose import applies its
  one recovered schema (generic `kp0..kp{K-1}` names, no skeleton) to **every** class
  declared in `data.yaml`'s `names`, not just classes observed with pose-shaped lines,
  since `kpt_shape`/`flip_idx` are dataset-global. **The rebuild step in
  `io_controller.import_annotations` (`_rebuild_imported_annotation`) builds a fully
  separate dict shape for keypoint vs. non-keypoint annotations** — it must never attach
  a `None`-valued `segmentation`/`type` key to a keypoint annotation. Several
  existence-only checks (`"segmentation" in annotation`, not a truthiness/None guard) in
  `image_label.py::draw_annotations`, `image_label.py::start_polygon_edit` (the
  double-click handler, which iterates every annotation across every class), and
  `widgets/tools/eraser_tool.py` would otherwise misfire: a pose instance would render
  nothing, and any double-click anywhere on the canvas would raise
  `TypeError: 'NoneType' object is not subscriptable` as soon as one keypoint instance
  exists in the current image. `AnnotationController._keypoint_instance_bbox` was
  relocated to `utils.keypoint_instance_bbox` (delegate kept for `finish_keypoint`) so
  COCO import can reuse the same bbox-from-labelled-points fallback instead of a second
  copy.
- **Training + prediction (PR-3).** `YOLOTrainer.prepare_dataset` threads
  `main_window.keypoint_schemas` into `export_yolo_v5plus` so the prepared `data.yaml`
  carries `kpt_shape`/`flip_idx` whenever the exported set is pose-shaped. `train_model`
  adds a pre-flight guard — before Ultralytics ever starts, it compares the loaded
  model's `.task` (`"pose"` or not) against whether the prepared yaml has a `kpt_shape`
  key, and raises `ValueError` on a mismatch in either direction (pose model /
  non-pose dataset, or vice versa). No new UI code: the existing `TrainingThread`
  exception-to-`QMessageBox` path already surfaces any exception `train_model` raises,
  so this fails loud with a plain-language message instead of Ultralytics throwing a
  cryptic shape error mid-epoch. `on_fit_epoch_end` also surfaces `val/pose_loss` and
  `val/kobj_loss` alongside the existing `val/box_loss`/`val/seg_loss`, following the
  same defensive "missing key never disturbs the run" pattern. `predict()` no longer
  hardcodes `task='segment'` on the `self.model(...)` call — that would have silently
  mis-decoded pose outputs — so the task is whatever the loaded model actually is.
  Schema round-trips through **two tiers**: `_register_trained_model` writes the
  richer, hand-authored `keypoint_schema` (names + skeleton) into the registered
  model's `data.yaml` only when every trained class shares one identical schema in
  this session's `keypoint_schemas`; otherwise (or for a model trained outside this
  app) it still carries the bare `kpt_shape`/`flip_idx`, and `load_prediction_model`
  falls back to reconstructing a generic `kp0..kp{K-1}`-named schema
  (`sanitize_schema`'d, so a malformed yaml degrades to `None` rather than crashing)
  from those. `YOLOController.process_yolo_results` gains an `is_pose` branch —
  decided once per result set from `yolo_trainer.model.task`, since one checkpoint is
  exclusively one task — that builds temp instance dicts
  (`{"keypoints", "num_keypoints", "bbox", "category_name": "Temp-<class>", "score",
  "temp": True}`, deliberately **no `segmentation` key**, matching the discriminator
  everywhere else in this ADR) and seeds `keypoint_schemas["Temp-<class>"]` from
  `yolo_trainer.prediction_keypoint_schema` the first time that temp class appears.
  Every predicted point is stamped **v=2 (visible)**, a deliberate simplification:
  Ultralytics exposes only a per-point presence confidence, not a true COCO 3-state
  occlusion signal, and the instance already cleared the box-level `conf_threshold`
  gate, so a second per-point threshold would just be noise dressed up as occlusion
  data — the user hand-corrects via the existing right-click visibility toggle on
  review. The one real gap this closed in the otherwise-already-keypoint-safe
  Temp-class review path: `accept_visible_temp_classes` now carries a
  `"Temp-<class>"` schema over to the permanent class name on accept (if the
  permanent class already has a schema with a different K, it warns and keeps the
  existing one rather than silently overwriting), and `reject_visible_temp_classes`
  pops any orphaned `"Temp-<class>"` schema entry on reject — without this, rejected
  or renamed temp poses would leak stale schema entries into `keypoint_schemas`.
- **Consecutive runs reload a pristine model (PR-3 manual-testing fix).** Ultralytics
  mutates a `YOLO` object's `overrides` during `train()` (it drops the `'model'` key),
  so a *second* `train()` on the same instance raises `KeyError('model')`. `train_model`
  therefore reloads a fresh `YOLO(self.loaded_model_path)` at the start of every run
  (with a best-effort GPU reclaim first, per the "Releasing Model GPU Memory" rule).
  `loaded_model_path` is kept in sync with **every** `self.model` assignment — both
  `load_model` and `load_prediction_model` set it — so it is a single source of truth
  for "the model to train" (loading a trained `best.pt` for prediction and then hitting
  Train continues from *it*, not the stale original pretrained). Semantics: each run
  fine-tunes the **loaded** checkpoint, not the previous run's output; to continue
  training from a run's result, load its `best.pt` (Prediction Settings → Load Model, or
  the trained-model dropdown) and train again. The Training Progress log is also cleared
  at the start of each run so consecutive runs don't visually stack.

**Why pure helpers** (`core/keypoint_schema.py`, `utils.clamp_keypoints`,
`ImageLabel._keypoint_bounds/_scale_keypoints/_translate_keypoints`): schema
validation, clamping (ADR-024 bounds enforcement extended to keypoints), and the affine
geometry are unit-tested without Qt or a model (`test_keypoint_schema.py`,
`test_keypoint_geometry.py`, `test_utils.py`); the tool logic via a fake-context
`ImageLabel` (`test_keypoint_tool.py`); the controller/persistence end-to-end on a real
offscreen window (`test_keypoint_controller.py`).

**Alternatives considered**:
- *Standalone labeled points* — rejected for this issue: the user wanted COCO/YOLO-pose,
  which needs the ordered, fixed-K, skeleton-bearing instance model.
- *Per-instance schema* — rejected: COCO mandates one schema per category; per-class
  storage enforces it and keeps instances small.
- *Double-click vertex editing (as polygons use)* — rejected: it's bound to
  `start_polygon_edit`; the #40 handle path generalizes cleanly to a `kpt` kind.
- *Graphical skeleton editor* — deferred; a list-based dialog ships first (lowest risk).

**Consequences**:
- ✅ Pose classes can be defined, annotated, edited, saved/reloaded; the data model is
  COCO/YOLO-pose-shaped, and PR-2 confirms it round-trips through both formats losslessly
  (mod point names, which YOLO-pose doesn't carry). PR-3 closes the loop end-to-end:
  a pose-shaped dataset can be exported, trained as a YOLO-pose model, and the
  resulting checkpoint used for prediction, with predicted instances flowing through
  the same Temp-class accept/reject review as every other detector.
- ⚠️ Predicted keypoints always come back v=2 (visible) — Ultralytics doesn't expose
  a true 3-state occlusion signal at inference — so occluded points must be
  hand-corrected via the right-click toggle after accepting.
- ⚠️ Each "Train Model" run restarts from the currently-loaded checkpoint (Ultralytics
  can't reliably re-train the same in-memory object), not from the previous run's
  weights — to continue a run, load its `best.pt` and train again.
- ⚠️ Finishing early pads not-yet-placed points with v=0 at the origin; they don't
  render and (in PR-1) can't be relabelled via right-click (only v>0 points are
  hit-testable). Acceptable — v=0 means "not labelled" per COCO.
- ⚠️ Old builds opening a project with keypoint instances preserve but don't render them
  (the old `if/elif` ignores the key); the schema and instances survive a save.
- ⚠️ A YOLO-pose dataset built outside this app that genuinely mixes pose and non-pose
  classes, or hand-edits per-class `kpt_shape`, is out of scope — import applies one
  recovered schema uniformly to every declared class, and export refuses to mix them.

This builds on **ADR-022/023** (canvas selection + #40 handle editing), **ADR-024**
(bounds clamping), and **ADR-026** (snapshot undo) — see those for the machinery reused.

---

## ADR-030: Centralized stdlib `logging`; `print()` Banned in `src/`

**Status**: Accepted (issue #33)

**Context**: The codebase had ~307 `print()` calls and 12 `traceback.print_exc()`
sites across 23 files and no use of the stdlib `logging` module. Log level could
not be controlled, output could not be redirected or silenced, and diagnosing a
user report meant asking them to copy console spam. It also blocked the
error-handling cleanup (ADR-031): silent `except` sites had no `logger.exception`
target to migrate to.

**Decision**: Adopt stdlib `logging` with a single package-level console handler.

- New module `core/logging_config.py` exposes `configure(level=None)` and
  `get_logger(name)`. `configure()` installs one stderr `StreamHandler` on the
  package logger `digitalsreeni_image_annotator`, is idempotent (a second call
  adds no second handler), and is called once from `main.py:main()` **before**
  `QApplication` is created.
- Every module does `logger = get_logger(__name__)`; `configure()` derives the
  package root from its own `__name__` (not a hardcoded string), so all loggers
  share that root and inherit its handler/level whether the app is imported as
  `digitalsreeni_image_annotator` or `src.digitalsreeni_image_annotator`.
- Default level INFO; `--debug` argv flag or `IMAGE_ANNOTATOR_DEBUG` env var
  switches to DEBUG.
- No third-party logging dependency. `print()` is banned in `src/` and enforced
  in review.
- Level policy: debug = diagnostic chatter (per-item loop progress, shape/metadata
  dumps), info = user-relevant state changes, warning = soft failures outside
  `except`, `logger.exception` / `error(exc_info=True)` = inside `except`. See the
  level table in [docs/08](08_crosscutting_concepts.md#logging-and-debug-output).

**Consequences**:
- One switch turns diagnostic chatter on/off; every log line carries its module
  name (`%(name)s`).
- `QMessageBox` / dialog behaviour is unchanged — logging is the diagnostic
  channel, dialogs are the user channel. The migration touched only the output
  channel of existing prints, never user-visible messaging.
- Enables ADR-031 (error-handling convention): swallowed exceptions now have a
  `logger.exception` target.
- Tests configure the package logger explicitly
  (`tests/unit/test_logging_config.py`); package loggers have `propagate = False`
  after `configure()`, so `caplog` tests must enable propagation on the specific
  logger or attach `caplog.handler`.

---

## ADR-031: Raise in Core, Catch + Dialog at the UI Boundary, No Silent `pass`

**Status**: Accepted (issue #34)

**Context**: `docs/11`'s "Inconsistent Error Handling" debt — a mix of raised
exceptions, message boxes, and silent `return None`, plus several
`except Exception: pass` and one bare `except:` that swallowed real failures
(CUDA cleanup, a broken MLflow log sink, DICOM VOI-LUT) with no log line. A
swallowed error left nothing in the bug report.

**Decision**: Adopt one convention (also written into docs/08):
1. Core / inference / io / training modules **raise**; they never dialog and
   never return `None` to signal a failed user action.
2. Controllers / dialogs (the UI boundary) **catch**, `logger.exception(...)`,
   and surface a `QMessageBox`.
3. Catch the **narrowest** exception type. `except Exception` only at a UI
   boundary or a documented crash-safety barrier.
4. **Never `pass` silently** — log (`exc_info=True` / `.exception`) or use a
   narrow type + `# reason` comment.
5. Bare `except:` is **banned**.

Concrete changes: the seven enumerated silent sites now log; the one bare
`except:` (`dicom_converter.apply_window_level`) became `except Exception` +
`logger.warning`. A friendly OOM dialog rides along: `core/torch_utils._is_oom`
(torch-free, unit-tested) drives a "pick a smaller model" message in
`SAMController.change_sam_model`, while non-OOM failures keep the generic dialog;
both reset the model selector.

**Consequences**:
- Every swallowed exception is now visible (logged) or surfaced (dialog).
- The three deliberate narrow catches stay as the convention's positive
  examples: `except TypeError: pass  # already disconnected` (yolo /
  sam_train controllers) and `except ImportError` in `main.py`.
- `mlflow_tracker`'s outer `except Exception` ("never let tracking abort
  training") and the ADR-013 `InferenceBusyError` re-entrancy swallow are
  documented barriers, not silent-swallow sites, and are unchanged.
- Depends on ADR-030 (logging) for the `logger.exception` target.

---

## ADR-032: Silent Recovery Autosave for Unsaved Projects via a QSettings-Known Location

**Status**: Accepted (issue #41)

**Context**: Autosave is event-driven — every mutation calls `auto_save()`. Before a
project has ever been saved there is no `.iap` to write to, so the old code popped a
modal "save now?" from inside the mutation handler. Declining it (or a crash) lost all
work, and a modal raised from deep in a mutation re-enters the event loop mid-edit.
Tracked as the "Autosave Doesn't Ask for File Location" debt. Related: ADR-005 (load
guard), ADR-020 (QSettings precedent).

**Decision**: With no `current_project_file`, `auto_save()` writes a **silent** recovery
snapshot instead of prompting. The snapshot is exactly a `build_project_data()` dict — a
new pure serializer factored out of `save_project()` (no dialogs, no file I/O, no image
copying) — serialized like a real `.iap`, written atomically (temp file + `os.replace`)
to `QStandardPaths.AppDataLocation/recovery/unsaved.iap.recovery`, its path stored under
the QSettings key `recovery/pending_path` (same org/app as `app_settings`). A trivially
empty session writes nothing. On the next launch, `main()` — after `window.show()`, never
the constructor, so tests that build `ImageAnnotator()` don't trigger it — calls
`ProjectController.offer_recovery()`, which offers to restore it. A real save (or New
Project) calls `clear_recovery()`.

**Consequences**:
- ✅ Work is never silently lost before the first save; no modal ever fires from `auto_save`.
- ✅ Restore reuses `load_project_data` unchanged (the snapshot has the `.iap` shape).
- ✅ **Failure-path policy:** a *successful* restore **keeps** the snapshot (the project is
  still unsaved) until the first real save retires it via `clear_recovery`, so a re-crash
  before that save can still re-offer the work; a *corrupt or partially-loaded* snapshot is
  **dropped** on the failed restore (with the UI reset to empty) so it can't nag on every
  launch. A user who restores and then quits cleanly without editing is re-offered it next
  launch — intentional, since the work is genuinely still unsaved.
- ⚠️ The recovery pointer lives in per-user QSettings, so it is machine-local — acceptable,
  since a restore is always on the same machine that crashed.

---

## ADR-033: Dual Absolute + Relative Image Paths in `.iap`; Relative-First Resolution; Load-Time Validation

**Status**: Accepted (issue #42; supersedes ADR-003)

**Context**: `.iap` stored only absolute image paths, so a project folder couldn't be
moved or shared (ADR-003). In practice the loader already rebuilt paths from
`<project_dir>/images/<file_name>` and ignored the stored absolutes — they were dead data
on load and merely leaked the author's machine paths.

**Decision**: `build_project_data()` writes a portable `image_paths_rel` map (POSIX
separators via `PurePath(os.path.relpath(...)).as_posix()`; a cross-drive `ValueError`
just omits that entry) **alongside** the unchanged absolute `image_paths` (dual storage,
so older app versions still open new files). `image_paths_rel` is written only when a
project dir exists — recovery snapshots (ADR-032) for an unsaved project have none and
rely on the absolutes. On load, `ProjectController.resolve_image_path()` returns the first
that exists: relative → absolute (revives the old dead data, fixing images referenced
outside `images/`) → the historical `images/` convention → missing. A new pure
`core/project_schema.py::validate_project_data` runs right after `json.load` in
`open_specific_project`, raising `ValueError` (surfaced by the existing backup-restore +
error dialog) on a structurally broken file; it is deliberately lenient about unknown keys
(the format keeps growing — keypoint schemas, DINO config, relative paths).

**Consequences**:
- ✅ A project folder (`.iap` + `images/`) is portable across machines/OSes.
- ✅ v1 projects (no `image_paths_rel`) resolve exactly as before via the `images/` fallback.
- ✅ A broken `.iap` fails with an actionable message instead of a random traceback.
- ⚠️ Dual storage means the absolute paths still appear in the file; they are kept only for
  backward-compatible opening, and relative paths win on load.

---

## ADR-034: Split ImageLabel into Renderer + Edit-Gesture Collaborators (Dispatch Stays)

**Status**: Accepted (issue #46)

**Context**: `widgets/image_label.py` had grown to ~1850 lines — the largest file in
the codebase — mixing three concerns: canvas painting/overlays, the direct-manipulation
edit-gesture state machine (ADR-023 bbox/segmentation handles, ADR-029 keypoint edits),
and event dispatch/zoom/pan/tool routing. Upcoming canvas-heavy work (video timeline
overlays #48, SAM 3 review/tracking #51) would pile onto it further.

**Decision**: Extract two collaborators by **composition**, leaving `ImageLabel` as the
dispatcher that owns state and thin delegates — a strictly behavior-preserving move
(diff = moves + delegates only, no logic/ordering/name changes):
- `widgets/canvas_renderer.py::CanvasRenderer` — constructed with the label
  (`self.label = image_label`); owns `draw_annotations`, `draw_temp_annotations`,
  `draw_tool_size_indicator`, `draw_sam_bbox`, `draw_selection_rect`,
  `_draw_keypoint_annotation`, `_draw_selection_overlay`, `draw_editing_polygon`,
  `calculate_centroid`, the painter helpers `_pen_w`/`_overlay_font`, and the
  `_SELECTION_COLOR` constant. `paintEvent` stays on `ImageLabel` and calls into
  `self.renderer.*` in the identical layer order; the per-tool `handler.paint_overlay`
  loop (ADR-019) stays in `paintEvent`.
- `widgets/edit_gestures.py` — seven module-level **pure functions**
  (`bbox_handle_points`, `resize_bbox`, `scale_segmentation`, `translate_segmentation`,
  `scale_keypoints`, `translate_keypoints`, `sync_bbox_key`) plus `class EditGestures`
  for the 15 stateful gesture methods (`_begin_shape_edit`, `_update_bbox_drag`,
  `_commit_bbox_drag`, keypoint edits, cursor updates, …). The state fields
  `bbox_edit`/`editing_keypoint`/`selection_*`/`highlighted_annotations` and
  `_BBOX_HANDLE_CURSORS` **stay on the label**; `EditGestures` mutates them via
  `self.label.*` and emits via `self.label.<signal>.emit(...)`.
- **Compatibility surface**: `ImageLabel` keeps `staticmethod` aliases
  (`_bbox_handle_points = staticmethod(edit_gestures.bbox_handle_points)`, ×7) so
  `ImageLabel._resize_bbox(...)` / `label._resize_bbox(...)` call sites and tests are
  unchanged; `_SELECTION_COLOR = CanvasRenderer._SELECTION_COLOR` is re-exported; a
  one-line delegate exists for every moved render/gesture method (exact names/sigs).
  `tests/unit/test_module_split.py` locks the seven alias identities +
  `_SELECTION_COLOR` re-export.

**Consequences**:
- ✅ `image_label.py` 1854 → 1197 lines; the two new modules carry module docstrings
  stating what they own and what stays on the label.
- ✅ Zero functional change — no signal signatures, dispatch order, paint order, or
  state-field names changed; every pre-existing test passes unmodified (the suite goes
  from 686 to 688 passed / 3 skipped purely from the +2 new identity-lock tests).
- ⚠️ The compatibility layer (~50 one-line pass-through delegates + 7 staticmethod
  aliases) is **transitional scaffolding**: it keeps every existing call site working for
  a zero-risk split. As callers migrate to reach `image_label.renderer.*` / the gesture
  collaborator directly (e.g. during #48/#51 canvas work), the delegate layer should be
  deleted rather than left to ossify — otherwise the split only adds an indirection tax.
- ✅ The shared handle geometry (`bbox_handle_points`) stays single-sourced, upholding
  the "visual == grab" invariant (`_draw_selection_overlay` and `_bbox_handle_at` both
  resolve to the same function object).
- Cross-references ADR-018 (CanvasContext), ADR-019 (tool handlers), ADR-022/023
  (selection + shape editing being relocated), ADR-016 (the AST inline-import gate that
  guards module moves).

---

## ADR-035: Flat Grouped Image List with Derived Status Badges (No Tree, No Thumbnails)

**Status**: Accepted (issue #43)

**Context**: The image list had alphabetical sort and an annotation-status filter
(#27/#60) but gave no at-a-glance "is this done?" signal and no way to organise a
large dataset. A tree/`QTreeWidget` with group headers or per-image thumbnails were
both considered and rejected: many consumers (DINO batch navigation, COCO import
reconciliation, `apply_image_filter`) read `image_list.item(i).text()` as a **file
name** and rely on the positional invariant `all_images[i] ↔ item(i)`, so any header
/ separator row would be interpreted as a phantom image; thumbnails were explicitly
out of scope (memory + async decode complexity).

**Decision**: Stay on the flat `QListWidget` and express both features as derived,
non-structural overlays on the existing sort/filter machinery:
- **Status badge** = a painted-pixmap `QIcon` per row (filled green dot if
  `image_has_annotations`, hollow gray otherwise), cached per `(state, dark_mode)`
  and rebuilt on theme flip. Nothing stored; both states derived.
- **Group** = an optional `"group"` key on the `all_images` entry (no registry; set
  derived by `sorted({...})`). Grouped images cluster via the sort key
  `(group.casefold(), name.casefold(), name)`; the group is shown only in the row
  tooltip so item text stays the bare file name. A second combo filters by group,
  OR-combined with the status filter. Persisted in the `.iap`; no load-time
  restoration is needed because `load_project_data` aliases `all_images` to the
  parsed `images` list and the load loop doesn't rebuild it.

**Consequences**:
- ✅ Both features ride the `update_slice_list_colors → apply_image_filter` contract,
  so badges/marks stay correct after every annotation mutation with no new call sites.
- ✅ The `.text() == file_name` contract and positional invariant are preserved, so
  DINO batch nav and COCO import are unaffected (regression-guarded).
- ✅ No hardcoded colours (painted pixmaps + palette-safe dot colours), dark-mode safe.
- ⚠️ Groups are a flat single-level tag, not nested folders — sufficient for the PRD
  US-1 remainder; a real hierarchy would need the tree widget this ADR rejects.
- Status-badge colours are theme-tuned (a brighter green / lighter gray on the dark
  sidebar), so the `(annotated, dark_mode)` cache dimension and the `on_theme_changed`
  rebuild produce genuinely different pixmaps — not dead machinery.

---

## ADR-036: Lazy Slice Extraction with a Bounded Shared LRU (Retained Source Array)

**Status**: Accepted (issue #45)

**Context**: Opening a multi-dimensional TIFF/CZI eagerly converted **every** slice to
an RGB888 `QImage` in `ImageController.create_slices` and held them all for the session
(`image_slices[base] = [(name, qimage), ...]`). A 5D 10×20×3 stack of 2048² frames is
~600 slices ≈ 7.5 GB of live QImages, plus a create-time peak of source-array +
growing-QImages. Annotation storage is keyed by slice *name* and is unaffected — only
the pixel data needed to become lazy.

**Decision**: Introduce `core/slice_cache.py` and make QImage materialisation lazy behind
a process-wide bounded LRU, keeping the exact `(name, qimage)` interface every consumer
uses (**Strategy A** — retain the already-decoded source ndarray, materialise QImages on
demand):
- `SliceProvider` retains the `tif.asarray()`/CZI ndarray and precomputes the ordered
  `names` + `name → full-index` map with the **byte-identical** naming logic of the old
  `create_slices` (`{base}_{dim}{index+1}`), then `extract(name)` reconstructs one slice
  through the exact ADR-010 pipeline (`convert_to_8bit_rgb`/`normalize_array`/
  `array_to_qimage`) — a **fresh** QImage each call (never mutate a cached one; the SAM
  worker may be reading it, ADR-013).
- `LazySliceList` is the drop-in for the old tuple list: `get(name)`, `__getitem__`,
  `__iter__` (one-at-a-time), `__len__`/`__bool__`/`.names`, `prefetch_around(name)`
  (pins current ±1 for instant Up/Down nav), `release()`. It is stored as BOTH
  `image_slices[base]` and `mw.slices` (the same object — several paths compare them).
- A single module-level `SliceLRU` (keyed `(provider_id, name)`, `LRU_CAPACITY = 8`)
  bounds live QImages across ALL open stacks; `evict_prefix`/`release_slices` drop a
  stack's entries on delete. Name-only consumers (save, `image_has_annotations`,
  `update_slice_list`, navigation membership checks) use `slice_names(...)` so they touch
  **no** pixels; iterating pixel consumers (exporters, SAM-dataset build, DINO batch,
  `io_controller`) keep working unchanged via `__iter__`.

**Consequences**:
- ✅ Opening a stack no longer decodes every slice; live QImages are bounded by the LRU;
  `save_project` materialises zero pixels (test-asserted via an `extract` spy).
- ✅ Byte-identical slice names + pixels (lazy == eager, regression-tested), so existing
  `.iap` annotations and exports are unaffected; Up/Down neighbours are prefetched.
- ✅ `slice_names()`/`release_slices()` also accept a plain `[(name, qimage)]` list, so
  legacy/test call sites that inject plain lists keep working with no pixel decode.
- ⚠️ **Strategy A retains the decoded source ndarray per open stack**, so peak RSS is
  bounded by the LRU *for QImages* but not for the array. Full array-free reading
  (memmap/zarr per slice; CZI lazy read) is a deliberate follow-up. The dominant
  all-QImages-in-RAM cost and the create-time peak are eliminated regardless.
- ⚠️ The LRU keys on `id(provider)`; providers are `release()`d before being replaced
  or deleted and `clear_all` wipes the cache, so a recycled `id()` cannot alias a stale
  QImage. Every dataset-replacing path (`remove_image`/`delete_selected_image`/
  `redefine_dimensions`/`open_images`/`clear_all`) drops the outgoing `image_slices`
  entries + their LRU cache so the retained source array cannot leak for the session.
- Feeds issue #47: video frames reuse the same lazy-slice contract (`None`-payload
  frames resolved on demand) rather than introducing a parallel cache.

---

## ADR-037: Video Frames as Multi-dimensional Slices (Lazy, Keyed `_F#####`, Reusing LazySliceList)

**Status**: Accepted (issue #47)

**Context**: Video annotation is the next platform feature. The app already handles
"one file, many annotatable 2D planes" for multi-dim TIFF/CZI stacks — per-slice
annotation storage, a slice list, Up/Down navigation, per-slice colour, save/load and
export all key on a slice *name*. The cheapest correct design is to treat a video frame
exactly like a stack slice. The one thing frames must NOT inherit is eager loading: a
300-frame 1080p clip is ~1.8 GB of QImages.

**Decision**: Model a video as a stack whose slices are frames, and **reuse the #45
lazy-slice machinery** rather than a parallel frame cache (the #45/#47 reconciliation):
- `core/video_handler.py::VideoHandler` wraps `cv2.VideoCapture` — metadata read once,
  `get_frame(idx)` seeks + decodes ONE frame (BGR→RGB via `cvtColor`, mandatory `.copy()`
  because the numpy buffer dies at return), `release()` idempotent. Decoding only, no
  internal LRU. GUI-thread only (`cv2.VideoCapture` is not thread-safe).
- `VideoSliceProvider` is **duck-type compatible** with `slice_cache.SliceProvider`
  (`provider_id` / `names` / `extract(name)`), so a video's `image_slices[base]` is an
  ordinary `LazySliceList` and the shared bounded `SliceLRU` (ADR-036) caps live frame
  QImages. Frame slice keys are `frame_key(base, idx) = f"{base}_F{idx:05d}"` (0-based),
  parsed by `parse_frame_index` (`_F(\d+)$` anchored at end so it never matches a
  `stack_T1_Z5` key).
- `ImageController.load_video` builds the handler + provider + `LazySliceList` (stored as
  both `image_slices[base]` and `mw.slices`); `add_images_to_list` gains an
  `is_video(...)` branch setting `is_multi_slice=True`, `is_video=True`,
  `video_metadata=handler.metadata()`. The Add/Open file dialogs accept `*.mp4 *.avi
  *.mov` via the shared `video_handler.file_dialog_filter()` (video globs derived from
  `VIDEO_EXTS`), used by `add_images` (the live "Add Images / Videos" button),
  `open_images` and `load_missing_images` so the filter can't drift from `is_video()`.
  Handlers live in `mw.video_handlers[base]` and are `release()`d on every drop path
  (delete/remove/redefine/`open_images`/`clear_all`). `.iap` round-trips `is_video`
  +`video_metadata`; load branches to `load_video`.

**Consequences**:
- ✅ Because a video is a `LazySliceList`, EVERY existing slice consumer
  (`switch_slice`/`activate_slice` `.get()`+prefetch, `switch_image` `slices[0]`, `.names`,
  `__iter__`, exporters, DINO batch, save-touches-no-pixels, delete→`release_slices`)
  works for video **unchanged** — no parallel resolver, no `None`-placeholder path.
- ✅ Frames decode lazily and are bounded by the shared LRU for interactive use
  (navigation, display); opening a video decodes nothing beyond frame 0;
  `save_project` decodes zero frames (test-asserted).
- ⚠️ One batch path is NOT LRU-bounded: `DINOController._collect_dino_batch_work_items`
  builds a flat `[(name, QImage), …]` list, so running DINO batch over a long video
  materialises all its frames at once (pre-existing for stacks; more costly for video).
  Streaming that collector frame-by-frame is a documented follow-up.
- ✅ Annotation storage, per-frame independence, navigation, save/load and export path
  matching all come for free from the slice machinery.
- ⚠️ `cv2.VideoCapture` seeking is per-codec-variable; heavy random scrubbing re-seeks.
  The LRU + `prefetch_around(±1)` keep sequential nav responsive; whole-video
  pre-decode is deliberately never done.
- ⚠️ Base-name collision (`video.mp4` vs `video.tif` → same `image_slices` key) is
  refused with a warning in `add_images_to_list`.
- Feeds #48 (timeline over `video_handlers`/frame keys) and #51 (SAM 3 tracking seeds a
  mask on a frame and writes per-frame annotations).

---

## ADR-038: SAM 3 Integration Path (Spike — #49)

**Status**: Accepted (issue #49 spike; findings confirmed in-env and implemented by #50 — see ADR-039. The D3 video items — arbitrary-frame seed/backward propagation, per-frame confidence, long-video memory — were **resolved on a real GPU run by #51**; see ADR-040 Consequences.)

**Context**: Milestone D plans two SAM 3 features — native text-prompt segmentation reusing the DINO
review workflow (#50) and video object tracking (#51). Both hinge on facts that were unverified when
the milestone was scoped: whether SAM 3 is consumable through Ultralytics (ADR-002), the exact
text/video APIs, model sizes, licensing and minimum dependency versions. This ADR records the spike
findings. Every claim carries a dated source; the environment was probed directly (`pip show
ultralytics`, an import smoke) rather than trusting docs.

**Findings (checked 2026-07-21):**

1. **Distribution** — SAM 3 (Meta, released 2025-11-20; "Promptable Concept Segmentation") is **fully
   integrated into Ultralytics as of v8.3.237** (PR ultralytics#22897; discussion #22378). This
   **upholds ADR-002** — Ultralytics stays the integration layer; no `facebookresearch/sam3` vendor
   dependency is needed. This dev environment already has `ultralytics 8.4.51`, `torch 2.11.0+cu128`,
   and `from ultralytics.models.sam import SAM3SemanticPredictor` imports successfully.
2. **Text-prompt image API** — `SAM3SemanticPredictor(overrides=dict(model="sam3.pt", task="segment",
   conf=0.25))`; `predictor.set_image(img)`; `results = predictor(text=["person", "bus"])` (a list of noun
   phrases). Returns a Results object with `.masks` and `.boxes` (numpy via `.cpu().numpy()`; boxes carry
   confidence). Image-exemplar prompting via `predictor(bboxes=[[x1,y1,x2,y2]])`; feature reuse via
   `inference_features(features, src_shape=..., text=[...])`. **Confidence is the single knob (`conf`)** —
   our DINO UI's per-class box/txt/nms thresholds map only to `conf` (reuse `box_thr` as the confidence
   filter; txt/nms have no SAM 3 equivalent). SAM 2-style visual prompts:
   `SAM("sam3.pt").predict(source, points=..., bboxes=...)`.
   **Correction (probed in ultralytics 8.4.51):** the published Ultralytics docs example also passes
   `quantize=16` and `mode="predict"`; `get_cfg(overrides=dict(..., quantize=16))` **raises**
   `'quantize' is not a valid YOLO argument`, and `mode` is redundant for a predictor — **D2 must omit
   both**. This is why the spike probes the environment instead of trusting the docs verbatim.
3. **Video tracking API** — `SAM3VideoPredictor` (visual: box/point/mask) and `SAM3VideoSemanticPredictor`
   (text). `predictor(source="video.mp4", bboxes=[[...]], stream=True)` yields **one Results per frame** in
   file order; `stream=True` is a frame-by-frame generator, omitting it processes the whole video. Docs seed
   from a **file path**; whether the predictor accepts incrementally-supplied numpy frames (our C1
   `VideoHandler` decodes on demand) is **UNRESOLVED** — resolve by testing on a GPU box before D3 commits
   to feeding frames vs. a path. **Backward propagation is not explicitly documented**; the predictor
   propagates automatically from the seed, so D3 should treat backward as "feed frames in reverse index
   order" and verify (this is an inference from the streaming-forward API, **not confirmed** without weights).
   **Long-video predictor memory behaviour is UNRESOLVED** and is D3's verify-first item
   (measure RSS over 500+ frames; decide whole-video vs chunked).
4. **Models & download** — `sam3.pt` is **3.45 GB, 473.6M params** (per the Ultralytics SAM 3 docs
   model table, docs.ultralytics.com/models/sam-3/). It is **NOT auto-downloaded**: the user
   must request access on the gated HF page `facebook/sam3`, accept Meta's terms, and download/place it
   manually (or pass a full path). Mirror our DINO gated-download UX + a clear status message. A CLIP
   dependency quirk may require `pip install git+https://github.com/ultralytics/CLIP.git`.
5. **License** — code + weights are under Meta's **custom “SAM License”** (NOT MIT/Apache/GPL, NOT
   OSI-“open source”). Permits research **and** commercial use with restrictions: no military/ITAR use, obey
   sanctions, no reverse-engineering; a **patent-retaliation clause** (sue over SAM 3 patents → license
   terminates); **redistributed weights/derivatives must stay under the SAM License**. Consequence: **we
   must NOT vendor or redistribute `sam3.pt`** — users accept Meta's terms and download it themselves (same
   posture as our gated DINO/HF models). Source: https://huggingface.co/facebook/sam3/blob/main/LICENSE.
6. **Minimum versions** — **ultralytics >= 8.3.237** (per the Ultralytics SAM 3 docs; our floor is
   `>=8.3.27,<9`; #50 must raise it to `>=8.3.237,<9`). **Caveat**: the installed 8.4.51 only proves
   `>= 8.3.237`, not that 8.3.237 was the *first* integrated version — **D2 must independently re-confirm
   the exact first-integrated release before pinning the floor** (check the Ultralytics changelog/release
   notes). Python/torch/torchvision minima are not separately pinned by the SAM 3 docs; our installed torch
   2.11 + Python 3.10+ floor satisfy it (verified by the successful import). No conflict with the
   #38-reconciled `<9` upper bound.
7. **CPU fallback** — GPU latency ~30 ms/image on an H200 at 100+ objects. **CPU is not documented and is
   impractical** given 3.45 GB / 473.6M params. Recommendation: document SAM 3 as **GPU-recommended**; keep
   the Grounding-DINO two-stage pipeline as the CPU fallback (D2 keeps DINO selectable).
8. **Decision for D2 (#50)** — Mirror `SAMUtils` via Ultralytics: a new `inference/sam3_utils.py` wrapping
   `SAM3SemanticPredictor`, reusing `sam_utils._run_sync` / `_qimage_to_numpy` / `_mask_to_polygon`
   (the shared `_inference_in_flight` flag then serialises SAM 3 against SAM 2/DINO — desirable). New
   packaging entries: raise `ultralytics` floor to `>=8.3.237,<9`; document the gated `sam3.pt` weights
   and the CLIP quirk. Real end-to-end model verification requires a GPU + approved weights (cannot run in
   CI or a dev box without the gated weights) — stubbed/monkeypatched tests cover the wiring; the real-model
   check is a documented manual step (CLAUDE.md inference checklist).

**Go/no-go for D3 (#51)**: GO on the API surface (SAM3VideoPredictor exists, streaming per-frame results,
seed-then-propagate), with two verify-first gates carried into D3: (a) numpy-frame vs file-path seeding,
(b) long-video predictor memory → whole-video vs chunked. Both are measured in the D3 branch before
building on them.

**Consequences**:
- ✅ D2/#50 and D3/#51 are unblocked with a concrete integration route, API names, versions and license.
- ✅ ADR-002 (Ultralytics for SAM) is upheld, not superseded — no new backend dependency.
- ⚠️ The weights are gated + non-redistributable (SAM License) and large (3.45 GB); CPU is impractical.
  These become the D2/D3 UX constraints (gated download, GPU-recommended, DINO fallback for CPU).
- ⚠️ Two D3 facts remain UNRESOLVED (numpy-frame seeding; long-video memory) and are its verify-first items.

**Sources** (dated 2026-07-21): docs.ultralytics.com/models/sam-3/ ; github.com/ultralytics/ultralytics
docs/en/models/sam-3.md ; newreleases.io … ultralytics v8.3.237 ; huggingface.co/facebook/sam3 (gated) +
/blob/main/LICENSE ; github.com/orgs/ultralytics/discussions/22378 ; local `pip show ultralytics` (8.4.51).

---

## ADR-039: SAM 3 Text-Prompt Segmentation via the DINO Review-Pipeline Reuse

**Status**: Accepted (issue #50; implements ADR-038's D2 recommendation)

**Context**: Text-prompted segmentation existed as a two-stage pipeline — Grounding-DINO turns
per-class phrases into boxes, then SAM 2 refines each box into a mask — wrapped in a mature review
workflow (temp-annotation overlay, Enter/Escape accept/reject, batch over images+slices, auto-accept,
`.iap` persistence). SAM 3 (ADR-038) does text→masks natively in one stage. The goal was to add SAM 3
WITHOUT a second review UI.

**Decision**: Plug SAM 3 in as a new **producer** at the exact spot "DINO boxes → SAM masks" sits, and
reuse everything downstream verbatim:
- New `inference/sam3_utils.py::SAM3Utils` mirrors `SAMUtils`/`DINOUtils`: lazy-imports
  `SAM3SemanticPredictor` (ADR-012/016), constructs with `overrides=dict(model="sam3.pt",
  task="segment", conf=<floor>, device=<resolved>, save=False, verbose=False)` — **no `quantize`/`mode`**
  (ADR-038: `quantize` raises in ultralytics 8.4.51); `save`/`verbose` are `False` to stop a
  `runs/segment/predict/` dir + per-call console summary on every detection — and reuses
  `sam_utils._run_sync` / `_qimage_to_numpy` / `_mask_to_polygon` + the shared
  `_inference_in_flight` flag (so SAM 3 serialises against SAM 2/DINO on the GPU). `detect_text(image,
  class_configs)` returns per-instance `{class_name, score, segmentation, bbox}`; `box_thr` is the
  confidence filter (SAM 3's only knob — `txt_thr`/`nms_thr` ignored). Weights (`sam3.pt`, 3.45 GB,
  gated) are never auto-downloaded; `_weights_available()` drives a "request access + place sam3.pt"
  status, mirroring the DINO gated-download UX.
- `DINOController` gains a `_run_text_detection(qimage) -> (results, sam_results)` choke point used by
  BOTH single + batch: for SAM 3 it splits each instance into a `results`-shaped `{class_name, score,
  bbox, source:"sam3"}` and a parallel `sam_results`-shaped `{segmentation, score}` — SAME length +
  order — so the existing zip/commit/temp-attach logic is byte-identical; for DINO it runs the unchanged
  two-stage path. SAM 3 skips the "No SAM Model" guard; DINO keeps it.
- Produced temps carry `"source": "sam3"`; every `source == "dino"` check (esp. the
  `DINOReviewEventFilter` Enter/Escape gate) was widened to `in ("dino", "sam3")` via a
  `.get("source", "dino")` default that keeps DINO's un-tagged dicts resolving to `"dino"`. SAM 3 batch
  results land in the SAME `dino_batch_results` dict (the single-field `_refresh_dino_temp_for_current`
  re-sync is untouched).

**Consequences**:
- ✅ One-stage SAM 3 masks flow through the entire existing review/accept/batch/auto-accept/undo/persist
  machinery with zero new review UI; commits reuse `_commit_dino_results` (so `record_history` fires —
  undoable).
- ✅ Grounding-DINO two-stage path is **functionally** unchanged — identical final annotations/temps,
  so its tests pass unmodified. (The single path drops a transient "Running SAM…" status line and
  unifies the two error-dialog titles into one — the only intra-run UX differences.) The model picker's
  "SAM 3 (text prompt)" entry sits alongside the DINO models.
- ✅ `ultralytics` floor raised to `>=8.3.237,<9` (ADR-038). SAM 3 lazy-imports, so older installs still
  launch and the app stays importable.
- ⚠️ Real end-to-end verification needs a GPU + the gated 3.45 GB `sam3.pt` (cannot run in CI or a
  weightless dev box) — the wiring is covered by monkeypatched/stubbed tests; the real-model check is a
  documented manual step (CLAUDE.md inference checklist). CPU is impractical — DINO stays the CPU fallback.
- ✅ Real-model check **done** (2026-07-22, RTX 4070, ultralytics 8.4.51): `detect_text` on `bus.jpg`
  returned sensible masks (bus 0.97 + 6 persons; `box_thr=0.25` filtered the raw 3 buss / 31 persons down
  to 1 + 6), and the live `SAM3SemanticPredictor` API (`set_image(np)` + `__call__(text=…)` →
  `.masks.data` / `.boxes.conf` / `.boxes.xyxy`) matches the code. Two real-only findings, both handled:
  (a) first use pip-installs `clip` + `timm` via ultralytics AutoUpdate (needs network — documented, not
  vendored; despite ultralytics' "restart runtime" warning the install+import complete in-process, so the
  first detection succeeds without a restart — observed on a clean env); (b) `save`/`verbose` are forced
  `False` (added to the overrides) to avoid a `runs/` dir + console spam on every detection.

---

## ADR-040: SAM 3 Video Object Tracking — Standard Commit Path, Per-Frame Undo + Run Rollback

**Status**: Accepted (issue #51; builds on ADR-037 video, ADR-039 SAM 3, ADR-026 undo)

**Context**: With videos as lazily-decoded frame slices (#47) and SAM 3 wired in (#50), the
flagship video feature: select an object's mask on one frame and track it across the clip,
writing a per-frame annotation. The ADR-038 verify-first items were **RESOLVED on a real GPU
run** (2026-07-23, RTX 4070, ultralytics 8.4.51 + gated `sam3.pt`); the design below was updated
to match what the real `SAM3VideoPredictor` actually does (see Consequences).

**Decision**:
- **Backend** — `SAM3Utils.track(video_path, seed_idx, seed_bbox, direction, should_cancel)`
  runs the WHOLE propagation inside ONE `_run_sync` call (the worker never touches Qt) and
  returns `[(frame_idx, {"segmentation","score"} | None), …]`. ALL real-model interaction is
  isolated in `_track_blocking` (the single monkeypatch seam): it lazy-imports
  `SAM3VideoPredictor`, constructs with the SAME overrides as detect (`model/task/conf/device`,
  no `quantize`/`mode`), seeds via the **video file path** + bbox with `stream=True`, and maps
  each per-frame result through `_mask_to_polygon`. **Arbitrary-frame seeding + bidirectional
  propagation are implemented and VERIFIED**: ultralytics' streaming predictor only seeds the
  FIRST frame and propagates forward, and both it and the low-level `init_state` path reject a
  frame *list* (`mode=="images"`), so `_track_blocking` slices the clip and re-muxes a short temp
  video per run -- forward = `frames[seed_idx:]`, backward = `frames[seed_idx::-1]` -- each seeding
  the bbox on its own frame 0 (== `seed_idx`) and mapping the streamed position back to the real
  index; `direction="both"` (the controller default) runs both and de-dupes the shared seed frame.
  `torch._dynamo.config.disable = True` is set before the video predictor (its encoder's
  `torch.compile` needs Triton, which has no Windows build); overrides add `save=False`/`verbose=False`.
- **Controller** — new `TrackingController`: `can_track()` (active video + SAM 3 loaded + exactly
  one selected annotation carrying a `"segmentation"` — pose instances excluded, ADR-029);
  `run_tracking()` (confirm dialog + confidence-threshold spinbox default 0.5; **modal**
  `QProgressDialog` blocks GUI navigation during the track and feeds `should_cancel`). Results
  route: `score >= threshold` → `_commit_tracked_result` (MIRRORS `_commit_dino_results`:
  `record_history(frame_name)` FIRST, `source=="sam3-track"`, a shared `track_run` uuid,
  per-class `number`, write to `image_label.annotations` if current else
  `all_annotations[frame_name]`); `0 < score < threshold` → a temp entry in `dino_batch_results`
  (`source=="sam3"`) so the EXISTING DINO batch-review pipeline + Enter/Escape filter handle it
  verbatim; `None` → nothing. The seed frame is skipped (it already carries the source). One
  `auto_save()` at the end, not per frame.
- **Undo granularity** (the explicit decision): per-frame Ctrl+Z undoes ONE frame (each commit
  `record_history`s its own key); `undo_last_track()` is the bulk convenience — removes every
  annotation whose `track_run == run_id` across the run's frames (per-frame `record_history`
  first, so it too is undoable). Rollback finds annotations by EXACT `track_run` match.
- **Timeline** — `VideoTimeline.set_frame_states({idx: "annotated"|"tracked"|"needs_review"})`
  paints palette-derived coloured segments (precedence needs_review > tracked > annotated);
  `set_annotated_frames` delegates to it (C2 back-compat). `needs_review` derives from
  `dino_batch_results` keys, `tracked` from `sam3-track` annotations.

**Consequences**:
- ✅ Tracked masks are ordinary annotations — saved, exported, undoable — with zero new review UI
  (uncertain frames reuse the DINO pipeline; commits reuse the `record_history` discipline).
- ✅ The model call is a single stub seam; the full controller/timeline logic is covered by
  stubbed tests (per-frame + run-rollback undo, commit/review routing, pose exclusion,
  save/reload). Real-model tracking is a documented manual GPU step.
- ✅ **Verified on the real model** (2026-07-23, RTX 4070, gated `sam3.pt`, 191-frame clip): a
  mid-clip seed (frame 60) with `direction="both"` returns a contiguous `0..190` with a mask on
  every frame (forward 60→190 and backward 60→0 both work). Findings that reshaped the design:
  (a) SAM 3 video's encoder is `torch.compile`'d and **crashes on Windows without Triton** →
  TorchDynamo disabled (eager; `suppress_errors` is not enough); (b) the streaming API is
  frame-0-forward-only and needs a *video* source → arbitrary seed + backward via the temp-video
  slices above; (c) it writes a `runs/` dir + per-frame console spam → `save=False`/`verbose=False`.
- ⚠️ **Per-frame confidence is a constant 1.0** (VERIFIED): SAM 3 video exposes `res.boxes.conf`
  but always at 1.0, so the confident/uncertain threshold is effectively **decorative** — object
  *absence* is signalled by an empty mask (→ `None`), not a low score. `_frame_result` still reads
  conf (future-proof) but near-everything commits as confident; the threshold UI is kept as a
  harmless safety valve and the uncertain→review path rarely fires for SAM 3 tracks.
- ⚠️ **Long-video memory**: `_track_blocking` decodes the whole clip into **host RAM** for the
  track's duration (~`n·H·W·3` bytes) plus a per-run temp `.avi` on disk (2× for `both`). This
  host-RAM frame buffer is one clip-length growth path; the OTHER is **VRAM**: the ~7 GB measured on
  the 191-frame run is mostly model weights (clip-length-independent), but SAM-lineage video
  predictors also accumulate a per-object memory bank across the propagated frames, so a run's VRAM
  activations rise with the number of frames it walks (`del predictor` resets it only between the
  forward and backward runs). Bounded by clip length — fine for typical annotation clips;
  a very long clip is the documented limit, and a frame-window/chunked read (avoiding the full
  RAM->disk->RAM round-trip) is the follow-up. **Revisit if a future ultralytics exposes video
  `init_state` with an in-memory tensor source** — that would drop the temp-video re-mux entirely;
  this is a version-pinned workaround, not the intended shape.

---

## Decisions Under Consideration

### Consider Relative Paths with Image Copying

**Status**: Resolved by ADR-033 (#42) — implemented as dual absolute+relative paths
*without* forced image copying (the copy-into-`images/` prompt already existed), so the
"disk duplication" con below does not apply.

**Proposal**: Copy images to project folder, store relative paths

**Pros**:
- Portable projects
- Self-contained

**Cons**:
- Disk space duplication
- Slow for large image sets
- Export already copies images

---

## ADR-041: Headless CLI as a Separate Entry Point Behind a Qt-Free Boundary

**Status**: Accepted (issue #76)

**Context**: Everything the app could do required a human, a screen and a mouse. There was no
way to regenerate a training dataset in a build script, convert annotation formats without
opening the GUI, fail a CI job because someone committed self-intersecting polygons, or run a
model over a folder overnight. For an ML engineer, an annotation tool that cannot be scripted is
a tool that has to be baby-sat.

**Decision**:

- **A separate console script**, `sreeni-cli`, not a `--headless` flag on the GUI entry point.
  `main.py` imports torch eagerly *before* constructing the `QApplication` to work around a
  Windows DLL conflict (ADR-017). A flag would inherit that whole startup path, so `validate`
  would load torch and require a display on a CI runner.
- **Four commands**: `export`, `convert`, `validate`, `predict`. `train` is deliberately out of
  scope — it needs a GPU, a progress UI and a stop button, none of which belong in a build step.
- **Exit codes are the contract**: 0 success, 1 usage/read/operation error, 2 `validate` findings
  at or above `--fail-on`. Progress narration to stderr, machine-readable output to stdout.
  `--fail-on` is inclusive-upward (`warning` also fails on errors), so a project can tighten its
  gate over time.
- **Read-only by construction**: `core/project_io.py` has no write path at all. Autosave and
  recovery exist to protect interactive editing (ADR-005/032); a build script silently rewriting
  the file it was asked to read is exactly the surprise a CI gate must not spring.
- **Lazy heavy imports per command**: only `predict` imports torch and Ultralytics.

**The architectural work was the separation, not the argument parsing.** Two Qt dependencies sat
in the shared layer, each sufficient on its own to make headless operation impossible:

| Dependency | Why it was there | Replacement |
|---|---|---|
| `QImage` in `io/export_formats.py` | reading a file's dimensions, nothing else | `core/image_size.image_dimensions` (Pillow header read — Qt-free and faster) |
| `QMessageBox` in `io/import_formats.py` | prompting when images and labels do not line up | a `confirm` callback; the GUI supplies the prompt (ADR-031) |

**Alternatives considered**:

- *A `--headless` flag on the existing entry point*: rejected for the ADR-017 startup path above.
- *A thin CLI calling the controllers with a hidden QApplication*: would require a display on
  every CI runner and drag the whole widget tree into a format conversion.
- *Duplicating the export logic for the CLI*: two implementations of a format drift, and the
  divergence surfaces as a dataset that trains differently depending on how it was produced.

**Consequences**:

- The Qt-free boundary is now load-bearing and **enforced by a subprocess test**
  (`tests/integration/test_cli.py`) over `cli/`, `cli.commands`, both `io/` modules,
  `core/project_io` and `core/annotation_qc`. The subprocess matters: the test session has
  already imported PyQt6, so an in-process `sys.modules` check would pass regardless.
- An accidental `from PyQt6 ...` in any shared module re-breaks headless operation, and would
  work perfectly on the machine of whoever added it. The test is the only thing that catches it.
- **Documented limits**: the CLI exports what a project already materialised; extracting new
  slices from a stack runs through the Qt-bound `ImageController` and is out of scope (the count
  skipped is reported). An unresolvable image path refuses the export rather than writing a
  partial dataset that looks complete but trains on fewer images than the user believes.
- `docs/07_deployment_view.md` exists now because there are two entry points with materially
  different runtime requirements.

---

## ADR-042: One Training Dialog, with the Task Derived Rather Than Asked

**Status**: Accepted (issue #73; builds on ADR-028 LR schedule, ADR-029 pose)

**Context**: Training a YOLO model *and actually using it* took six menu navigations and about
ten dialogs, including a step in the middle where the user reloaded the model they had just
produced. SAM fine-tuning was a separate top-level menu with four more actions, one of which was
a manual "Refresh Model Selector". Eleven menu actions for what is conceptually one operation —
and dataset preparation, YAML handling and saving are mechanics, not decisions anyone wanted to
make.

**Decision**:

- **One `Train Model…` action, one dialog**, covering both YOLO training and SAM 2 fine-tuning.
  Preparation, YAML writing, model loading, saving and selector refresh happen implicitly.
- **The task is derived from the annotations, not asked**: boxes only means detect, polygons mean
  segment, keypoints mean pose. Asking is asking the user to restate their own data.
- **`core/task_inference.py` is a module, not a dialog method.** `train_model` already infers the
  task a second time from the prepared dataset YAML (`kpt_shape` present means pose) and raises
  pre-flight if the loaded model's `.task` disagrees. A dialog announcing one task while the
  trainer decides another would be a bug by construction, so both derive from one place.
- **Pre-flight refusals before the run**: mixed-K pose and pose-plus-non-pose projects cannot be
  expressed in YOLO-pose's single dataset-global `kpt_shape`, and an *annotated* stack or video
  whose slices were never materialised has no pixels to export. Both previously surfaced late —
  the first deep inside Ultralytics, the second during dataset preparation.
  **Amended**: the second refusal originally blocked every stack and video outright, inherited
  from a note that predated slice-aware export. It is wrong: a video's `image_slices[base]` is an
  ordinary `LazySliceList` and the exporters resolve slice pixels through it (#45/#47), so
  annotated frames train like any other image. Refusing them rejected valid datasets — and made
  SAM 3 tracking (#51), whose entire purpose is generating training data from video, a dead end.
- **Advanced options collapsed, not removed.** ADR-028 deliberately kept `lr0`/`cos_lr`/
  `patience` off the main surface; they stay reachable.
- **Advanced menu entries demoted, not deleted.** Preparing a dataset for external use and
  training from an externally-prepared folder are real workflows for someone who already has one.
  "Refresh Fine-Tuned Model List" *is* deleted: a manual refresh action existing at all was the
  symptom this set out to remove (issue #74 makes registration automatic).

**Alternatives considered**:

- *A wizard*: more clicks, not fewer, for a form that fits on one screen.
- *Keeping two dialogs and just tidying the menus*: the duplicate mechanics (prepare, save,
  reload) were the actual cost; the menu depth was a symptom.
- *Asking for the task*: every project already answers it, and a user who picks wrongly discovers
  it as an opaque Ultralytics failure.

**Consequences**:

- The trainers (`YOLOTrainer`, `SAMFineTuner`) are untouched — this is UI and orchestration only,
  so the GPU gate, busy guard, progress/stop dialog and automatic MLflow configuration (ADR-027)
  all keep working unchanged.
- The base model must be loaded *before* dataset preparation, so `train_model`'s `.task`-vs-YAML
  pre-flight still runs. Reordering those two steps would silently disable the check that turns
  an opaque failure into an actionable message.
- Irrelevant fields hide rather than grey out when the type changes: a permanently-disabled
  control invites the user to wonder what would enable it.

---

## ADR-043: A Gated Shortcut Registry Instead of More Top-Level Event Filters

**Status**: Accepted (issue #65; fulfils the ADR-015 follow-up)

**Context**: The app had exactly three global shortcuts (F2, Undo, Redo). Every class switch cost
a trip to the class list with the mouse and every tool switch a trip to the sidebar — in a dense
annotation session, the single largest source of wasted motion. Digits 1-9 and every letter were
unclaimed.

**Decision**: Bare-key bindings go through a **gated application-wide event filter**, not
`QShortcut`.

- A `QShortcut` on `3` with `ApplicationShortcut` context swallows that keystroke in **every**
  `QLineEdit` in the app: renaming a class to "Layer 3" or typing a DINO phrase would silently
  drop characters. An event filter is the only mechanism that can be *conditional* on where the
  focus currently is. `QShortcut` remains right for modified keys (Ctrl+Z, F2) that no text field
  wants.
- `ShortcutEventFilter` is a **registry** — `{(key, modifiers): callable}` plus a list of gate
  predicates that must all pass — rather than another bespoke filter. This is what ADR-015's
  follow-up note asked for: *"Future review modes should share filter or layer via strategy
  registry, not install multiple top-level filters."*
- The two gate predicates live in `ui/input_gates.py` and are **shared with
  `DINOReviewEventFilter`**, which previously duplicated them inline. Two filters drifting apart
  on a *safety* predicate is precisely the failure that note warned about. The extraction also
  widened the DINO filter's notion of "typing" to include spin boxes and editable combos.
- `widget_is_text_entry` is split from `focus_is_text_entry` so the classification is testable on
  its own: which widget holds focus is a property of the windowing system and unreliable under
  the offscreen platform, whereas "is a QSpinBox a text entry" is a decision this module owns.
- Tool bindings call `ImageAnnotator.activate_tool` and nothing else — never `current_tool` or
  the SAM flags directly — so a SAM tool still cannot end up active alongside a manual one. A
  test asserts the binding does not write `current_tool`.

**Alternatives considered**:

- *`QShortcut` for everything*: breaks text entry, as above.
- *`keyPressEvent` on the main window*: `QTableWidget` and other focusable children consume keys
  before they bubble — the original reason `ui/shortcuts.py` uses `ApplicationShortcut` at all.
- *A second bespoke top-level filter*: the thing ADR-015 explicitly warned against.
- *A modifier-based second bank for classes 10+*: two-key class selection is slower than
  clicking, so it would add surface without adding speed. Digits address the first nine; the rest
  stay mouse-reachable.

**Consequences**:

- Adding a global-key feature now means registering a binding, not installing a filter. Ctrl+C /
  Ctrl+V for the annotation clipboard (issue #66) joined this way, and inherit the same gating —
  Ctrl+C in a rename field still copies text.
- A binding that returns `False` does not consume the event, which is what makes an out-of-range
  digit a silent no-op rather than an error.
- Discoverability is a paint-pass concern: `ClassShortcutDelegate` draws the 1-9 badge rather
  than appending it to the item text, because **the item text IS the class name** across the app
  (`findItems(MatchExactly)`, the `Temp-` prefix checks, `text()[5:]` on accept, rename).
  Decorating it would break all of those at runtime only.
- **Accepted trade-off**: the bare letter bindings shadow `QListWidget`'s built-in type-ahead
  search in the class, image and slice lists — neither is a text-entry widget, so the gate
  correctly passes and `P` activates the polygon tool rather than jumping to "platelet". Gating
  letters on list focus was considered and rejected: switching tools right after clicking an
  image is the *common* case, and breaking it would defeat the feature. Digits are unaffected in
  practice (type-ahead on a numeric prefix is rare). Revisit if type-ahead turns out to be load-
  bearing for anyone.

## ADR-044: The Train/Val Split Key Is the Group, Not the Image Name

**Status**: Accepted (issue #81; the decision came out of the #80 curation discussion)

**Context**: `assign_train_val` partitioned train/val by a stable MD5 of the image *name*. That is
correct exactly when every name is an independent observation, and in this app it routinely is
not. A multi-dimensional stack contributes one name per slice (`stack_T1_Z5_C1`) and a video one
per frame (`video_F00042`), and consecutive frames are near-identical. A name-keyed split
therefore scatters the frames of one recording across both sides **by construction** — the model
is validated on data it effectively trained on.

Nothing fails. The dataset exports, training runs, and the validation metrics come back *better*
— the more redundant the data, the better they look. That is the worst available failure mode for
a number people use to decide whether a model is ready.

The function was the single choke point for all three training paths: `export_yolo_v4`,
`export_yolo_v5plus` (and therefore `prepare_dataset` → in-app YOLO training) and
`training/sam_dataset.py::split_groups` (SAM fine-tuning — where val loss also drives early
stopping, so a leaky split does not merely misreport, it changes when the run stops).

**Decision**: The split key is a **group**, and a group never straddles the split.

- Grouping is derived from **structure, not from a model**. `image_slices` is already keyed by the
  ext-stripped base name, so `{slice_name: base}` is exact and needs no parsing and no pixel work.
  Names it does not cover fall back to a name-prefix heuristic; anything left is its own group.
- **Always on, no opt-in.** The exporters derive the grouping themselves, so no caller has to
  remember to ask for it. The headless CLI is safe for a different reason and it is worth being
  precise about which: it cannot resolve slice or frame pixels at all, so `_is_exportable` drops
  those names before the split sees them and every surviving name is a file on disk, hence its own
  group. There is no group larger than one image there, so there is nothing to leak and nothing to
  warn about — the CLI's existing `note:` about unexported slices is the honest signal. An earlier
  revision of this change shipped a CLI warning branch that could not fire.
- Embedding clusters from the curation feature (#72) could **refine** the grouping — union them
  into the derived groups so two near-identical images from different files also stay on one side
  — but are never required: the worst case, a 200-frame video, is fixed with no model, no GPU and
  no curation run. None of that is in this change, not even the seam. `similarity.cluster` is
  pure-Python all-pairs, so calling it during an export would freeze the GUI for minutes on a few
  hundred images; the merge helper and the parameter to feed it land in #82 together with the
  vectorised clusterer and an actual caller.
- The exporters filter the split input to names they will actually **write** (`_is_exportable`).
  A name the export loop skips must not consume a slot in the train/val budget: once whole groups
  move together, a video's worth of unwritable frames takes the entire train side with it. That
  is not hypothetical — headlessly, where no slice collection is loaded, an earlier revision of
  this change produced an empty `images/train` with `data.yaml` still pointing at it. As a side
  effect the requested percentage now describes what lands on disk rather than what was counted.
- New Qt-free `core/dataset_split.py`. `assign_train_val` moved there and stays **re-exported**
  from `io/export_formats.py`, which is where `training/sam_dataset.py` and the split tests import
  it from.
- **Degenerate case: warn, do not silently disable validation.** When everything belongs to one
  group — a project that is a single video — no honest split exists. `plan_split` returns a
  `fell_back` flag, uses the per-name split, and the UI states plainly that the validation numbers
  will be optimistic.

**Alternatives considered**:

- *An empty val set in the degenerate case*: the truthful answer, and rejected. The trainer skips
  the validation pass and early stopping when val is empty (ADR-028), so the user silently loses
  two features and gets no explanation — that reads as a regression, not as information.
- *Opt-in checkbox, default off*: no behavioural change for anyone, and the leak stays the default
  for everyone who never finds the checkbox. The bug would remain in practice.
- *Grouping from embedding clusters only*: conceptually cleaner, practically worse. It requires a
  GPU curation run before every export, and without one there is no protection at all.
- *Splitting a group proportionally instead of whole*: defeats the purpose. A partially split
  group leaks exactly as much as an unsplit one.

**Consequences**:

- The requested val percentage becomes a **target rather than a guarantee**, and the honest bound
  is narrow: *no single group added, dropped, or swapped for another would land closer to the
  target*. Choosing the best subset is subset-sum, so `_split_by_group` fills with groups that
  fit, allows one large group to replace the selection when that lands nearer, then hill-climbs on
  single add/drop/swap moves until none improves. The climb enumerates one representative per
  distinct group *size*, since same-size groups are interchangeable for hitting a count — without
  that, the ungrouped path (one group per image) would compare millions of identical singletons.
  On a project of one 100-frame video plus one photo, a requested 20 % delivers a single image:
  that is the closer of the two available answers, not a defect. **Group cohesion is not
  sufficient test coverage here** — a selection can keep every group whole and still deliver 1 %
  for a requested 20 %, so the delivered *size* needs its own property test, as do the bounds that
  keep both sides non-empty, and so does every boolean filter feeding the budget.
- `groups=None` is bit-for-bit the historical per-name split (every name its own group), so
  the existing `test_yolo_split.py` tests are unaffected.
- The name-prefix fallback deliberately errs toward **over-grouping**: a stack literally named
  `run_T1` yields base `run`, merging it with `run_T2`. That costs some split granularity; the
  opposite error would reopen the leak this ADR exists to close. The suffix is restricted to the
  letters `DimensionDialog` actually assigns (T/Z/C/S) plus F for video, because matching any
  `[A-Z]\d+` collapsed a whole 96-well plate into one group. Ambiguity remains where the name is
  genuinely ambiguous — `Plate1_C1_Z1` cannot be told apart from a stack `Plate1` with a channel
  and a Z dimension, so wells in rows C, F, S, T and Z still merge. Nothing in a filename can
  resolve that, which is the argument for the `image_slices` mapping being the primary source and
  the heuristic only the fallback for paths that have none.
- **Known gap:** a folder of *extracted* video frames on disk (`clip_F00001.png` as real files)
  does not group. The dot means "a file, therefore an independent observation", and that is the
  discriminator both the exporters and `_slice_base` rely on. Grouping ext-stripped filenames too
  would close it, at the cost of collapsing legitimately independent photographs that happen to be
  named `sample_T1.png` / `sample_T2.png` into one group — a false alarm on a flat dataset,
  reported rather than silent, but disruptive. Left open deliberately, and recorded here rather
  than discovered later: this is the one remaining path where the leak is still silent.
  `dialogs/dataset_splitter.py` — the standalone folder-splitting tool, which is not one of the
  three choke points above — walks straight into it and is also unseeded, so its splits are not
  even reproducible. Tracked as **#85**; the app is **not** uniformly group-aware, and that tool
  is where it is not.
  - One exception, deliberately inconsistent with the paragraph above:
    `build_groups_from_folder` **does** ext-strip, because a prepared SAM dataset is written by
    `export_sam_dataset` from slice names and is therefore known to contain them. The cost is the
    one just described — `sample_T1.png` and `sample_T2.png` in such a folder merge — and it is
    accepted there because the alternative was no grouping at all on that path. `_is_exportable`
    carries a second, narrower version of the same gap: it cannot decode pixels, so a slice that
    is *indexed* but turns out to be undecodable passes the filter and consumes a split slot the
    export never fills. Same shape as the empty-`images/val` bug, much smaller blast radius, and
    unavoidable without decoding during planning. The two SAM entry
    points consequently group the same data slightly differently.
- `SAMFineTuner.train` calls `split_groups` on a worker thread with no access to `image_slices`,
  so it uses the prefix fallback — which covers every name the app itself produces.
- The wording lives in `core/dataset_split.split_warning`, **not** on the controller. It began
  next to the dialog, in a module importing Qt at top level, which is what forced the CLI to
  hand-roll a subset of it — the same duplication that had already let the split preview and the
  export drift apart. `io_controller` is now only the QMessageBox shell. (The CLI ended up needing
  no warning at all, per the point above, but keeping pure text out of a Qt module is right
  regardless.)
- The warning offers **Cancel**, and all three GUI call sites honour it: `prompt_validation_split`
  (YOLO export menu and Prepare YOLO Dataset) returns to its input dialog,
  `TrainingController.run_yolo` and `SAMTrainController._launch` abandon the run. A warning saying
  the validation numbers cannot be trusted, with only an OK button, trains exactly the
  click-through reflex it exists to prevent — and its advice ("choose a different percentage") had
  nothing to act on. `run_yolo` is the one that must not be missed: the unified Train dialog
  carries its own split slider and never passes through the prompt, so it would otherwise be the
  app's main training path with no signal at all.
- It covers a second case besides the degenerate grouping: a split that leaves **training with a
  single group**. Holding out a percentage by image count can route every small group to
  validation when one group dominates, which is optimal by the count the split aims at and useless
  as a dataset — and silent, because the grouping technically succeeded.
- The preview behind the warning and the export itself must be computed over the **same** set of
  names, which is why both go through `exportable_annotated_names`. They were computed separately
  once, and a preview that counted an unopened video's frames stayed quiet while the export fell
  back to the per-name split.
- `training/sam_dataset.py::build_groups_from_folder` normalises `SampleGroup.name` to the
  ext-stripped basename. The manifest stores `images/clip_F00042.png`, and that dot made every
  frame its own group — so "Fine-Tune SAM from Dataset Folder" got no grouping at all while the
  project path was correctly grouped. Normalising at the producer keeps every consumer of `name`
  seeing one shape.
