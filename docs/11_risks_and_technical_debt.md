# Risks and Technical Debt

## Technical Risks

### Linux Platform Support

**Risk Level**: Medium

**Description**: Application has limited testing on Linux, known XCB plugin issues

**Impact**:
- May not work correctly on Linux systems
- Potential crashes or rendering issues
- Limits user base

**Mitigation**:
- Environment variable workaround in main.py (removes `QT_QPA_PLATFORM_PLUGIN_PATH`)
- Document Windows/macOS as primary platforms
- Community testing and feedback

**Future Action**: Dedicated Linux testing and fixes

---

### SAM 2 Large Model Crashes

**Risk Level**: Medium

**Description**: SAM 2 large model can crash application on systems with limited RAM

**Impact**:
- Application termination
- Loss of unsaved work
- Poor user experience

**Mitigation**:
- Documentation recommends tiny/small models
- UI warns about large model
- Autosave reduces data loss
- Out-of-memory on model load now shows an actionable "pick a smaller model"
  dialog instead of a generic error (`core/torch_utils._is_oom` +
  `SAMController.change_sam_model`, issue #34)

**Future Action**:
- Add RAM detection and warning
- Catch OOM exceptions gracefully

---

### Project File Portability

**Status**: Resolved (#42, ADR-033)

**Risk Level**: Low-Medium (historical)

**Description**: ~~Projects store absolute paths, not portable between machines.~~
`.iap` now stores portable `image_paths_rel` (POSIX separators) alongside the absolutes;
`resolve_image_path()` resolves relative-first, so a moved or shared project folder opens
without a missing-images prompt. v1 projects still resolve via the `images/` convention.

**Original impact** (pre-#42):
- Cannot share projects easily
- Moving images breaks projects
- Collaboration difficult

**Mitigation**:
- Export functions copy images
- Users can manually update paths in JSON

**Future Action**: Consider relative paths or image embedding option

---

### Large Image Memory Usage

**Risk Level**: Medium

**Description**: Loading very large images or many slices can exhaust memory

**Impact**:
- Application slowdown
- Potential crashes
- Poor performance

**Mitigation**:
- Slice-by-slice loading for multi-dimensional images
- **Lazy slice QImage materialisation with a bounded LRU (ADR-036, #45)** —
  `create_slices` no longer builds every slice's `QImage` up front; slice
  QImages decode on demand and at most `slice_cache.LRU_CAPACITY` (8) are held
  live process-wide. Removes the dominant all-QImages-in-RAM cost and the
  create-time peak.
- Image downsampling for display (future)

**Current Limitation**: The decoded source ndarray is still retained per open
stack (Strategy A). Full array-free reading (memmap/zarr per-slice, or CZI
lazy read) is a documented follow-up; the live-QImage count is now bounded.

---

## Technical Debt

### Low Test Coverage of Interactive Paths

**Status**: ✅ Largely resolved for the canvas layer (issue #77)

**Debt Level**: Low (was Medium)

**Description (historical)**: The canvas event flow — mouse event → tool
handler → signal emission → controller slot — had no automated coverage. The
per-tool handlers sat at 22–27 % by line and `canvas_renderer.py` at 49 %,
which meant every canvas refactor leaned on a manual QA checklist.

**Resolution**: Issue #77 added three layers of coverage, built on the shared
doubles in `tests/canvas_fixtures.py` (`FakeCanvasContext`, `FakeMouseEvent`,
`RecordingPainter`):

1. `tests/unit/test_tool_handlers.py` — every `ToolHandler` subclass driven
   through press / move / release / Enter / Escape and its `paint_overlay`,
   asserting the **emitted signal and payload** rather than internal state.
   Includes the right-button (occluded keypoint) path that the left-only press
   dispatch would otherwise hide (ADR-029).
2. `tests/ui/test_canvas_gestures.py` — real `qtbot` mouse events through
   `mousePressEvent`, so the dispatch *priority order* is covered as well as
   the gesture logic: handle resize anchoring, drag-gated move, rubber-band
   selection, double-click into vertex-edit mode, and the ADR-026 rule that an
   Esc-aborted gesture leaves no history entry.
3. `tests/unit/test_canvas_renderer_contract.py` — `CanvasRenderer` against a
   recording painter, pinning draw order (selection overlay last, temp
   annotations on top) and class-visibility filtering. This is the harness
   onion-skinning (#67) inserts a layer into.

Plus `tests/unit/test_coordinate_conversion.py` for the screen↔image funnel
every gesture passes through.

**Measured effect** (full suite, `--cov`):

| Module | Before | After |
|--------|--------|-------|
| `widgets/tools/eraser_tool.py` | 22 % | 75 % |
| `widgets/tools/paint_tool.py` | 26 % | 79 % |
| `widgets/tools/polygon_tool.py` | 26 % | 76 % |
| `widgets/tools/rectangle_tool.py` | 27 % | 85 % |
| `widgets/tools/keypoint_tool.py` | 65 % | 92 % |
| `widgets/canvas_renderer.py` | 49 % | 69 % |
| `widgets/image_label.py` | 59 % | 69 % |

A `--cov-fail-under` floor is now configured in `pytest.ini` so the number
cannot quietly slide back. The floor is set at the level actually reached, not
an aspirational one — a gate that fails on day one gets disabled on day two.

**Remaining gap**: the SAM/DINO/YOLO inference paths are still exercised only
via the smoke boot and mocked controller tests, never under real model loads
(those would slow CI prohibitively). That is a deliberate limit, not an
oversight.

---

### Limited Coverage — Inline Imports Not Caught by Module Tests

**Debt Level**: Medium

**Description**: Smoke tests verify modules import cleanly at top-level, but inline `from .module` imports inside function bodies are deferred and only fail when the function is called. Phase 1 modular refactoring moved 25 modules; four stale inline imports (`from .dino_utils`, `.annotation_statistics`, `.project_details`, `.project_search`) were missed and only surfaced in manual QA.

**Impact**:
- Subpackage refactor PRs require functional QA paths (not just module import CI) to verify inline imports
- Silent regressions until user clicks the specific button/dialog that triggers the stale import

**Mitigation**:
- Added AST-based static smoke test (ADR-016) that parses `annotator_window.py` and asserts every bare relative import resolves to an existing module in the package root
- The test now catches inline import drift in CI before merge

**Future Action**: Extend the AST check to any other file that uses inline deferred imports (currently only `annotator_window.py` has them).

---

### Inconsistent Error Handling

**Status**: ✅ Resolved with a written convention (issue #34)

**Debt Level**: Medium (historical)

**Resolution**: A single error-handling convention now governs the codebase —
core/inference/io/training raise; controllers/dialogs catch, `logger.exception`,
and surface a `QMessageBox`; catch the narrowest type; never `pass` silently;
bare `except:` banned. Seven silent `except: pass` sites were fixed and the one
bare `except:` removed. See ADR-031 and the Error-Handling Convention in
[docs/08](08_crosscutting_concepts.md#error-handling-convention-issue-34).

**Description (historical)**: Mix of exceptions, return values, and UI warnings

**Examples**:
```python
# Some functions raise exceptions
raise ValueError("Invalid dimension")

# Some show message boxes
QMessageBox.warning(self, "Error", "...")

# Some return None
return None
```

**Impact**:
- Inconsistent user experience
- Hard to predict error behavior
- Difficult to add global error handling

**Effort to Resolve**: Medium (weeks)

**Priority**: Low

**Plan**: Standardize on exception-based approach with top-level handler

---

### Print Statements for Logging

**Status**: ✅ Resolved (issue #33)

**Description**: Historically used `print()` instead of a logging framework.
All ~307 `print()` calls and 12 `traceback.print_exc()` sites in `src/` were
migrated to the stdlib `logging` module: one package-level logger tree rooted
at `digitalsreeni_image_annotator`, configured once in
`core/logging_config.py`, with a `--debug` / `IMAGE_ANNOTATOR_DEBUG` level
switch. `print()` is now banned in `src/` (ADR-030). See the
"Logging and Debug Output" section in
[docs/08](08_crosscutting_concepts.md#logging-and-debug-output).

**Plan**: Replace with `logging` module

---

### Tight Coupling Between ImageAnnotator and ImageLabel — Resolved (Phase 6)

**Status**: Resolved. `ImageLabel.main_window` and `set_main_window()`
were removed; every write path is now a `pyqtSignal` emission and every
read goes through a narrow `CanvasContext` accessor.

**Pattern**: see `widgets/canvas_context.py` and
`ImageAnnotator._connect_image_label_signals`. ImageLabel emits ~20
signals (annotation lifecycle, SAM, class, tool/UI state, navigation);
the orchestrator wires each to the matching controller slot.

**ADR**: see ADR-018 in `09_architecture_decisions.md`.

---

### Duplicate Code in Export Functions

**Debt Level**: Low

**Description**: Export formats share similar code (image copying, directory creation)

**Impact**:
- Bug fixes must be applied multiple times
- Inconsistent behavior across formats
- More maintenance

**Effort to Resolve**: Low (extract common functions)

**Priority**: Low

---

### No Type Hints

**Status**: Partially resolved — the Qt-free core is typed and checked (#78)

**Debt Level**: Low for `core/`, unchanged elsewhere

**Description (historical)**: The codebase was essentially untyped, with no
configuration and no checking step. The cost showed up most in the structures
carrying the most meaning: an annotation is a dict whose valid shapes were
documented in prose, and a pose instance is distinguished from a polygon by the
**absence** of a key (ADR-029) — a rule only a comment protected.

**What is now covered**

`core/annotation_types.py` defines the annotation shapes as `TypedDict`s:
`PolygonAnnotation`, `BBoxAnnotation`, `PoseAnnotation`, plus `KeypointSchema`
and aliases for the recurring shapes (`Polygon`, `BBox`, `Keypoints`,
`AnnotationsByImage`). **`PoseAnnotation` declares no `segmentation` key at
all** — the type definition expresses the discriminator, because declaring one
even as optional would legitimise writing it, and writing it breaks every
existence-only `"segmentation" in ann` check. `is_pose` / `is_polygon` /
`is_bbox_only` express the test once.

Every `TypedDict` is `total=False`, deliberately: the annotation dict
legitimately gains keys at runtime (`segmentation_raw` lazily, ADR-025;
`source` and `track_run` on tracked results, ADR-040; `assigned_class` on
unprompted proposals, #69). Forcing a rigid schema onto genuinely open data
would produce false errors and teach people to ignore the checker.

mypy is configured in `pyproject.toml` with **global `ignore_errors = true` and
a per-module opt-in**. That direction matters: the boundary of what is actually
checked stays visible, whereas checking everything and suppressing failures
would hide it. Currently opted in: `annotation_types`, `annotation_qc`,
`constants`, `disagreement`, `image_size`, `mask_filters`, `model_sidecar`,
`onion`, `project_io`, `similarity`, `task_inference`.

Untyped third-party packages are listed **individually** rather than behind a
global `ignore_missing_imports`, so the list stays visible and shrinks as
upstreams ship stubs. Several of them ship source written for a newer Python
than the project's 3.10 floor, so the override also sets
`follow_imports = "skip"`.

**Deliberately out of scope**: widget internals and dialogs. The PyQt6 stubs are
incomplete, so annotating them produces noise rather than safety — which is
exactly why an `mypy --strict` sweep over the whole tree is a different project.

**The gate is verified to be non-vacuous.**
`tests/unit/test_annotation_types.py` copies the tree, injects a deliberately
wrong return type into an in-scope module, and asserts the real gate fails. A
type-check step that checks nothing is worse than none: it reports success
forever while teaching everyone to trust it.

**Remaining**: `io/`, `utils.py` and the controller signatures are annotated
only where they already were. Extending the opt-in list module by module is the
intended path.

---

### No Linting

**Status**: Resolved (issue #78)

**Description**: The project had no linter at all.

**Resolution**: `ruff` is configured in `pyproject.toml` with a **deliberately
narrow** rule set — `E4`, `E7`, `E9`, `F`. The codebase predates any linter, so a
broad selection would produce hundreds of findings nobody reads and the gate
would be switched off within a week. These rules catch real defects (unused
names, shadowed builtins, ambiguous identifiers, syntax-level errors) rather
than style preferences. It found and fixed 17 pre-existing issues on first run;
the tree is clean.

Run both gates separately from the tests, so a type error is distinguishable
from a test failure:

```bash
python -m ruff check src tests
python -m mypy
pytest
```

---

### Hardcoded UI Strings

**Debt Level**: Low

**Description**: No internationalization (i18n) support

**Impact**:
- Cannot translate to other languages
- Limits international user base

**Effort to Resolve**: Medium (Qt has i18n support)

**Priority**: Very Low (no current demand)

---

## Known Issues

### YOLO Training Needs a Stack's Slices to Be Loaded

**Status**: Resolved for the common case

**Description**: This entry previously read "YOLO training only works with single images, not
TIFF/CZI slices", with "export slices as individual images first" as the workaround. That was
wrong by the time it was written down: the exporters resolve slice pixels through `image_slices`
(#45/#47), so stack slices and video frames export like any other image. The training dialog's
pre-flight nonetheless refused every stack and video, which rejected valid datasets — including
the one SAM 3 tracking (#51) exists to produce.

What remains is narrower: a stack or video contributes **no pixels** until its slices have been
materialised in this session. Project load and `add_images_to_list` both do that eagerly, so the
reachable causes are a cancelled dimension dialog, an unreadable codec, or a moved file. The
dialog blocks only when such a stack **has annotations**, since an unopened but unannotated stack
cannot affect the export at all.

**Priority**: Low (narrow residual case, reported explicitly rather than silently dropped)

---

### Annotation Merge Only Works for Connected Regions

**Status**: Known Limitation

**Description**: Merge tool requires annotations to overlap or touch

**Workaround**: Use paint brush to connect regions first

**Priority**: Low

---

### Keypoint / Pose Constraints (issue #35)

**Status**: Known Limitations (ADR-029)

**Description**:
- The keypoint count **K is locked** once a pose class has instances (changing K
  would corrupt existing instances). Renaming points / editing the skeleton / flip
  stays allowed.
- A schema is **per class** (the COCO rule); all instances of a class share it.
- A point set to *not labelled* (v=0) via "finish early" doesn't render and can't be
  relabelled with a right-click in PR-1 (only v>0 points are hit-testable).
- ~~**Defining a schema on a class that already holds normal (polygon/bbox) annotations
  is unguarded**~~ **— Resolved (#44).** The UI now blocks *new* mixing in both
  directions (schema-on-plain-class; shape/SAM-tool-on-pose-class;
  pose-class-selection-while-a-tool-is-active; DINO skips pose classes); see ADR-029
  Guards. Legacy-mixed classes still load/render/save, and `_pose_export_check` remains
  the export backstop.
- **Forthcoming** (PR-2/PR-3): YOLO-pose export requires a **single `kpt_shape` per
  dataset**, so a project mixing pose classes of different K (or pose + non-pose) can't
  export to YOLO-pose (COCO has no such limit). YOLO-pose *training* stays unsupported
  for multi-dimensional stacks (same constraint as detect/segment training).

**Priority**: Low (documented constraints, not bugs)

---

### SAM Point Mode Requires Manual Confirmation

**Status**: By Design

**Description**: User must press Enter to accept SAM prediction

**Rationale**: Allows user to add more points or reject prediction

**Priority**: N/A (intentional)

---

### Autosave Doesn't Ask for File Location

**Status**: Resolved (#41, ADR-032)

**Description**: ~~Autosave only works after first manual save.~~ Before the first save,
`auto_save()` now writes a silent recovery snapshot (no dialog) that the app offers to
restore on next launch; a real save clears it. New projects are protected from the first
mutation.

---

## Upstream Fork Divergence

**Risk Level**: Medium

**Description**: This is a fork of https://github.com/bnsreenu/digitalsreeni-image-annotator

**Impact**:
- May miss upstream features
- May miss upstream bug fixes
- Merge conflicts on updates

**Mitigation**:
- Document fork-specific changes
- Periodically review upstream
- Consider contributing changes back

**Current Fork-Specific Changes** (derived from the merge history and ADR index):
- PyQt6 migration replacing PyQt5 (ADR-014), with a torch-before-Qt DLL
  load-order guard in `main.py` (ADR-017).
- In-process SAM 2 / Grounding-DINO inference on a `QThread` with a re-entrancy
  guard, replacing the old subprocess workers (ADR-013).
- Grounding-DINO text-prompted detection — single image and batch — with an
  Enter/Escape review-and-accept overlay.
- SAM 2 fine-tuning via a custom loop over the Ultralytics SAM2 module (ADR-021),
  with always-on MLflow experiment tracking (ADR-027).
- YOLO training + prediction for detection, segmentation, and pose (issue #35).
- Keypoint / pose annotation: per-class named schema + skeleton (COCO instance
  model), with COCO-keypoints and YOLO-pose export/import (ADR-029).
- Undo / redo via per-image annotation snapshots (ADR-026).
- Canvas selection unified with the annotations table + handle-based shape
  editing and vertex editing (ADR-022 / 023 / 025), with bounds clamping and
  augmentation clipping (ADR-024).
- Modular architecture: thin `ImageAnnotator` orchestrator + per-responsibility
  controllers + per-tool handlers (ADR-018 / 019).
- Central stdlib `logging` framework (ADR-030) and a written error-handling
  convention (ADR-031).
- A pytest + pytest-qt automated test suite run in CI on 3 OS × Python
  3.10-3.14, superseding the original manual-testing-only decision (ADR-004).

---

## SAM 3 Dependency & Licensing Constraints (Milestone D, ADR-038)

**Risk Level**: Medium (forward-looking; applies once #50/#51 ship)

**Description**: The SAM 3 spike (#49, ADR-038) confirmed SAM 3 is consumable via Ultralytics
(>=8.3.237) but with three material constraints:

- **Non-redistributable, gated weights**: `sam3.pt` (3.45 GB) is under Meta's custom **SAM License**
  (not OSI open-source) and is access-gated on Hugging Face. We must **not** vendor or ship the
  weights; users request access, accept Meta's terms, and download them (same posture as gated DINO
  models). Redistributed weights/derivatives must stay under the SAM License; a patent-retaliation
  clause applies.
- **CPU impractical**: no documented CPU path; 3.45 GB / 473.6M params make CPU inference infeasible.
  SAM 3 is GPU-recommended; the Grounding-DINO two-stage pipeline remains the CPU fallback.
- **Version floor bump**: `ultralytics` floor rises to `>=8.3.237,<9` for #50; a CLIP dependency quirk
  may require `pip install git+https://github.com/ultralytics/CLIP.git`.

**Mitigation**: Lazy-load SAM 3 (ADR-012) so older installs still launch; keep DINO selectable for CPU
users; surface a clear gated-download status message; document the manual weight step. Two D3 facts
(numpy-frame seeding, long-video predictor memory) remain unresolved and are #51's verify-first items.

**Priority**: Tracked in ADR-038; addressed by #50/#51.

---

## Security Considerations

### No Input Validation on JSON Loading

**Risk Level**: Low

**Description**: Project JSON files loaded without strict schema validation

**Impact**:
- Malformed files can crash application
- Potential for malicious project files

**Mitigation**:
- Projects are local files (user-controlled)
- Try-catch around JSON loading

**Priority**: Low (desktop app, local files)

---

### Arbitrary File Paths in Projects

**Risk Level**: Low

**Description**: Project files can reference any file path

**Impact**:
- Could load unintended files
- Path traversal (theoretical)

**Mitigation**:
- Desktop app (user has filesystem access anyway)
- File existence checks before loading

**Priority**: Very Low
