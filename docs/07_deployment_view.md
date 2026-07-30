# 7. Deployment View

This chapter was missing until issue #76; the headless CLI is what made it
necessary, because the app now has **two** entry points with materially
different runtime requirements.

## 7.1 Entry points

| Console script | Module | Needs a display | Needs torch |
|---|---|---|---|
| `digitalsreeni-image-annotator` | `main:main` | yes | yes (eagerly, see below) |
| `sreeni` | `main:main` | yes | yes |
| `sreeni-cli` | `cli:main` | **no** | only for `predict` |

The CLI is a **separate console script rather than a flag on the GUI entry**.
That is not cosmetic: `main.py` imports torch eagerly *before* constructing the
`QApplication`, to work around a Windows DLL conflict (ADR-017). A `--headless`
flag on that entry point would inherit the whole startup path, so `validate`
would load torch and require a display on a CI runner.

## 7.2 The Qt-free boundary

```
   cli/                     GUI
    │                        │
    ├── core/project_io.py   ├── controllers/  ── Qt
    ├── core/annotation_qc.py│   dialogs/     ── Qt
    ├── core/task_inference  │   widgets/      ── Qt
    ├── core/disagreement    │
    ├── core/similarity      │
    ├── core/model_sidecar   │
    ├── core/image_size      │
    └── io/  ────────────────┘   (shared, Qt-free)
```

Everything on the left of that boundary is imported by both, and **must not
import Qt at module level**. `tests/integration/test_cli.py` enforces this in a
subprocess for `cli/`, `cli.commands`, `io/export_formats`, `io/import_formats`,
`core/project_io` and `core/annotation_qc`. The subprocess matters: the test
session has already imported PyQt6, so an in-process `sys.modules` check would
pass regardless of what those modules do.

Two Qt dependencies were removed to reach this boundary (issue #76):

- `io/export_formats.py` imported `QImage` for one purpose — reading a file's
  dimensions. Replaced by `core/image_size.image_dimensions`, which reads the
  header via Pillow (already a dependency) without decoding pixels.
- `io/import_formats.py` raised a `QMessageBox` from inside `import_yolo_v4`
  when images and labels did not line up — a UI concern in a core module
  (ADR-031). Replaced by a `confirm` callback; the GUI supplies the prompt, and
  a non-interactive caller defaults to proceeding.

## 7.3 CLI commands

```
sreeni-cli export   --project data.iap --format coco --out ./dataset [--val-split 20]
sreeni-cli convert  --in ./coco.json --from coco --to yolov5 --out ./yolo [--images DIR]
sreeni-cli validate --project data.iap [--json report.json] [--fail-on error|warning|info|never]
sreeni-cli predict  --model best.pt --images ./raw --out ./preds [--format coco|yolov5] [--conf 0.25]
```

`train` is deliberately out of scope.

**Exit codes** — the contract that makes `validate` a CI gate:

| Code | Meaning |
|---|---|
| 0 | success |
| 1 | usage error, unreadable input, or a failed operation |
| 2 | `validate` found issues at or above `--fail-on` |

Progress narration goes to **stderr** and machine-readable output to **stdout**,
so `sreeni-cli validate --project p.iap \| jq .total` works while the per-finding
lines stay visible.

`--fail-on` is inclusive-upward: `warning` also fails on errors, `info` fails on
everything. That is what lets a project tighten its gate over time.

## 7.4 Read-only guarantee

The CLI opens projects **read-only**. `core/project_io.py` has no write path at
all — not a discipline, a structural fact. Autosave and recovery snapshots exist
to protect interactive editing (ADR-005/032); a build script silently rewriting
the file it was asked to read is exactly the surprise a CI gate must not spring.
Tests assert the project's bytes *and* mtime are unchanged after `validate` and
`export`, and that no new file appears beside it.

## 7.5 Documented limits

- **Multi-dimensional images and video.** The CLI exports what a project has
  already materialised. Extracting new slices from a stack runs through
  `ImageController`, which is Qt-bound, so it is out of scope headlessly.
  `export` reports the number of slices it skipped rather than omitting them
  silently.
- **Unresolvable image paths are an error, not a warning.** `export` refuses
  rather than writing a partial dataset: one that looks complete but trains on
  fewer images than the user believes is worse than no dataset. Path resolution
  is relative-first (ADR-033), which is what makes a project portable to a CI
  runner in the first place.
- **`predict` needs torch and Ultralytics**; the other three commands do not and
  import neither.

## 7.6 Installation

```bash
pip install -e .            # runtime
pip install -e ".[dev]"     # + pytest, pytest-qt, ruff, mypy
```

On Linux the GUI needs `libxcb-cursor0` (Qt 6 requires it; it was optional under
Qt 5). The CLI needs no such package — that is the point.
