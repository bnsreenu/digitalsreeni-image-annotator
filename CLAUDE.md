# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

DigitalSreeni Image Annotator - PyQt5 desktop app for image annotation with SAM 2 integration and multi-dimensional image support.

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

Python 3.10+ | PyQt5 5.15.11 | Ultralytics 8.3.27 (SAM 2) | NumPy | OpenCV | Shapely

**No automated tests exist** - all testing is manual.

## Documentation

For detailed architecture and design information, see **[docs/arc42/](docs/arc42/)**:

- **[Building Block View](docs/arc42/05_building_block_view.md)** - Components, data model, class responsibilities
- **[Runtime View](docs/arc42/06_runtime_view.md)** - Workflows and key scenarios
- **[Cross-cutting Concepts](docs/arc42/08_crosscutting_concepts.md)** - Coordinate systems, conversions, patterns
- **[Architecture Decisions](docs/arc42/09_architecture_decisions.md)** - Why we made key choices
- **[Glossary](docs/arc42/12_glossary.md)** - Terms, acronyms, data structures

See [docs/arc42/README.md](docs/arc42/README.md) for full documentation index.

## Project Structure

```
src/digitalsreeni_image_annotator/
├── main.py                    # Entry point
├── annotator_window.py        # ImageAnnotator - main window, project state
├── image_label.py             # ImageLabel - display, mouse events, rendering
├── sam_utils.py               # SAMUtils - SAM model management
├── utils.py                   # Utility functions
├── export_formats.py          # COCO, YOLO, Pascal VOC exporters
├── import_formats.py          # COCO, YOLO importers
└── [tool dialogs]             # Standalone utility windows
```

## Key Classes

| Class | File | Responsibility |
|-------|------|----------------|
| `ImageAnnotator` | annotator_window.py | Main window, state (`all_annotations`, `class_mapping`, etc.) |
| `ImageLabel` | image_label.py | Image display, zoom/pan, annotation interaction |
| `SAMUtils` | sam_utils.py | Load SAM models, run inference |

See [Building Block View](docs/arc42/05_building_block_view.md) for detailed class documentation.

## Common Development Tasks

### Adding a New Annotation Tool

1. Add button in `ImageAnnotator.create_tool_section()`
2. Set `image_label.current_tool` on click
3. Handle mouse events in `ImageLabel` (mousePressEvent, mouseMoveEvent)
4. Render in `ImageLabel.paintEvent()`
5. Call `main_window.add_annotation()` to commit

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

```python
# Load model (first use downloads, ~40-400MB)
self.sam_utils.change_sam_model("SAM 2 tiny")

# Run inference
prediction = self.sam_utils.apply_sam_points(
    qimage,
    positive_points=[(x1, y1)],
    negative_points=[(x2, y2)]
)
# Returns: {"segmentation": [...], "score": float}
```

## Important Notes

### Platform Support
- ✅ Windows, macOS fully supported
- ⚠️ Linux has XCB issues, limited testing

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

See [Cross-cutting Concepts](docs/arc42/08_crosscutting_concepts.md#coordinate-systems) for details.

### Multi-dimensional Images
- User assigns dimensions (T, Z, C, H, W) via dialog
- Slices extracted with names like `stack.tif_T0_Z5_C0`
- Each slice annotated independently
- Stored in `image_slices` dict

See [Runtime View](docs/arc42/06_runtime_view.md#multi-dimensional-image-loading) for workflow.

## Development Workflow

1. **Test manually** - No automated tests, run full application to verify
2. **Test on Windows/macOS** if possible
3. **Support dark mode** - Use both stylesheets, check rendering
4. **Consider memory** - Large images and stacks can exhaust RAM
5. **Follow PyQt5 patterns** - Use signals/slots, existing widget styles

## Known Constraints

- No type hints (gradual addition encouraged)
- Print statements instead of logging (acceptable)
- Absolute paths in projects (not portable)
- SAM 2 large crashes on limited RAM
- YOLO training not supported for multi-dimensional images

See [Risks and Technical Debt](docs/arc42/11_risks_and_technical_debt.md) for full list.

## Keyboard Shortcuts

| Global | Action |
|--------|--------|
| Ctrl+N/O/S | New/Open/Save Project |
| F1 | Help |

| Canvas | Action |
|--------|--------|
| Ctrl+Wheel | Zoom |
| Ctrl+Drag | Pan |
| Enter | Finish/Accept |
| Esc | Cancel |
| Up/Down | Navigate slices |
| -/= | Brush size |

## Quick Tips

- SAM models cache in working directory (Ultralytics)
- Recommend SAM 2 tiny/small (avoid large)
- Polygon area uses shoelace formula (utils.py)
- Export formats copy images to output directory
- Dark mode changes annotation colors for visibility
- Snake game is hidden Easter egg 🐍
