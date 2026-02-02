# Building Block View

## Level 1: System Overview

```
┌─────────────────────────────────────────────┐
│   DigitalSreeni Image Annotator            │
│                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐ │
│  │   GUI    │  │  SAM 2   │  │  YOLO    │ │
│  │ (PyQt5)  │  │(Ultraly.)│  │ Trainer  │ │
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
├── annotator_window.py            # ImageAnnotator - main window
├── image_label.py                 # ImageLabel - custom display widget
├── sam_utils.py                   # SAMUtils - SAM model management
└── utils.py                       # Utility functions
```

### ImageAnnotator (annotator_window.py)

**Responsibility**: Main application window and state management

**Key Attributes**:
```python
all_annotations: dict[str, list]    # {filename: [annotation_dicts]}
all_images: list[str]               # List of loaded image filenames
class_mapping: dict[str, int]       # {class_name: class_id}
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

### ImageLabel (image_label.py)

**Responsibility**: Image display and annotation interaction

**Key Attributes**:
```python
current_tool: str                   # Active annotation tool
zoom_factor: float                  # Current zoom level
annotations: dict                   # Displayed annotations
class_colors: dict                  # Class color mapping
temp_paint_mask: np.ndarray         # Temporary paint strokes
sam_positive_points: list           # SAM positive points
sam_negative_points: list           # SAM negative points
```

**Key Methods**:
- `mousePressEvent()`: Handle mouse clicks for annotation
- `mouseMoveEvent()`: Handle mouse dragging
- `paintEvent()`: Render image and annotations
- `zoom_in()`, `zoom_out()`: Zoom controls
- `start_painting()`, `start_erasing()`: Brush tools

### SAMUtils (sam_utils.py)

**Responsibility**: SAM model loading and inference

**Key Attributes**:
```python
sam_models: dict                    # Available SAM model variants
current_sam_model: str              # Currently loaded model
sam_model: SAM                      # Ultralytics SAM instance
```

**Key Methods**:
- `change_sam_model(model_name)`: Load SAM model
- `apply_sam_points(image, positive_points, negative_points)`: Run inference
- `qimage_to_numpy(qimage)`: Convert QImage to numpy array
- `mask_to_polygon(mask)`: Convert SAM mask to polygon contours

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
| `dataset_splitter.py` | Train/val/test split | Stratified splitting, configurable ratios |
| `image_patcher.py` | Create patches | Sliding window with overlap |
| `image_augmenter.py` | Data augmentation | Rotation, flip, brightness, preview |
| `slice_registration.py` | Align slices | Multiple registration algorithms (pystackreg) |
| `stack_interpolator.py` | Z-spacing adjustment | Interpolation methods, memory-efficient |
| `dicom_converter.py` | DICOM to TIFF | Preserve metadata, export to JSON |
| `yolo_trainer.py` | Model training | Train YOLO, load predictions |

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

### Multi-dimensional Image Handling

**Slice Naming Convention**:
```
{filename}_T{t}_Z{z}_C{c}_S{s}
Example: stack.tif_T0_Z5_C0_S0
```

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
    ├── references ──> ImageAnnotator (callbacks)
    └── uses ──> utils (area, bbox calculations)

SAMUtils
    └── depends on ──> ultralytics.SAM

Tool Dialogs
    └── operate independently (standalone windows)
```
