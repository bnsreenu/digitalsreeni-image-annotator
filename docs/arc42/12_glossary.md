# Glossary

## Terms and Definitions

### Annotation
A marked region on an image, either a polygon (segmentation) or rectangle (bounding box), associated with a class label.

### Bounding Box (bbox)
A rectangular annotation defined by `[x, y, width, height]` in COCO format. Stored in annotation as `"bbox"` key.

### Class
A category label for annotations (e.g., "cell", "nucleus", "mitochondria"). Each class has an ID and color.

### COCO Format
Common Objects in Context - a standardized JSON format for object detection and segmentation annotations. Includes images, categories, and annotations with segmentation polygons or bounding boxes.

### CZI File
Carl Zeiss Image file format for multi-dimensional microscopy images. Contains metadata and multi-channel Z-stacks.

### Multi-dimensional Image
An image with more than 2 dimensions, typically from microscopy. Dimensions include T (time), Z (depth), C (channel), S (scene), H (height), W (width).

### Paint Brush Tool
Drawing tool that creates freeform annotations by painting a mask with adjustable brush size. Converted to polygon contours when finished.

### Pascal VOC
Visual Object Classes dataset format. XML-based annotation format primarily for bounding boxes.

### Polygon / Segmentation
A closed shape annotation defined by a list of vertex coordinates `[x1, y1, x2, y2, ...]`. Stored in annotation as `"segmentation"` key.

### Project
A saved workspace containing images, classes, and annotations. Stored as a `.json` file with absolute paths to images.

### SAM / SAM 2
Segment Anything Model - Meta's foundation model for image segmentation. Version 2 (SAM 2) is used in this application.

### SAM Point Mode
Annotation mode where user clicks positive points (inside object) and negative points (outside object) to guide SAM segmentation.

### Semantic Labels
Single-channel image where each pixel value represents the class ID. Used for semantic segmentation training.

### Slice
A 2D image extracted from a multi-dimensional image stack. Named with format `{filename}_T{t}_Z{z}_C{c}_S{s}`.

### Stack
A multi-dimensional image, typically a TIFF or CZI file with multiple 2D slices in Z-dimension (depth).

### TIFF Stack
Multi-page TIFF file containing multiple 2D images, often used for Z-stacks in microscopy.

### YOLO Format
You Only Look Once - object detection format. Uses `.txt` files with normalized coordinates: `class_id x_center y_center width height`.

### Z-Stack
A series of 2D images taken at different focal depths (Z positions), used in microscopy to capture 3D structure.

## Acronyms

| Acronym | Full Term |
|---------|-----------|
| ADR | Architecture Decision Record |
| API | Application Programming Interface |
| bbox | Bounding Box |
| COCO | Common Objects in Context |
| CZI | Carl Zeiss Image |
| DICOM | Digital Imaging and Communications in Medicine |
| GUI | Graphical User Interface |
| JSON | JavaScript Object Notation |
| ML | Machine Learning |
| OOM | Out Of Memory |
| PNG | Portable Network Graphics |
| PyQt | Python bindings for Qt framework |
| RGB | Red Green Blue (color model) |
| SAM | Segment Anything Model |
| TIFF | Tagged Image File Format |
| UI | User Interface |
| VOC | Visual Object Classes |
| XML | eXtensible Markup Language |
| YOLO | You Only Look Once |

## File Extensions

| Extension | Description |
|-----------|-------------|
| `.json` | Project file or COCO annotation file |
| `.tif`, `.tiff` | TIFF image, possibly multi-dimensional stack |
| `.czi` | Carl Zeiss microscopy image |
| `.png`, `.jpg`, `.jpeg` | Standard image formats |
| `.txt` | YOLO annotation file |
| `.xml` | Pascal VOC annotation file |
| `.yaml`, `.yml` | YOLO data configuration file |
| `.pt` | PyTorch model file (SAM weights) |
| `.dcm` | DICOM medical image file |

## Key Classes (Code)

| Class | Module | Description |
|-------|--------|-------------|
| `ImageAnnotator` | annotator_window.py | Main application window (QMainWindow) |
| `ImageLabel` | image_label.py | Custom QLabel for image display and interaction |
| `SAMUtils` | sam_utils.py | SAM model loading and inference |
| `DimensionDialog` | annotator_window.py | Dialog for assigning dimensions to stacks |
| `TrainingThread` | annotator_window.py | Background thread for YOLO training |
| `YOLOTrainer` | yolo_trainer.py | YOLO model training and prediction |

## Data Structure Keys

### Project JSON
- `images`: List of image filenames
- `image_paths`: Dict mapping filename to absolute path
- `classes`: List of class names
- `class_colors`: Dict mapping class name to RGB tuple
- `annotations`: Dict mapping filename/slice to list of annotation dicts
- `image_dimensions`: Dict mapping filename to dimension string (e.g., "TZCYX")
- `image_shapes`: Dict mapping filename to shape tuple

### Annotation Dict
- `segmentation`: Flattened polygon coordinates `[x1, y1, x2, y2, ...]`
- `bbox`: Rectangle `[x, y, width, height]` (mutually exclusive with segmentation)
- `category`: Class name string

### COCO JSON
- `images`: List of image metadata dicts
- `categories`: List of class dicts with id and name
- `annotations`: List of annotation dicts with id, image_id, category_id, segmentation/bbox

## UI Components

| Component | Description |
|-----------|-------------|
| Tool Section | Buttons for Polygon, Rectangle, Paint Brush, Eraser, SAM tools |
| Class List | QListWidget showing all classes with colors |
| Annotation List | QListWidget showing all annotations for current image |
| Image Label | Central QLabel displaying image with zoom/pan |
| Slice Slider | Navigate through multi-dimensional image slices |
| Menu Bar | File, Edit, View, Tools, Help menus |

## Coordinate Systems

| System | Origin | Units | Used For |
|--------|--------|-------|----------|
| Image Coordinates | Top-left (0,0) | Pixels | Annotation storage, calculations |
| Screen Coordinates | Top-left of window | Pixels | Mouse events, rendering |
| Normalized Coordinates | Top-left (0,0) to (1,1) | Fractional | YOLO export format |
