# Introduction and Goals

## Overview

DigitalSreeni Image Annotator is a PyQt6-based desktop application for annotating images with polygons, rectangles, and paint tools. It integrates SAM 2 (Segment Anything Model) for semi-automated annotations and supports multi-dimensional images (TIFF stacks, CZI files).

**Repository**: https://github.com/bnsreenu/digitalsreeni-image-annotator

## Key Features

- **Manual Annotation Tools**: Polygon, rectangle, paint brush, and eraser tools
- **SAM 2 Integration**: Semi-automated segmentation with Segment Anything Model
- **Grounding DINO Detection**: Text-prompted object detection (single image or batch) with a review/accept overlay
- **SAM 2 Fine-Tuning**: Fine-tune the SAM 2 decoder/encoder in-app with MLflow experiment tracking
- **Keypoint / Pose Annotation**: Per-class named keypoint schema + skeleton (COCO instance model, 3-state visibility) with COCO-keypoints and YOLO-pose export/import
- **Editing & Review**: Undo/redo, canvas selection with handle-based resize/move, vertex editing, and an annotations table with Area and per-mask Detail % simplification
- **Multi-dimensional Image Support**: TIFF stacks, CZI files with dimension assignment
- **Export Formats**: COCO JSON, YOLO v8/v11, Pascal VOC, labeled images, semantic labels
- **Import Formats**: COCO JSON, YOLO datasets (detection + pose)
- **YOLO Training & Prediction**: Train and predict YOLO models (detection, segmentation, and pose) directly from annotations
- **Additional Tools**:
  - Annotation statistics
  - COCO JSON combiner
  - Dataset splitter (train/val/test)
  - Image patcher
  - Image augmenter
  - Slice registration
  - Stack interpolator
  - DICOM converter
- **Project Management**: Save/load projects, autosave, project search
- **UI Features**: Dark mode, adjustable font size, zoom/pan

## Quality Goals

| Priority | Quality Goal | Scenario |
|----------|-------------|----------|
| 1 | **Usability** | Researchers can quickly annotate images with minimal training |
| 2 | **Reliability** | Projects don't get corrupted during save/load operations |
| 3 | **Performance** | Handle large multi-dimensional images without crashes |
| 4 | **Flexibility** | Support multiple export formats for different ML frameworks |

## Stakeholders

| Role | Expectations |
|------|--------------|
| **Researchers** | Easy-to-use tool for image annotation with SAM assistance |
| **Data Scientists** | Export annotations in standard formats (COCO, YOLO) |
| **Microscopy Users** | Handle multi-dimensional images (TIFF stacks, CZI) |
| **Developers** | Extend with new tools and annotation modes |
