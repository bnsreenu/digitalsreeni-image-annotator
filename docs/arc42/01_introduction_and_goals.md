# Introduction and Goals

## Overview

DigitalSreeni Image Annotator is a PyQt5-based desktop application for annotating images with polygons, rectangles, and paint tools. It integrates SAM 2 (Segment Anything Model) for semi-automated annotations and supports multi-dimensional images (TIFF stacks, CZI files).

**Repository**: https://github.com/cofade/digitalsreeni-image-annotator (fork of https://github.com/bnsreenu/digitalsreeni-image-annotator)

## Key Features

- **Manual Annotation Tools**: Polygon, rectangle, paint brush, and eraser tools
- **SAM 2 Integration**: Semi-automated segmentation with Segment Anything Model
- **Multi-dimensional Image Support**: TIFF stacks, CZI files with dimension assignment
- **Export Formats**: COCO JSON, YOLO v8/v11, Pascal VOC, labeled images, semantic labels
- **Import Formats**: COCO JSON, YOLO datasets
- **YOLO Training**: Train YOLO models directly from annotations
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
