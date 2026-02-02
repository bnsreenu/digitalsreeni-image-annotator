# Product Requirements Document
## DigitalSreeni Image Annotator - Enhancement Roadmap

**Version:** 1.0
**Date:** 2026-02-02
**Status:** Draft

---

## Executive Summary

This PRD outlines a comprehensive enhancement roadmap for the DigitalSreeni Image Annotator, focusing on modernization (PyQt6), UX improvements, video support, and integration of SAM 3's text prompting and tracking capabilities. The project follows an incremental approach with three main phases, prioritizing backward compatibility and offline functionality.

---

## Product Vision

Transform DigitalSreeni Image Annotator from an image-focused annotation tool into a comprehensive multimedia annotation platform supporting images, multi-dimensional stacks, and videos, powered by next-generation AI (SAM 3) with text-based prompting and automated tracking.

---

## Goals and Success Metrics

### Primary Goals
1. **Modernize Technology Stack**: Migrate to PyQt6 for long-term maintainability
2. **Improve User Experience**: Streamline workflows, add organization features, improve visibility
3. **Enable Video Annotation**: Support common video formats with frame-by-frame and tracked annotation
4. **Integrate SAM 3**: Leverage text prompting for more intuitive semi-automated annotation
5. **Establish Testing Foundation**: Implement comprehensive test coverage for reliability

### Success Metrics
- **User Efficiency**: 50% reduction in time to organize and navigate large datasets
- **Test Coverage**: 80%+ code coverage for core functionality
- **Video Support**: Successfully annotate 30 FPS video without lag
- **Batch Processing**: Process 100+ images with progress tracking
- **SAM 3 Adoption**: 60%+ of annotations use text prompting vs. manual tools

---

## User Stories

### US-1: Image List Organization
**As a** researcher with 500+ images
**I want** to organize images into folders, see annotation status, and filter/search
**So that** I can quickly navigate my dataset and track progress

**Acceptance Criteria:**
- [ ] Visual indicators show which images have annotations (icon/badge)
- [ ] Images can be organized into collapsible folders/groups
- [ ] Filter by annotation status (not started, in progress, complete)
- [ ] Filter by class (show images with "cell" annotations)
- [ ] Search by filename
- [ ] Sort by name, date, annotation count, completion status
- [ ] Status visible without clicking into each image

---

### US-2: Video File Support
**As a** microscopy researcher
**I want** to load video files (MP4, AVI, MOV) and annotate frames
**So that** I can analyze time-series microscopy videos

**Acceptance Criteria:**
- [ ] Load common video formats: MP4, AVI, MOV
- [ ] Display video timeline with frame scrubbing
- [ ] Navigate frame-by-frame (keyboard arrows)
- [ ] Annotate individual frames like static images
- [ ] Save annotations per-frame (similar to slice naming)
- [ ] Display frame number and timestamp
- [ ] Export annotated frames as images or video overlay

---

### US-3: SAM 3 Text Prompting for Images
**As an** annotator
**I want** to use text prompts like "red blood cell" to segment objects
**So that** I can quickly annotate all matching objects without clicking each one

**Acceptance Criteria:**
- [ ] Text input field for concept prompts (e.g., "mitochondria", "cell nucleus")
- [ ] SAM 3 model downloads automatically on first use
- [ ] Preview all detected instances before accepting
- [ ] Accept/reject individual predictions or all at once
- [ ] Adjust confidence threshold slider
- [ ] Combine text prompts with point refinement if needed
- [ ] Works with current class system (assign detected objects to class)

**Technical Notes:**
- Use Ultralytics SAM 3 implementation if available
- Fall back to GitHub implementation: https://github.com/facebookresearch/sam3
- Model size: 848M parameters (~3GB download)

---

### US-4: SAM 3 Video Tracking
**As a** video annotator
**I want** SAM 3 to track objects across frames automatically
**So that** I don't have to re-annotate the same object in every frame

**Acceptance Criteria:**
- [ ] Annotate object on first frame (manual or SAM 3 text prompt)
- [ ] Click "Track" to propagate annotation across subsequent frames
- [ ] Visual indication of tracked frames (timeline colored segments)
- [ ] Edit/correct tracking on specific frames if needed
- [ ] Handle occlusions and reappearance gracefully
- [ ] Option to track forward, backward, or full video
- [ ] Batch track multiple objects simultaneously

---

### US-5: Batch Processing with SAM 3
**As a** researcher with large datasets
**I want** to apply SAM 3 text prompts to 100+ images in batch
**So that** I can quickly annotate similar objects across my entire dataset

**Acceptance Criteria:**
- [ ] Select multiple images/frames for batch processing
- [ ] Apply single text prompt to all selected items
- [ ] Progress bar shows processing status
- [ ] Review interface: scroll through results, accept/reject per image
- [ ] Keyboard shortcuts for quick review (Accept=Space, Reject=X, Next=Arrow)
- [ ] Quality filtering: only show predictions above threshold
- [ ] Export batch results to COCO/YOLO after review

---

### US-6: PyQt6 Migration
**As a** developer
**I want** the application migrated to PyQt6
**So that** we have long-term support and modern Qt features

**Acceptance Criteria:**
- [ ] All PyQt5 imports replaced with PyQt6
- [ ] Qt enum changes handled (Qt.AlignLeft → Qt.AlignmentFlag.AlignLeft)
- [ ] Signal/slot connections updated to PyQt6 syntax
- [ ] All existing functionality works identically
- [ ] No visual regressions in UI
- [ ] Update requirements.txt and setup.py
- [ ] Update documentation with PyQt6 references
- [ ] Test on Windows, macOS, Linux

**Migration Notes:**
- PyQt6 changes: https://www.riverbankcomputing.com/static/Docs/PyQt6/pyqt5_differences.html
- Update imports: `PyQt5.QtCore` → `PyQt6.QtCore`
- Enum changes: `Qt.LeftButton` → `Qt.MouseButton.LeftButton`

---

## Technical Requirements

### TR-1: Testing Infrastructure
**Priority:** Critical
**Phase:** 1

Establish comprehensive test coverage for all existing and new functionality.

**Requirements:**
- [ ] Set up pytest and pytest-qt testing framework
- [ ] Unit tests for utility functions (calculate_area, conversions, etc.)
- [ ] Integration tests for export/import functions
- [ ] UI tests for critical workflows (create annotation, save project)
- [ ] Achieve 80%+ code coverage
- [ ] CI/CD pipeline runs tests automatically (GitHub Actions)
- [ ] Test both PyQt5 (current) and PyQt6 (target) during migration

**Example Test Structure:**
```
tests/
├── unit/
│   ├── test_utils.py              # calculate_area, calculate_bbox
│   ├── test_conversions.py        # QImage ↔ numpy
│   └── test_polygon_operations.py # Shapely operations
├── integration/
│   ├── test_export_formats.py     # COCO, YOLO exports
│   ├── test_import_formats.py     # Import workflows
│   └── test_sam_utils.py          # SAM inference
└── ui/
    ├── test_annotation_workflow.py # Create/edit annotations
    └── test_project_management.py  # Save/load projects
```

---

### TR-2: Performance Requirements
**Priority:** High
**Phase:** 2-3

**Requirements:**
- [ ] Video playback: 30 FPS minimum for 1080p
- [ ] Batch processing: Handle 100+ images with progress tracking
- [ ] Lazy loading: Load multi-dimensional slices on-demand (not all in memory)
- [ ] GPU acceleration: Detect and use GPU for SAM inference when available
- [ ] Memory optimization: Limit RAM usage to 4GB for typical datasets
- [ ] Responsive UI: No blocking operations over 100ms without progress indicator

---

### TR-3: Backward Compatibility
**Priority:** Critical
**Phase:** All

**Requirements:**
- [ ] Existing project JSON files load correctly
- [ ] Old annotations render properly
- [ ] Export formats remain compatible
- [ ] Settings and preferences migrate automatically
- [ ] Provide migration script if breaking changes necessary

---

### TR-4: Offline Functionality
**Priority:** High
**Phase:** All

**Requirements:**
- [ ] All features work without internet connection (after initial model download)
- [ ] SAM models cached locally
- [ ] No cloud API dependencies
- [ ] Local-only data processing

---

### TR-5: Installation Simplicity
**Priority:** High
**Phase:** All

**Requirements:**
- [ ] Single command install: `pip install digitalsreeni-image-annotator`
- [ ] Automatic dependency resolution
- [ ] Models download automatically on first use
- [ ] Clear error messages for missing dependencies
- [ ] Support Python 3.10, 3.11, 3.12

---

## Implementation Phases

### Phase 1: Foundation & Modernization
**Duration:** 2-3 months
**Goal:** Migrate to PyQt6, establish testing, improve UX foundations

#### Milestones

**M1.1: Testing Infrastructure** ✓
- Set up pytest, pytest-qt
- Write unit tests for existing utilities
- Achieve 40% coverage baseline
- Set up CI/CD pipeline

**M1.2: PyQt6 Migration** ✓
- Update all imports and enums
- Test core functionality parity
- Fix any regressions
- Update documentation
- Target: 100% feature parity with PyQt5

**M1.3: Image List UX Improvements** ✓
- Add visual annotation status indicators
- Implement folder/group organization
- Add filtering (status, class, search)
- Add custom sorting options
- Polish UI based on user feedback

**M1.4: Video Loading Infrastructure** ✓
- Add video file format support (MP4, AVI, MOV)
- Implement frame extraction and navigation
- Display video timeline UI
- Enable frame-by-frame annotation (no tracking yet)
- Save per-frame annotations

#### Success Criteria
- [ ] All existing features work in PyQt6
- [ ] 60%+ test coverage
- [ ] Users can organize and navigate large datasets efficiently
- [ ] Videos load and display frames correctly

---

### Phase 2: SAM 3 Integration
**Duration:** 2-3 months
**Goal:** Integrate SAM 3 text prompting and basic tracking

#### Milestones

**M2.1: SAM 3 Text Prompting for Images** ✓
- Integrate SAM 3 model (Ultralytics or GitHub)
- Implement text prompt UI
- Preview and accept/reject predictions
- Confidence threshold adjustment
- Assign predictions to classes

**M2.2: Batch Processing with SAM 3** ✓
- Select multiple images for batch processing
- Apply text prompts in batch
- Implement review workflow UI
- Progress tracking and cancellation
- Keyboard shortcuts for efficient review

**M2.3: SAM 3 Video Tracking** ✓
- Implement tracking mode for video frames
- Track forward/backward from seed frame
- Handle occlusions and tracking failures
- Visual timeline indicators for tracked segments
- Manual correction of tracking errors

**M2.4: Performance Optimization** ✓
- GPU acceleration for SAM inference
- Lazy loading for large datasets
- Memory profiling and optimization
- Background processing threads

#### Success Criteria
- [ ] Text prompts successfully segment objects in images
- [ ] Batch processing handles 100+ images smoothly
- [ ] Video tracking maintains object identity across frames
- [ ] 80%+ test coverage

---

### Phase 3: Advanced Features
**Duration:** 2-3 months
**Goal:** Polish and extend with advanced capabilities

#### Milestones

**M3.1: Annotation Propagation** ✓
- Copy annotations to similar frames
- Interpolate annotations across slices
- Smart propagation based on image similarity

**M3.2: Advanced Editing Tools** ✓
- Vertex editing for polygons
- Polygon splitting
- Edge smoothing and simplification
- Merge/subtract operations

**M3.3: Quality Control Tools** ✓
- Annotation validation rules
- Consistency checks across dataset
- Highlight potential errors
- Export quality report

**M3.4: Export Enhancements** ✓
- Custom export templates
- More format options
- Batch export operations
- Export progress tracking

#### Success Criteria
- [ ] Users can efficiently edit and refine annotations
- [ ] Quality control catches common errors
- [ ] Export supports diverse workflows
- [ ] 90%+ test coverage

---

## Detailed Feature Specifications

### Image List Organization (US-1)

#### Visual Status Indicators
- **Indicator Types:**
  - ⬜ Not started (no annotations)
  - 🟨 In progress (some annotations, not complete)
  - ✅ Complete (user-marked or meets criteria)
  - ⚠️ Review needed (quality issues flagged)

- **Display:** Small icon badge on thumbnail or in list row
- **Interaction:** Click to filter by status

#### Folder/Group Organization
- **Structure:** Hierarchical tree view (similar to file explorer)
- **Operations:**
  - Create folder/group
  - Drag-drop images to organize
  - Collapse/expand folders
  - Rename/delete folders
- **Persistence:** Folder structure saved in project JSON

#### Filtering and Search
- **Filter Options:**
  - Annotation status (dropdown)
  - Class (multi-select checkboxes)
  - Date range (for imported images)
  - Custom tags (future)
- **Search:** Real-time filter as user types
- **Clear Filters:** One-click reset button

#### Custom Sorting
- **Sort By:**
  - Name (A-Z, Z-A)
  - Date added (newest/oldest)
  - Annotation count (most/least)
  - Completion status (complete first/last)
  - File size (largest/smallest)
- **UI:** Dropdown menu or column headers (sortable table view)

---

### Video File Support (US-2)

#### Video Loading
- **Formats:** MP4, AVI, MOV (via OpenCV VideoCapture)
- **Metadata Extraction:**
  - Total frames
  - Frame rate (FPS)
  - Duration
  - Resolution
- **UI:** Video files appear in image list with ▶️ icon

#### Timeline Navigation
- **Timeline Slider:** Scrub through video frames
- **Frame Counter:** Display current frame / total frames
- **Timestamp:** Display time in MM:SS format
- **Keyboard Controls:**
  - Left/Right arrows: Previous/next frame
  - Space: Play/pause (future)
  - Home/End: First/last frame

#### Frame Annotation
- **Storage:** Annotations stored with key: `{video_filename}_F{frame_number}`
  - Example: `microscopy.mp4_F0042`
- **Visual:** Current frame displayed in ImageLabel, annotate like static image
- **Indicator:** Timeline shows which frames have annotations (colored marks)

#### Export
- **Per-Frame Export:** Export annotated frames as individual images
- **Video Overlay Export:** Render annotations on video and export as new video file (future)
- **Format Compatibility:** COCO JSON (frames as separate images), YOLO (per-frame)

---

### SAM 3 Integration

#### Text Prompting UI (US-3)

**UI Components:**
- **Text Input:** Single-line field with placeholder "Enter concept (e.g., 'red blood cell', 'mitochondria')"
- **Class Assignment:** Dropdown to select which class detected objects belong to
- **Confidence Threshold:** Slider (0.0 - 1.0, default 0.5) with live update
- **Action Buttons:**
  - "Detect" - Run SAM 3 inference
  - "Accept All" - Add all predictions as annotations
  - "Clear" - Remove predictions from view

**Workflow:**
1. User enters text prompt: "cell nucleus"
2. User clicks "Detect"
3. SAM 3 runs inference, returns masks
4. Masks displayed as semi-transparent overlays (different colors per instance)
5. User adjusts threshold if needed (predictions update)
6. User clicks individual predictions to accept/reject, or "Accept All"
7. Accepted predictions added to annotations list

**Technical Implementation:**
```python
# Example API usage
from sam3 import SAM3  # or ultralytics.SAM3

model = SAM3("sam3.pt")
results = model.predict(
    image,
    text_prompt="cell nucleus",
    confidence=0.5
)

for mask, score in zip(results.masks, results.scores):
    if score > threshold:
        # Convert mask to polygon, add annotation
        polygon = mask_to_polygon(mask)
        add_annotation(polygon, category="nucleus")
```

#### Video Tracking (US-4)

**UI Components:**
- **Track Button:** Appears after annotation is created
- **Tracking Direction:** Radio buttons (Forward / Backward / Full Video)
- **Tracking Status:** Timeline shows tracked segments in color
- **Edit Mode:** Click frame on timeline to view/edit tracking result

**Workflow:**
1. User annotates object on frame 100 (manually or via text prompt)
2. User clicks "Track" button
3. SAM 3 propagates annotation to frames 101, 102, ..., N
4. Timeline shows green bar for successfully tracked frames
5. User scrubs through, sees annotation move with object
6. If tracking error on frame 150, user corrects manually
7. User can re-track from frame 150 onward

**Technical Implementation:**
```python
# SAM 3 tracking mode
tracker = SAM3Tracker(model)
tracker.init(frame_0, mask_0)

for frame in video_frames[1:]:
    mask_t, confidence = tracker.track(frame)
    if confidence > threshold:
        add_annotation(frame_id, mask_to_polygon(mask_t))
    else:
        # Flag for manual review
        mark_tracking_uncertain(frame_id)
```

#### Batch Processing (US-5)

**UI Components:**
- **Batch Mode Toggle:** Enable batch mode in image list
- **Selection:** Checkboxes on images, "Select All" button
- **Batch Dialog:**
  - Text prompt input
  - Class assignment
  - Confidence threshold
  - "Process" button

**Review Interface:**
- **Grid View:** Thumbnails of all processed images with overlay
- **Status Indicators:** ✅ Accepted, ❌ Rejected, ⏳ Pending
- **Navigation:** Arrow keys to move between images
- **Actions:**
  - Space: Accept
  - X: Reject
  - E: Edit (open in main view)

**Workflow:**
1. User selects 200 images
2. User clicks "Batch Process"
3. Dialog opens, user enters "red blood cell", threshold 0.6
4. User clicks "Process"
5. Progress bar shows "Processing 45/200..."
6. Review interface opens with all results
7. User quickly reviews using keyboard shortcuts
8. Click "Finalize" - accepted annotations added to project

---

## Technical Architecture

### Updated Component Diagram

```
┌─────────────────────────────────────────────────┐
│   DigitalSreeni Image Annotator (PyQt6)        │
│                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐     │
│  │   GUI    │  │  SAM 3   │  │  YOLO    │     │
│  │ (PyQt6)  │  │(Ultraly.)│  │ Trainer  │     │
│  └──────────┘  └──────────┘  └──────────┘     │
│                                                 │
│  ┌──────────────────────────────────────────┐  │
│  │   Video Processing (OpenCV)              │  │
│  │   - Frame extraction                     │  │
│  │   - Timeline navigation                  │  │
│  │   - SAM 3 tracking integration           │  │
│  └──────────────────────────────────────────┘  │
│                                                 │
│  ┌──────────────────────────────────────────┐  │
│  │   Image Processing                       │  │
│  │   (NumPy, OpenCV, Shapely)              │  │
│  └──────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
         │                  │
         ▼                  ▼
   File System        SAM 3 Model Cache
```

### New Modules

**video_handler.py**
- Load video files (MP4, AVI, MOV)
- Extract frames on-demand
- Provide frame iterator
- Cache recent frames for performance

**sam3_utils.py**
- Load SAM 3 model
- Text prompt inference
- Video tracking mode
- Batch processing utilities

**batch_processor.py**
- Batch inference manager
- Progress tracking
- Result aggregation
- Review UI integration

**annotation_organizer.py**
- Folder/group management
- Filtering and sorting logic
- Status tracking

---

## Data Model Updates

### Project JSON Schema (v2.0)

```json
{
  "version": "2.0",
  "images": ["image1.png", "video1.mp4"],
  "image_paths": {
    "image1.png": "/path/to/image1.png",
    "video1.mp4": "/path/to/video1.mp4"
  },
  "video_metadata": {
    "video1.mp4": {
      "total_frames": 300,
      "fps": 30,
      "duration_seconds": 10,
      "resolution": [1920, 1080]
    }
  },
  "folders": {
    "Experiment 1": ["image1.png"],
    "Time Series": ["video1.mp4"]
  },
  "annotation_status": {
    "image1.png": "complete",
    "video1.mp4_F0000": "in_progress"
  },
  "classes": ["cell", "nucleus"],
  "class_colors": {
    "cell": [255, 0, 0],
    "nucleus": [0, 255, 0]
  },
  "annotations": {
    "image1.png": [...],
    "video1.mp4_F0000": [...],
    "video1.mp4_F0001": [...]
  },
  "tracking_segments": {
    "video1.mp4": [
      {
        "annotation_id": "cell_track_1",
        "start_frame": 10,
        "end_frame": 50,
        "frames": [10, 11, 12, ..., 50]
      }
    ]
  }
}
```

---

## Testing Strategy

### Unit Tests (Target: 80% coverage)

**Core Utilities:**
- `test_utils.py`: calculate_area, calculate_bbox, normalize_image
- `test_conversions.py`: QImage ↔ numpy, coordinate transformations
- `test_polygon_operations.py`: Shapely merge, validation, simplification

**Video Processing:**
- `test_video_handler.py`: Load video, extract frames, frame iteration
- `test_frame_naming.py`: Frame naming convention (`video_F0042`)

**SAM 3 Integration:**
- `test_sam3_utils.py`: Model loading, text prompt inference, mask conversion
- Mock SAM 3 model for fast testing

### Integration Tests

**Export/Import:**
- `test_export_formats.py`: COCO, YOLO, Pascal VOC with video frames
- `test_import_formats.py`: Backward compatibility with v1 projects

**Annotation Workflows:**
- `test_annotation_propagation.py`: Copy/interpolate across frames
- `test_batch_processing.py`: Batch SAM 3 inference and review

### UI Tests (pytest-qt)

**Critical Workflows:**
- `test_video_loading.py`: Load video, navigate timeline, annotate frame
- `test_folder_organization.py`: Create folder, drag-drop, filter
- `test_sam3_text_prompt.py`: Enter prompt, preview, accept predictions
- `test_project_management.py`: Save/load with video and folders

### Performance Tests

**Benchmarks:**
- Load 1080p video and extract first frame: < 1 second
- SAM 3 text prompt inference on 1 image: < 5 seconds (GPU)
- Batch process 100 images: < 10 minutes (GPU)
- Navigate 300-frame video: < 100ms per frame switch

---

## Risks and Mitigations

### Risk: SAM 3 Model Size (3GB)

**Impact:** Long download time on first use, storage concerns

**Mitigation:**
- Clear messaging during download with progress bar
- Option to select smaller SAM 2 models as fallback
- Compress model using quantization (future)

### Risk: PyQt6 Migration Breakage

**Impact:** Regressions in existing functionality

**Mitigation:**
- Comprehensive test coverage before migration
- Parallel testing: run tests on both PyQt5 and PyQt6
- Beta release for user testing before full release

### Risk: Video Performance on Large Files

**Impact:** Lag, crashes with 4K video

**Mitigation:**
- Lazy frame loading (load on-demand)
- Downscale for display, keep original for export
- Provide performance settings (quality vs. speed)

### Risk: Backward Compatibility Challenges

**Impact:** Old projects fail to load in new version

**Mitigation:**
- Schema versioning in project JSON
- Automatic migration script for v1 → v2
- Fallback: export old project, import into new version

---

## Success Criteria & KPIs

### Phase 1: Foundation
- [ ] All PyQt5 features work in PyQt6 (100% parity)
- [ ] Test coverage ≥ 60%
- [ ] Users can organize 500+ images efficiently (subjective feedback)
- [ ] Videos load in < 2 seconds

### Phase 2: SAM 3
- [ ] Text prompting accuracy ≥ 80% (user acceptance rate)
- [ ] Batch processing: 100 images in < 10 minutes
- [ ] Video tracking: maintains identity for 95% of frames (no occlusion)
- [ ] Test coverage ≥ 80%

### Phase 3: Advanced
- [ ] Annotation propagation reduces manual work by 40%
- [ ] Quality control catches 90% of obvious errors
- [ ] Export supports 5+ formats reliably
- [ ] Test coverage ≥ 90%

---

## Open Questions

1. **SAM 3 Availability:** Is SAM 3 available via Ultralytics? If not, use GitHub repo.
2. **Video Export Formats:** Should we support video overlay export (annotations drawn on video)?
3. **Collaboration Features:** Future consideration for multi-user annotation?
4. **Cloud Storage:** Any interest in cloud export (S3, GCS) or keep local-only?
5. **3D Visualization:** Interest in 3D viewer for Z-stacks (Phase 4)?

---

## Appendix

### References

**SAM 3 Documentation:**
- [Ultralytics SAM 3 Docs](https://docs.ultralytics.com/models/sam-3/)
- [GitHub Repository](https://github.com/facebookresearch/sam3)
- [Research Paper](https://arxiv.org/abs/2511.16719)
- [Roboflow SAM 3 Overview](https://blog.roboflow.com/what-is-sam3/)
- [MarkTechPost Article](https://www.marktechpost.com/2025/11/20/meta-ai-releases-segment-anything-model-3-sam-3-for-promptable-concept-segmentation-in-images-and-videos/)

**PyQt6 Migration:**
- [PyQt5 to PyQt6 Differences](https://www.riverbankcomputing.com/static/Docs/PyQt6/pyqt5_differences.html)

### Glossary

- **Frame:** Single image extracted from a video at a specific timestamp
- **Text Prompting:** Using natural language (e.g., "red blood cell") to specify segmentation target
- **Tracking:** Propagating annotations across video frames automatically
- **Batch Processing:** Applying operations to multiple images/frames at once

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-02-02 | Initial PRD based on user interview |

---

**Document Owner:** Repository Maintainer
**Next Review:** After Phase 1 completion
