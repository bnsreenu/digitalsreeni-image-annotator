# Runtime View

## Application Startup

```
┌──────────┐
│  main()  │
└────┬─────┘
     │
     ├─> Create QApplication
     │
     ├─> Initialize ImageAnnotator
     │   │
     │   ├─> Create ImageLabel
     │   ├─> Initialize SAMUtils
     │   ├─> Create Menu Bar
     │   ├─> Create Tool Buttons
     │   ├─> Create Class List Widget
     │   └─> Create Annotation List Widget
     │
     ├─> Show Main Window
     │
     └─> Enter Event Loop (app.exec())
```

## Annotation Creation - Manual Polygon

```
User clicks on image
    │
    ├─> ImageLabel.mousePressEvent()
    │   │
    │   ├─> Check current_tool == "Polygon"
    │   │
    │   ├─> Convert screen coords to image coords
    │   │   (account for zoom_factor, offset)
    │   │
    │   ├─> Add point to current_annotation list
    │   │
    │   └─> update() to trigger repaint
    │
User continues clicking points...
    │
User presses Enter
    │
    ├─> ImageLabel.keyPressEvent()
    │   │
    │   ├─> Check key == Qt.Key_Return
    │   │
    │   ├─> main_window.add_annotation(polygon_points)
    │   │   │
    │   │   ├─> Create annotation dict:
    │   │   │   {
    │   │   │     "segmentation": [x1, y1, x2, y2, ...],
    │   │   │     "category": current_class
    │   │   │   }
    │   │   │
    │   │   ├─> all_annotations[image_file_name].append(annotation)
    │   │   │
    │   │   ├─> Update annotation list widget
    │   │   │
    │   │   └─> Trigger autosave (if enabled)
    │   │
    │   └─> Clear current_annotation
    │
    └─> update() to show final annotation
```

## Mask Selection & Deletion on the Canvas (issue #75)

Active only when no drawing/SAM tool is selected (`ImageLabel._is_select_mode()`).
Double-click still enters vertex-edit; Ctrl+drag still pans.

```
User clicks / drags on image (no tool active)
    │
    ├─> ImageLabel mouse press/move/release
    │   ├─> click            → annotation_at(pos)        (smallest mask, seg or bbox)
    │   ├─> click empty      → []                        (clears selection)
    │   ├─> drag             → annotations_in_rect(rect) (rubber band, bounds-intersect)
    │   └─> Shift            → toggle (click) / add (drag)
    │
    ├─> emit canvasSelectionChanged(annotations, mode)   mode = replace|add|toggle
    │
    └─> AnnotationController.apply_canvas_selection()
        ├─> compute new set from highlighted_annotations + annotations per mode
        ├─> image_label.highlighted_annotations = new    (blue selection overlay)
        ├─> mirror onto annotation_list (blockSignals while selecting)
        └─> enable Merge (≥2) / Change Class (≥1)

User presses Delete (canvas focused)
    │
    ├─> ImageLabel.keyPressEvent → deleteSelectionRequested
    └─> AnnotationController.delete_selected_annotations()  (record_history → remove → re-sort → autosave)
```

The canvas and the list share one selection (matched by dict value-equality), so
Delete/Merge/Change-Class behave the same from either surface. See ADR-022.

**Delete and merge are now frictionless and reversible.** Delete removes the
selection immediately — no "Are you sure?" confirmation and no "N deleted" success
dialog. Merge always replaces the originals with their union (no keep/delete prompt)
and shows no success dialog. Both snapshot the pre-edit state first, so **Ctrl+Z**
restores it; the removed confirmations are unnecessary now that undo is the net.
See [ADR-026](09_architecture_decisions.md#adr-026-snapshot-based-undoredo-for-annotation-edits).

## Shape Editing on the Canvas (issue #40)

When exactly one shape is selected (idle mode), its 8 selection handles become
draggable — direct manipulation, no separate mode, for **any** shape (polygon,
mask, or imported box). The geometry mutates in place so the canvas updates live;
release clamps it into the image and persists.

```
One shape selected → handles are grab targets (hover shows resize/move cursors)
    │
    ├─> press on a handle      → "resize"  (anchor = opposite corner/edge)
    ├─> press inside the shape → "pending_move" → "move" once drag > 3px/zoom
    │                            (plain click, no drag → falls through to select)
    ├─> press outside          → normal rubber-band selection (#75)
    │
    ├─> drag → _update_bbox_drag(): mutate geometry in place
    │          ├─ bbox kind → set [x,y,w,h]   (resize trims; move translates)
    │          └─ seg  kind → scale vertices (resize) / translate (move);
    │                         _sync_bbox_key keeps an imported bbox consistent
    │
    ├─> release → clamp into the image (ADR-024: move slides inside, resize clamps)
    │             emit bboxEditCommitted
    │             └─> AnnotationController.commit_bbox_edit()
    │                 save → rebuild list (area refreshes) → re-mirror selection → autosave
    │
    └─> Esc during drag → restore original geometry, cancel
```

Polygon vertex edits (double-click) are likewise clamped into the image on Enter.
See ADR-023 (shape editing) and ADR-024 (bounds enforcement).

## Placing a Keypoint / Pose Instance (issue #35, ADR-029)

A "pose class" first needs a keypoint schema (right-click the class → **Define
Keypoint Schema** → ordered names + skeleton). Then the Keypoint tool places one
instance's K points **in schema order**:

```
Define schema on the class (names, skeleton, flip_idx) → keypoint_schemas[class]
    │
Activate Keypoint tool (gated: warns if the current class has no schema)
    │
Place points in order:
    ├─ left-click       → next point VISIBLE (v=2)
    ├─ right / Shift+left → next point OCCLUDED (v=1)
    ├─ Backspace        → remove the last placed point (go back)
    ├─ auto-finish at K  ─┐
    └─ Enter (finish early: pad remaining points with v=0) ─┐
                                                            │
    KeypointTool.finishKeypointsRequested ──────────────────┘
        └─> AnnotationController.finish_keypoint():
            record_history() → build {keypoints, num_keypoints, bbox, category}
            → clamp into image → add_annotation_to_list → save → autosave
    │
Rendering: draw_annotations "keypoints" branch — skeleton (labelled points only) +
           markers coloured by visibility + faint instance box + label
    │
Editing (select the instance, idle mode):
    ├─ drag a marker              → single-point move (editing_keypoint)
    │      commit → keypointEditCommitted → commit_keypoint_edit
    ├─ right-click a marker       → toggle visible ↔ occluded
    │      commit → keypointEditCommitted → commit_keypoint_edit
    └─ drag a box handle / inside → transform the WHOLE pose (kind="kpt",
           the existing #40 bbox_edit machinery — _scale_keypoints /
           _translate_keypoints instead of _scale_segmentation)
           commit → bboxEditCommitted → commit_bbox_edit
    (both commit paths: save + undo, ADR-026)
```

Merge and cross-schema change-class are blocked for keypoint instances. See ADR-029.

## Exporting / Importing a Pose Class (issue #35 PR-2, ADR-029)

```
Export → COCO JSON:
    export_coco_json(..., keypoint_schemas=mw.keypoint_schemas)
        ├─ per pose class: category gains "keypoints" (names), "skeleton"
        │      (0-based → 1-based per COCO spec), "flip_idx" (app extension,
        │      kept 0-based)
        └─ per instance: create_coco_annotation() checks "keypoints" in ann
               FIRST (before segmentation/bbox) → keypoints/num_keypoints/bbox,
               no "segmentation" key

Export → YOLO (v5+):
    export_yolo_v5plus(..., keypoint_schemas=mw.keypoint_schemas)
        ├─ _pose_export_check() scans the annotations actually being exported
        │      ├─ no keypoints anywhere → ordinary export, unchanged
        │      ├─ exactly one (K, flip_idx) shared by every exported class
        │      │      → proceed: label lines gain 3K trailing (x,y,v) tokens,
        │      │        data.yaml gains kpt_shape:[K,3] + flip_idx
        │      └─ >1 distinct K, or a pose class mixed with a non-pose class
        │             → raise ValueError BEFORE any file is written
        └─ io_controller.export_annotations catches ValueError →
               QMessageBox.warning("Export Error", ...) (same pattern as the
               existing YOLO import-error surfacing)

Import (COCO or YOLO v5+):
    import_coco_json() / import_yolo_v5plus() → uniformly return
        (annotations, image_info, keypoint_schemas) — {} where nothing recovered
            ├─ COCO: schema recovered per category carrying "keypoints"
            │      (skeleton 1-based → 0-based; flip_idx read straight through)
            └─ YOLO-pose: one schema (generic kp0..kp{K-1} names, no skeleton)
                   from data.yaml's kpt_shape/flip_idx, applied to EVERY class
                   in `names` (kpt_shape is dataset-global, not per-class)
    │
    io_controller.import_annotations():
        ├─ _rebuild_imported_annotation(ann, ...) — a keypoint-shaped result
        │      gets a FULLY SEPARATE dict (no "segmentation"/"type" keys at
        │      all), never a shared base dict with those keys set to None.
        │      Existence-only checks elsewhere ("segmentation" in annotation,
        │      not a None-guard — draw_annotations, start_polygon_edit,
        │      eraser_tool.py) would otherwise misfire on a None-valued key.
        └─ recovered schemas registered into mw.keypoint_schemas via
               sanitize_schema() (malformed → dropped with a print, same
               pattern as project load)
```

## Training + Predicting with a Pose Model (issue #35 PR-3, ADR-029)

Reuses the existing in-app YOLO train/predict loop (see "In-app YOLO Training"
below) end to end; pose only changes what the dataset/registered-model yaml
carries and how a "pose" result is unpacked into a temp annotation.

```
Prepare YOLO Dataset:
    YOLOTrainer.prepare_dataset()
        └─> export_yolo_v5plus(..., keypoint_schemas=mw.keypoint_schemas)
              (schema-aware export, PR-2) — data.yaml gains kpt_shape/
              flip_idx IFF a pose class is among the exported annotations

Load Model (Training menu): a '*-pose.pt' checkpoint → model.task == "pose"
    │
Train Model → YOLOTrainer.train_model() pre-flight guard, BEFORE any
    training work starts:
        model.task == "pose"  XOR  "kpt_shape" in the prepared yaml
            → raise ValueError (both directions guarded — a pose model on a
              non-pose dataset, and vice versa) → TrainingThread.run() →
              training_finished() → QMessageBox.critical("Training Error")
    │
model.train(...) proceeds — on_fit_epoch_end() also surfaces val/pose_loss
    + val/kobj_loss in the progress dialog (same pattern as the existing
    val/box_loss / val/seg_loss for detect/segment runs)
    │
_register_trained_model(): sibling data.yaml gets kpt_shape/flip_idx read
    back from the training yaml, PLUS — best-effort — a full
    "keypoint_schema" key when every trained class shares one identical
    schema in mw.keypoint_schemas (richer than bare kpt_shape/flip_idx, so
    a later prediction load doesn't fall back to generic point names)
    │
    ... later, possibly a new session ...
    │
Prediction Settings > Load Model → load_prediction_model(model_path, yaml)
    └─> prediction_keypoint_schema reconstructed from the registered yaml:
            "keypoint_schema" present → sanitize_schema(that)       (rich)
            else "kpt_shape" present  → sanitize_schema(generic
                                          kp0..kp{K-1} names, no skeleton)
            else                      → None (not a pose model)
    │
"Predict with YOLO Model" dialog → Predict on the current image
    └─> YOLOTrainer.predict() — no hardcoded task='segment' any more, so a
          pose checkpoint's result carries .keypoints instead of .masks
        └─> YOLOController.process_yolo_results():
              is_pose = (yolo_trainer.model.task == "pose")
              ├─ pose: build one temp instance per detection —
              │      {keypoints: flat [x,y,v]*K (v ALWAYS 2 — Ultralytics
              │      gives no true 3-state occlusion signal), num_keypoints,
              │      bbox, category_name: "Temp-<class>", score, temp: True}
              │      — deliberately NO "segmentation" key (ADR-029
              │      discriminator, unchanged)
              │      seed mw.keypoint_schemas["Temp-<class>"] from
              │      prediction_keypoint_schema if not already present
              └─ detect/segment: unchanged box/polygon temp-annotation path
    │
Review (shared Temp-* machinery, DINOReviewEventFilter):
    rendering: draw_annotations "keypoints" branch — markers + skeleton
        lines if the seeded schema carries skeleton edges, points only
        otherwise, plus the faint instance box
    ├─ Enter → DINOController.accept_visible_temp_classes():
    │      "Temp-<class>" → "<class>"; a seeded schema is carried to the
    │      permanent class name (warns and keeps the existing schema
    │      instead of overwriting it if K differs)
    └─ Esc   → DINOController.reject_visible_temp_classes(): temp
               annotations dropped, any orphaned "Temp-<class>" schema
               entry popped too
```

Output lands in the same `models/yolo/custom/<project>/weights/best.pt`
location as any other YOLO run — only the sibling `data.yaml` gains the pose
keys. See ADR-029.

## Adjusting Mask Complexity — Detail % (issue #24)

The Annotations table carries a per-row **Detail %** spinbox (100 = raw). Dialing
it down thins a dense SAM/DINO mask; dialing back to 100 restores it exactly.

```
User changes a row's Detail % spinbox (1..100)
    │
    └─> AnnotationController.on_detail_pct_changed(row, pct)
        ├─> resolve the live drawn object (value-equality, _live_annotation)
        ├─> pct == 100 → segmentation = segmentation_raw (restore)
        │   pct  < 100 → lazy-init segmentation_raw (first time);
        │                segmentation = simplify_polygon(raw, pct)  [Douglas-Peucker]
        ├─> recompute bbox key if present
        ├─> refresh the row's Area cell + UserRole in place (no rebuild)
        └─> image_label.update() → save_current_annotations() → auto_save()
```

The effective (simplified) `segmentation` renders and exports; `segmentation_raw`
+ `detail_pct` persist in the `.iap`. See ADR-025.

## SAM-Assisted Annotation (SAM-box / SAM-points)

```
User selects SAM model
    │
    ├─> ImageAnnotator.change_sam_model()
    │   │
    │   └─> SAMUtils.change_sam_model("SAM 2 tiny")
    │       │
    │       ├─> Download model if first use (cached after)
    │       │
    │       └─> Load SAM model instance
    │
User clicks "SAM Point" button
    │
    ├─> sam_points_active = True
    │
User clicks positive points (left click)
    │
    ├─> ImageLabel.mousePressEvent()
    │   │
    │   └─> sam_positive_points.append((x, y))
    │
User clicks negative points (right click)
    │
    ├─> ImageLabel.mousePressEvent()
    │   │
    │   └─> sam_negative_points.append((x, y))
    │
User presses Enter to run SAM
    │
    ├─> ImageLabel.keyPressEvent()
    │   │
    │   ├─> SAMUtils.apply_sam_points(
    │   │       image=current_qimage,
    │   │       positive_points=sam_positive_points,
    │   │       negative_points=sam_negative_points
    │   │   )
    │   │   │
    │   │   ├─> Convert QImage to numpy array
    │   │   │   (handle 8-bit, 16-bit, grayscale, RGB)
    │   │   │
    │   │   ├─> sam_model.predict(
    │   │   │       image,
    │   │   │       points=[[...positive...], [...negative...]],
    │   │   │       labels=[[1, 1, ...], [0, 0, ...]]
    │   │   │   )
    │   │   │
    │   │   ├─> Extract mask from results[0].masks.data[0]
    │   │   │
    │   │   ├─> Convert mask to polygon contours
    │   │   │   (cv2.findContours)
    │   │   │
    │   │   └─> Return {"segmentation": [...], "score": float}
    │   │
    │   ├─> Display prediction as temp_sam_prediction
    │   │
    │   └─> User accepts (Enter) or rejects (Esc)
    │
User accepts prediction
    │
    ├─> main_window.add_annotation(prediction["segmentation"])
    │
    └─> Clear SAM state, reset to normal mode
```

## LLM-Assisted Detection (Grounding DINO + SAM)

End-to-end flow when the user clicks "Detect Current Image" in the DINO panel:

```
User clicks "Detect Current Image"
    │
    ├─> Preflight: dino_model_loaded? sam_model selected? image loaded?
    │   (early return with QMessageBox if any check fails)
    │
    ├─> Resolve DINO model path via _resolve_dino_model_path()
    │   │
    │   ├─> Path exists → skip download
    │   └─> Missing  → DINOUtils.download_model() pulls from HuggingFace Hub
    │                  (huggingface_hub.snapshot_download into models/<name>/)
    │
    ├─> Build class_configs from widgets (single source of truth):
    │   - phrases:    dino_phrase_panel.get_phrases_for(class_name)
    │   - thresholds: dino_class_table.get_class_configs()
    │
    ├─> DINOUtils.detect(qimage, class_configs, model_name)
    │   │
    │   ├─> Convert QImage to numpy (on calling thread)
    │   ├─> _run_sync: spawn QThread, pump caller's event loop while waiting
    │   ├─> On the worker thread:
    │   │     - Load (or reuse cached) GroundingDinoForObjectDetection
    │   │     - Run inference per phrase, apply per-class NMS
    │   │     - Apply cross-class NMS
    │   └─> Returns [{class_name, bbox: [x1,y1,x2,y2], score, label}, ...]
    │
    ├─> Feed DINO bboxes into SAMUtils.apply_sam_predictions_batch()
    │   │
    │   ├─> Convert QImage to numpy, run Ultralytics SAM on worker thread
    │   └─> Returns one {segmentation: [...], score: ...} per bbox
    │
    ├─> Build temp_annotations (segmentation + class + score + source="dino")
    │
    ├─> image_label.temp_annotations = ...
    ├─> image_label.setFocus()                ← so Enter/Esc work without clicking
    └─> image_label.update()                  ← orange preview masks render

User presses Enter
    │
    └─> accept_dino_results()
        │
        ├─> For each temp annotation:
        │     - add_class(class_name) if new
        │     - image_label.annotations.setdefault(class_name, []).append(ann)
        │     - add_annotation_to_list(ann)   ← assigns per-class "number"
        │
        └─> save_current_annotations()        ← syncs to all_annotations

User presses Esc
    │
    └─> reject_dino_results() → discard temp_annotations
```

**Batch mode** (`Detect All Images`) loops over every image. In "Review before
accepting" the results land in `dino_batch_results[image_name]` and the GUI
walks the user through them image-by-image. In "Auto-accept all detections"
`_commit_dino_results()` writes directly to `all_annotations` for non-current
images; for the currently-displayed image it routes through
`image_label.annotations` so the canvas stays in sync and the next
`save_current_annotations()` doesn't overwrite the additions.

## Project Save

```
User clicks "Save" or Ctrl+S
    │
    ├─> ImageAnnotator.save_project()
    │   │
    │   ├─> Check is_loading_project flag
    │   │   (skip if loading to prevent corruption)
    │   │
    │   ├─> Build project data dict:
    │   │   {
    │   │     "images": all_images,
    │   │     "image_paths": image_paths,
    │   │     "classes": list(class_mapping.keys()),
    │   │     "class_colors": class_colors,
    │   │     "annotations": all_annotations,
    │   │     "image_dimensions": image_dimensions,
    │   │     "image_shapes": image_shapes
    │   │   }
    │   │
    │   ├─> json.dump(project_data, file)
    │   │
    │   └─> Show success message (if show_message=True)
    │
    └─> Return
```

## Project Load

```
User clicks "Open" or Ctrl+O
    │
    ├─> Select .json file via QFileDialog
    │
    ├─> ImageAnnotator.load_project_data()
    │   │
    │   ├─> Set is_loading_project = True
    │   │   (disable autosave during load)
    │   │
    │   ├─> Parse JSON file
    │   │
    │   ├─> Load images:
    │   │   │
    │   │   ├─> For each image_path:
    │   │   │   │
    │   │   │   ├─> Check if multi-dimensional (TIFF/CZI)
    │   │   │   │   │
    │   │   │   │   ├─> Extract slices
    │   │   │   │   │
    │   │   │   │   └─> Store in image_slices
    │   │   │   │
    │   │   │   └─> Load as QImage for regular images
    │   │   │
    │   │   └─> Update all_images list
    │   │
    │   ├─> Load classes and colors
    │   │   │
    │   │   └─> Populate class list widget
    │   │
    │   ├─> Load annotations
    │   │   │
    │   │   ├─> all_annotations = project_data["annotations"]
    │   │   │
    │   │   └─> Update annotation list widget
    │   │
    │   ├─> Display first image
    │   │
    │   ├─> Set is_loading_project = False
    │   │
    │   └─> Show success message
    │
    └─> Return
```

## Multi-dimensional Image Loading

```
User adds TIFF stack
    │
    ├─> ImageAnnotator.add_images()
    │   │
    │   ├─> Detect .tif/.tiff extension
    │   │
    │   ├─> TiffFile(path).asarray()
    │   │   │
    │   │   └─> shape = (10, 50, 3, 512, 512)
    │   │
    │   ├─> Show DimensionDialog
    │   │   │
    │   │   ├─> User assigns: T, Z, C, _, H, W
    │   │   │   (for each dimension)
    │   │   │
    │   │   └─> dimension_string = "TZCHW"
    │   │
    │   ├─> Extract slices:
    │   │   │
    │   │   ├─> For each T, Z, C combination:
    │   │   │   │
    │   │   │   ├─> Extract 2D slice
    │   │   │   │
    │   │   │   ├─> Convert to QImage
    │   │   │   │
    │   │   │   ├─> Name: "file_T0_Z5_C0"
    │   │   │   │
    │   │   │   └─> Store in image_slices[filename]
    │   │   │
    │   │   └─> Display first slice
    │   │
    │   └─> Store dimension metadata
    │       (image_dimensions, image_shapes)
    │
User navigates slices (Up/Down arrows)
    │
    ├─> ImageLabel.keyPressEvent()
    │   │
    │   ├─> Get slice list for current stack
    │   │
    │   ├─> current_slice_index += 1 or -1
    │   │
    │   ├─> Load new slice QImage
    │   │
    │   ├─> Load annotations for this slice
    │   │   (from all_annotations[slice_name])
    │   │
    │   └─> update() to display
    │
    └─> Return
```

## Export to YOLO

```
User clicks "Export" > "YOLO v8/v11"
    │
    ├─> Select output directory
    │
    ├─> Prompt for validation split % (QInputDialog, default 20, 0 = all train)
    │       assign_train_val() deterministically partitions the annotated
    │       images via a stable filename hash; the val count is exact so a
    │       requested split is never silently empty (issue #83)
    │
    ├─> export_yolo_v5plus(all_annotations, class_mapping, ..., val_split)
    │   │
    │   ├─> Create directory structure:
    │   │   output_dir/
    │   │   ├── data.yaml
    │   │   ├── images/
    │   │   │   ├── train/
    │   │   │   └── val/
    │   │   └── labels/
    │   │       ├── train/
    │   │       └── val/
    │   │
    │   ├─> For each annotated image:
    │   │   │
    │   │   ├─> Copy image to the train or val split it was assigned to
    │   │   │   (val_split == 0 -> everything in train, the original behaviour)
    │   │   │
    │   │   ├─> Convert annotations to YOLO format:
    │   │   │   │
    │   │   │   ├─> For polygon: compute bounding box
    │   │   │   │   class_id x_center y_center width height
    │   │   │   │   (normalized to 0-1)
    │   │   │   │
    │   │   │   └─> Write to labels/image_name.txt
    │   │   │
    │   │   └─> Next image
    │   │
    │   ├─> Write data.yaml:
    │   │   train: images/train
    │   │   val: images/val
    │   │   nc: num_classes
    │   │   names: [class1, class2, ...]
    │   │
    │   └─> Show success message
    │
    └─> Return
```

## SAM Fine-Tuning (annotate → train → use)

See [ADR-021](09_architecture_decisions.md#adr-021-sam-fine-tuning-via-a-custom-loop-over-the-ultralytics-sam2-module).

```
User: SAM Fine-Tune (beta) > Train on Current Project…
    │
    ├─> build_groups_from_project(all_annotations, image_paths, slices, image_slices)
    │       polygons/bboxes → SampleGroup(image_loader, specs, name)   (masks rasterised lazily)
    │
    ├─> _gpu_gate(): resolve_torch_device(); if "cpu" → warn + let user back out
    │
    ├─> SAMTrainConfigDialog: base model, epochs, PEAK lr, batch, prompt (bbox/point),
    │                          train split %, early-stop patience, warmup→cosine toggle,
    │                          "also fine-tune image encoder?"  (OK disabled at 0% train)
    │
    ├─> deactivate_sam_tools() + lock SAM inference UI (tools, selector, menu)
    │       trainer loads its OWN SAM instance; locking avoids a 2nd model on the same CUDA context
    │
    └─> SAMTrainingThread → SAMFineTuner.train(...)
            │  split_groups(train_pct, seed) → train/val (deterministic; empty val at 100%)
            │  build predictor (one warmup predict), pin device, apply freeze policy
            │  LambdaLR(warmup_cosine_lambda(total_steps)) when the schedule is on
            │  for each epoch:
            │     train pass / image / instance:
            │        _image_instance_losses(train=True): set_image → get_im_features,
            │        prompt_inference(bbox|point) → focal+dice loss → backward
            │        AdamW step (every batch_size images) → scheduler.step()
            │     val pass (no_grad, net.eval()): _validation_loss over held-out images
            │     log {train_loss, val_loss, lr}; EarlyStopper(patience) on val_loss
            │        → snapshot best-val weights; stop early if patience exceeded
            │     progress_signal → TrainingInfoDialog (Stop supported)
            │  save {"model": best_state | last_state} as <name>_<base_token>.pt → reload-verify via SAM()
            │
            └─> training_finished: register in SAMUtils.custom_models,
                add "★ <name>" to the SAM selector and select it
                → SAM-box / SAM-points now use the fine-tuned model

Offline variant: "Prepare SAM Dataset…" → export_sam_dataset (images/ + manifest.json),
then "Train from Dataset Folder…" → build_groups_from_folder → same training path.
```

## In-app YOLO Training (annotate → train → predict)

Mirrors the SAM fine-tuning loop's "train then use" shape: a run lands in a
predictable, per-project folder and is then selectable for prediction.

```
User: YOLO (beta) > Training > Train Model
    │
    ├─> _configure_mlflow(): set MLFLOW_TRACKING_URI (file:// URI), enable the
    │       Ultralytics mlflow setting  (no link yet — just the store path line)
    │
    │   (Train dialog also collects: warmup→cosine toggle (cos_lr), peak lr0,
    │    early-stop patience. Warmup_epochs=round(0.1·epochs) and lrf=0.1 derived.)
    │
    └─> TrainingThread → YOLOTrainer.train_model(epochs, imgsz, cos_lr, lr0, lrf,
            │                                     warmup_epochs, patience)
            │  _resolve_training_yaml → temp_train.yaml (honors the train/val split)
            │  model.train(..., cos_lr, lr0, lrf, warmup_epochs, patience,
            │              project=models/yolo/custom, name=<project>)
            │     ├─ on_train_epoch_end (epoch 1): _emit_mlflow_url()
            │     │     mlflow.active_run() is set (Ultralytics started it in
            │     │     on_pretrain_routine_end) → emit mlflow_run_url(deep link)
            │     │       → YOLOController._on_mlflow_run_url: clickable link in
            │     │         the dialog + start MLflow UI server once + open browser
            │     ├─ on_train_epoch_end: train-loss line → TrainingInfoDialog
            │     └─ on_fit_epoch_end (after validation): val_loss + mAP50 +
            │           mAP50-95 + lr line → TrainingInfoDialog
            │           (trainer.metrics; native MLflow callback logs them too)
            │  _register_trained_model(): from trainer.best (fallback save_dir),
            │     write sibling data.yaml (class names) → last_saved_model_path
            │     _prune_run_artifacts(): if the run was MLflow-tracked, delete
            │       everything except best.pt + data.yaml — Ultralytics' MLflow
            │       callback already logged the full run dir (weights + plots +
            │       csv) into the run, so the local diagnostics are redundant.
            │       (Not tracked → keep the whole folder; it lives nowhere else.)
            │
            └─> training_finished: report the saved best.pt path in the dialog.
                Prediction > Load Model lists it via list_custom_yolo_models()
                ("★ <project>"), pre-filling model + yaml → predict.
```

Output lands in `models/yolo/custom/<project>/weights/best.pt` (Ultralytics
auto-increments on collision), **not** the default `./runs` — parallel to SAM's
`models/sam/custom`. After a tracked run the folder is pruned to `best.pt` +
`data.yaml` (the diagnostics — curves, confusion matrix, batch mosaics,
`results.csv` — remain in the MLflow run via Ultralytics' `on_train_end`
`log_artifact`). The MLflow link path reuses the SAM machinery
(`run_ui_url`, `start_mlflow_ui_server`); the only difference is YOLO reads the
run id from Ultralytics' *native* MLflow callback rather than the in-process
`MLflowTracker`.
