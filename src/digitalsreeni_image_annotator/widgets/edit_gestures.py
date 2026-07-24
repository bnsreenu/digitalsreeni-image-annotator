"""
Edit-gesture helpers for ImageLabel (issue #46 split).

This module owns the direct-manipulation shape-editing logic for the canvas.
It has two parts:

* Module-level PURE geometry functions — ``bbox_handle_points``, ``resize_bbox``,
  ``scale_segmentation``, ``translate_segmentation``, ``scale_keypoints``,
  ``translate_keypoints``, ``sync_bbox_key``. (These were ``@staticmethod`` on
  ImageLabel named ``_bbox_handle_points`` etc.; ImageLabel keeps class-level
  ``staticmethod`` aliases so ``ImageLabel._resize_bbox(...)`` / ``label._resize_bbox(...)``
  still resolve here.)
* ``EditGestures`` — the stateful driver for the handle/interior drag of the
  single selected shape (#40), the single-keypoint drag, and the keypoint
  visibility toggle (#35). Constructed with the label; reads/writes canvas
  state via ``self.label``.

The gesture STATE fields ``bbox_edit`` and ``editing_keypoint`` live on the
ImageLabel, not here — external code, ``reset_annotation_state`` and tests reach
them as ``image_label.bbox_edit`` / ``image_label.editing_keypoint``.
EditGestures manipulates them only via ``self.label.bbox_edit = ...`` etc. All
``pyqtSignal``s stay declared on ImageLabel and are emitted through
``self.label.<signal>.emit(...)``.
"""

from PyQt6.QtCore import Qt

from ..utils import (
    calculate_bbox,
    clamp_bbox,
    clamp_keypoints,
    clamp_segmentation,
    fit_bbox_inside,
)


def bbox_handle_points(bb):
    """Handle id -> (x, y) for the 8 resize handles of an AABB
    (x0, y0, x1, y1). Shared by the selection overlay (draw) and the
    handle hit-test so the squares you see are exactly the grab targets."""
    x0, y0, x1, y1 = bb
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    return {
        "tl": (x0, y0), "tm": (cx, y0), "tr": (x1, y0),
        "ml": (x0, cy), "mr": (x1, cy),
        "bl": (x0, y1), "bm": (cx, y1), "br": (x1, y1),
    }


def resize_bbox(orig_bbox, handle, pos):
    """New [x, y, w, h] after dragging `handle` to `pos`, keeping the
    opposite edge/corner fixed. Normalised so width/height never go
    negative (dragging past the anchor flips the box) and stay >= 1."""
    x, y, w, h = orig_bbox
    x0, y0, x1, y1 = x, y, x + w, y + h
    px, py = pos
    if "l" in handle:
        x0 = px
    if "r" in handle:
        x1 = px
    if "t" in handle:
        y0 = py
    if "b" in handle:
        y1 = py
    nx, ny = min(x0, x1), min(y0, y1)
    nw, nh = max(1, abs(x1 - x0)), max(1, abs(y1 - y0))
    return [nx, ny, nw, nh]


def scale_segmentation(orig_seg, old_aabb, new_aabb):
    """Map every vertex from the old bounding box to the new one (affine
    scale), so resizing the box scales the whole polygon proportionally."""
    ox0, oy0, ox1, oy1 = old_aabb
    nx0, ny0, nx1, ny1 = new_aabb
    sx = (nx1 - nx0) / ((ox1 - ox0) or 1.0)
    sy = (ny1 - ny0) / ((oy1 - oy0) or 1.0)
    out = list(orig_seg)
    for i in range(0, len(out) - 1, 2):
        out[i] = nx0 + (orig_seg[i] - ox0) * sx
        out[i + 1] = ny0 + (orig_seg[i + 1] - oy0) * sy
    return out


def translate_segmentation(orig_seg, dx, dy):
    """Shift every vertex by (dx, dy)."""
    out = list(orig_seg)
    for i in range(0, len(out) - 1, 2):
        out[i] = orig_seg[i] + dx
        out[i + 1] = orig_seg[i + 1] + dy
    return out


def scale_keypoints(orig_kpts, old_aabb, new_aabb):
    """Affine-scale every LABELLED keypoint (x, y) from the old box to the
    new one, leaving the visibility flag untouched. Mirrors
    _scale_segmentation so a pose instance's box resizes the whole pose
    proportionally (#35). Not-labelled points (v=0) are left at (0, 0) —
    COCO/YOLO-pose expect v=0 to always carry (0, 0), and transforming a
    padding point would plant a bogus but plausible-looking coordinate."""
    ox0, oy0, ox1, oy1 = old_aabb
    nx0, ny0, nx1, ny1 = new_aabb
    sx = (nx1 - nx0) / ((ox1 - ox0) or 1.0)
    sy = (ny1 - ny0) / ((oy1 - oy0) or 1.0)
    out = list(orig_kpts)
    for i in range(0, len(out) - 2, 3):
        if orig_kpts[i + 2] <= 0:
            continue
        out[i] = nx0 + (orig_kpts[i] - ox0) * sx
        out[i + 1] = ny0 + (orig_kpts[i + 1] - oy0) * sy
    return out


def translate_keypoints(orig_kpts, dx, dy):
    """Shift every LABELLED keypoint (x, y) by (dx, dy), keeping visibility
    (#35). Not-labelled points (v=0) are left at (0, 0) — see
    _scale_keypoints for why."""
    out = list(orig_kpts)
    for i in range(0, len(out) - 2, 3):
        if orig_kpts[i + 2] <= 0:
            continue
        out[i] = orig_kpts[i] + dx
        out[i + 1] = orig_kpts[i + 1] + dy
    return out


def sync_bbox_key(ann):
    """Keep a stored bbox key consistent with an edited segmentation.
    Imported annotations carry both (the bbox feeds export / SAM training);
    drawn shapes have no bbox key and are left untouched."""
    if ann.get("bbox") is not None and ann.get("segmentation"):
        ann["bbox"] = calculate_bbox(ann["segmentation"])


class EditGestures:
    """Stateful driver for direct-manipulation shape editing on the canvas.

    Owns the gesture logic; the state fields ``bbox_edit`` and
    ``editing_keypoint`` live on the ImageLabel (``self.label``), which also
    declares all the pyqtSignals emitted here."""

    def __init__(self, image_label):
        self.label = image_label

    # --- Direct-manipulation shape editing via the selection handles (#40) ---

    def _single_selected_shape(self):
        """The selected annotation iff exactly one shape is selected and it has
        a bounding box (segmentation or bbox) — the state in which the handles
        are draggable. Returns the highlighted entry as-is (its geometry is all
        the hover cursor + handle hit-test need); the press handler resolves it
        to the live object via ``_live_annotation`` before mutating. None for a
        multi-selection. Cheap — no scan, so it's safe per hover frame."""
        if len(self.label.highlighted_annotations) != 1:
            return None
        sel = self.label.highlighted_annotations[0]
        return sel if self.label._annotation_bbox(sel) is not None else None

    def _live_annotation(self, annotation):
        """Resolve a (possibly list-copied) annotation to the live object inside
        ``self.annotations`` — what the canvas renders and
        ``save_current_annotations`` persists. A list-driven selection stores
        ``item.data(UserRole)``, which PyQt round-trips as a *copy*, so mutating
        that copy in place would be lost. Match by value-equality; identity is
        unstable (ADR-022). Only the drag-arm path pays this scan, not hover."""
        for anns in self.label.annotations.values():
            for ann in anns:
                if ann == annotation:
                    return ann
        return annotation

    def _bbox_handle_at(self, annotation, pos):
        """Handle id under pos for a shape's bounding box, or None. Grab radius
        is zoom-compensated so it stays a constant on-screen target."""
        bb = self.label._annotation_bbox(annotation)
        if bb is None:
            return None
        radius = 8 * self.label.ui_scale / max(self.label.zoom_factor, 1e-6)
        for hid, point in bbox_handle_points(bb).items():
            if self.label.distance(pos, point) <= radius:
                return hid
        return None

    def _begin_shape_edit(self, live, mode, handle, pos):
        """Arm a resize/move of one selected shape. `kind` picks the geometry
        the handles drive: a polygon ("seg") scales/translates its vertices, a
        box-only annotation ("bbox") edits [x, y, w, h]. orig_bbox is the AABB
        used as the resize reference for both."""
        bb = self.label._annotation_bbox(live)
        if live.get("segmentation"):
            kind = "seg"
        elif live.get("keypoints"):
            kind = "kpt"  # the box transforms the whole pose instance (#35)
        else:
            kind = "bbox"
        # Capture the pre-gesture state for undo now: the drag mutates the
        # annotation in place, so the controller can't snapshot a clean
        # "before" at commit time (ADR-026).
        self.label.editBaselineRequested.emit()
        self.label.bbox_edit = {
            "annotation": live,
            "mode": mode,
            "handle": handle,
            "kind": kind,
            "orig_bbox": [bb[0], bb[1], bb[2] - bb[0], bb[3] - bb[1]],
            "orig_seg": list(live["segmentation"]) if kind == "seg" else None,
            "orig_kpts": list(live["keypoints"]) if kind == "kpt" else None,
            "start_pos": pos,
            "moved": False,
        }

    def _update_select_cursor(self, pos):
        """Hover feedback in select mode: resize cursors over a selected shape's
        handles, a move cursor over its interior, arrow otherwise."""
        shape = self._single_selected_shape()
        if shape is not None:
            if self._keypoint_at(shape, pos) is not None:
                # Over a draggable keypoint of the selected pose instance (#35).
                self.label.setCursor(Qt.CursorShape.PointingHandCursor)
                return
            handle = self._bbox_handle_at(shape, pos)
            if handle is not None:
                self.label.setCursor(self.label._BBOX_HANDLE_CURSORS[handle])
                return
            if self.label._annotation_contains(shape, pos):
                self.label.setCursor(Qt.CursorShape.SizeAllCursor)
                return
        self.label.setCursor(Qt.CursorShape.ArrowCursor)

    def _update_bbox_drag(self, pos):
        """Advance an in-progress shape resize/move, mutating the annotation's
        geometry in place so the canvas + selection overlay redraw live."""
        edit = self.label.bbox_edit
        if edit is None:
            return
        if edit["mode"] == "pending_move":
            # Promote to a real move only once the drag clears the click
            # threshold, so a plain click still falls through to selection
            # (e.g. picking a smaller mask nested in the shape).
            threshold = 3.0 / max(self.label.zoom_factor, 1e-6)
            if self.label.distance(pos, edit["start_pos"]) < threshold:
                return
            edit["mode"] = "move"
        ann = edit["annotation"]
        if edit["mode"] == "resize":
            new_box = resize_bbox(edit["orig_bbox"], edit["handle"], pos)
            if edit["kind"] == "seg":
                ox, oy, ow, oh = edit["orig_bbox"]
                nx, ny, nw, nh = new_box
                ann["segmentation"] = scale_segmentation(
                    edit["orig_seg"], (ox, oy, ox + ow, oy + oh),
                    (nx, ny, nx + nw, ny + nh),
                )
                sync_bbox_key(ann)
            elif edit["kind"] == "kpt":
                ox, oy, ow, oh = edit["orig_bbox"]
                nx, ny, nw, nh = new_box
                ann["keypoints"] = scale_keypoints(
                    edit["orig_kpts"], (ox, oy, ox + ow, oy + oh),
                    (nx, ny, nx + nw, ny + nh),
                )
                ann["bbox"] = new_box
            else:
                ann["bbox"] = new_box
            edit["moved"] = True
        elif edit["mode"] == "move":
            dx, dy = pos[0] - edit["start_pos"][0], pos[1] - edit["start_pos"][1]
            if edit["kind"] == "seg":
                ann["segmentation"] = translate_segmentation(
                    edit["orig_seg"], dx, dy
                )
                sync_bbox_key(ann)
            elif edit["kind"] == "kpt":
                ann["keypoints"] = translate_keypoints(
                    edit["orig_kpts"], dx, dy
                )
                x, y, w, h = edit["orig_bbox"]
                ann["bbox"] = [x + dx, y + dy, w, h]
            else:
                x, y, w, h = edit["orig_bbox"]
                ann["bbox"] = [x + dx, y + dy, w, h]
            edit["moved"] = True

    def _clamp_edited_shape(self, ann, edit, width, height):
        """Clamp a just-edited shape into the image. A move slides the intact
        shape back inside (size/shape preserving); a resize trims to the border.
        clamp_segmentation is the final safety net guaranteeing the bounds even
        for an oversized shape that a translate can't fully fit."""
        if edit["kind"] == "seg":
            seg = ann["segmentation"]
            if edit["mode"] == "move":
                bb = self.label._annotation_bbox(ann)
                fitted = fit_bbox_inside(
                    [bb[0], bb[1], bb[2] - bb[0], bb[3] - bb[1]], width, height
                )
                seg = translate_segmentation(
                    seg, fitted[0] - bb[0], fitted[1] - bb[1]
                )
            ann["segmentation"] = clamp_segmentation(seg, width, height)
            sync_bbox_key(ann)
            # The polygon was reshaped — its old dense "raw" (issue #24) no
            # longer describes it, so a later Detail %=100 must not revert this
            # edit. Reset the simplification baseline to the edited geometry.
            if ann.get("segmentation_raw"):
                ann.pop("segmentation_raw", None)
                ann["detail_pct"] = 100
        elif edit["kind"] == "kpt":
            # Slide the intact pose back inside on a move, then clamp points +
            # box to the border as the final safety net (ADR-024). (#35)
            if edit["mode"] == "move":
                fitted = fit_bbox_inside(ann["bbox"], width, height)
                dx, dy = fitted[0] - ann["bbox"][0], fitted[1] - ann["bbox"][1]
                ann["keypoints"] = translate_keypoints(ann["keypoints"], dx, dy)
                ann["bbox"] = fitted
            ann["keypoints"] = clamp_keypoints(ann["keypoints"], width, height)
            ann["bbox"] = clamp_bbox(ann["bbox"], width, height)
        elif edit["mode"] == "move":
            ann["bbox"] = fit_bbox_inside(ann["bbox"], width, height)
        else:
            ann["bbox"] = clamp_bbox(ann["bbox"], width, height)

    def _commit_bbox_drag(self, pos, event):
        """Finish a shape drag: clamp into the image and persist if it actually
        moved; otherwise treat the press as a plain click → select."""
        edit = self.label.bbox_edit
        self.label.bbox_edit = None
        if edit is None:
            return
        if edit["moved"]:
            ann = edit["annotation"]
            if self.label.original_pixmap is not None:
                self._clamp_edited_shape(
                    ann, edit, self.label.original_pixmap.width(),
                    self.label.original_pixmap.height(),
                )
            # Point the selection at the live, mutated object so the controller
            # can re-select it by value-equality after the list rebuild. A no-op
            # for a canvas-click selection (already the live object); the real
            # work is replacing a stale list-selection copy.
            self.label.highlighted_annotations = [ann]
            self.label.bboxEditCommitted.emit()
        else:
            # No drag happened — behave exactly like an idle click so
            # nested-mask click-through keeps working.
            self.label.selection_origin = edit["start_pos"]
            self.label.selecting = False
            self.label.selection_rect = None
            self.label._finish_selection(pos, event)

    def _cancel_bbox_drag(self):
        """Escape during a shape drag: restore the original geometry, drop it."""
        edit = self.label.bbox_edit
        self.label.bbox_edit = None
        if edit is not None:
            ann = edit["annotation"]
            if edit["kind"] == "seg":
                ann["segmentation"] = list(edit["orig_seg"])
                sync_bbox_key(ann)
            elif edit["kind"] == "kpt":
                ann["keypoints"] = list(edit["orig_kpts"])
                x, y, w, h = edit["orig_bbox"]
                ann["bbox"] = [x, y, w, h]
            else:
                x, y, w, h = edit["orig_bbox"]
                ann["bbox"] = [x, y, w, h]
        self.label.update()

    # --- Single-keypoint editing for a selected pose instance (#35) ---

    def _keypoint_at(self, annotation, pos):
        """Index of the labelled (v>0) keypoint under pos, or None. Generous,
        zoom-compensated grab radius so individual points stay easy to grab."""
        if annotation is None:
            return None
        kps = annotation.get("keypoints")
        if not kps:
            return None
        radius = 8 * self.label.ui_scale / max(self.label.zoom_factor, 1e-6)
        best, best_d = None, None
        for i, (x, y, v) in enumerate(zip(kps[0::3], kps[1::3], kps[2::3])):
            if v <= 0:
                continue
            d = self.label.distance(pos, (x, y))
            if d <= radius and (best is None or d < best_d):
                best, best_d = i, d
        return best

    def _begin_keypoint_edit(self, live, index, pos):
        """Arm a single-point drag; capture the undo baseline now (ADR-026)."""
        kps = live.get("keypoints") or []
        self.label.editBaselineRequested.emit()
        self.label.editing_keypoint = {
            "annotation": live,
            "index": index,
            "orig": (kps[3 * index], kps[3 * index + 1]),
            "moved": False,
        }

    def _update_keypoint_drag(self, pos):
        edit = self.label.editing_keypoint
        if edit is None:
            return
        ann = edit["annotation"]
        i = edit["index"]
        ann["keypoints"][3 * i] = pos[0]
        ann["keypoints"][3 * i + 1] = pos[1]
        edit["moved"] = True

    def _commit_keypoint_drag(self, pos, event):
        edit = self.label.editing_keypoint
        self.label.editing_keypoint = None
        if edit is None:
            return
        if edit["moved"]:
            ann = edit["annotation"]
            if self.label.original_pixmap is not None:
                ann["keypoints"] = clamp_keypoints(
                    ann["keypoints"],
                    self.label.original_pixmap.width(),
                    self.label.original_pixmap.height(),
                )
            self.label.highlighted_annotations = [ann]
            self.label.keypointEditCommitted.emit()  # save + push undo baseline
        self.label.update()

    def _cancel_keypoint_drag(self):
        edit = self.label.editing_keypoint
        self.label.editing_keypoint = None
        if edit is not None:
            ann = edit["annotation"]
            i = edit["index"]
            ann["keypoints"][3 * i] = edit["orig"][0]
            ann["keypoints"][3 * i + 1] = edit["orig"][1]
        self.label.update()

    def _toggle_keypoint_visibility(self, live, index):
        """Right-click a committed point: toggle visible(2) <-> occluded(1),
        keeping its position. Padded not-labelled points (v=0) are left as-is."""
        kps = live.get("keypoints")
        if not kps:
            return
        v = kps[3 * index + 2]
        if v <= 0:
            return
        self.label.editBaselineRequested.emit()
        kps[3 * index + 2] = 1 if v == 2 else 2
        # Point the selection at the live, mutated object (mirrors
        # _commit_keypoint_drag / _commit_bbox_drag) so a list-selected
        # (UserRole copy) instance doesn't go stale after the list rebuild.
        self.label.highlighted_annotations = [live]
        self.label.keypointEditCommitted.emit()
        self.label.update()
