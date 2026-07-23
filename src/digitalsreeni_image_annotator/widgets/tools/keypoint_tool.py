"""KeypointTool — place a pose instance's keypoints in schema order (issue #35).

Left-click places the next keypoint as *visible* (v=2); right-click (or
Shift+left-click) places it *occluded* (v=1). Placement auto-finishes once all
K points are placed; Enter finishes early (remaining points become v=0, "not
labelled"); Backspace removes the last placed point; Esc discards the instance.

Like the polygon/rectangle tools, the in-progress state lives on the ImageLabel
(`current_keypoints`, `drawing_keypoints`, `keypoint_next_index`) because
`AnnotationController.finish_keypoint` reads it from there.
"""

from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QBrush, QColor, QPen

from .base import ToolHandler


class KeypointTool(ToolHandler):
    def __init__(self, label):
        super().__init__(label)
        self._hover = None  # last cursor position in image coords (for the overlay)

    # ----------------------------------------------------------- schema access
    def _schema(self):
        ctx = self.label._ctx
        if ctx is None:
            return None
        return ctx.keypoint_schema(ctx.current_class())

    def _schema_k(self, schema):
        return len(schema["names"]) if schema else 0

    # ------------------------------------------------------------- mouse hooks
    def on_mouse_press(self, event, img_pt) -> bool:
        schema = self._schema()
        if schema is None:
            return False
        if event.button() == Qt.MouseButton.RightButton:
            visibility = 1  # occluded
        elif event.button() == Qt.MouseButton.LeftButton:
            visibility = (
                1
                if event.modifiers() & Qt.KeyboardModifier.ShiftModifier
                else 2  # visible
            )
        else:
            return False

        if not self.label.drawing_keypoints:
            self.label.drawing_keypoints = True
            self.label.current_keypoints = []
            self.label.keypoint_next_index = 0

        k = self._schema_k(schema)
        if len(self.label.current_keypoints) >= k:
            return True  # already full (shouldn't happen — we auto-finish at K)

        self.label.current_keypoints.append((img_pt[0], img_pt[1], visibility))
        self.label.keypoint_next_index = len(self.label.current_keypoints)
        if len(self.label.current_keypoints) >= k:
            self.label.finishKeypointsRequested.emit()
        return True

    def on_mouse_move(self, event, img_pt) -> bool:
        # Only dispatched while the keypoint tool is active; track the cursor so
        # the overlay can show the "next: <name>" hint from the very first move.
        self._hover = img_pt
        return True

    def on_enter(self) -> bool:
        # Finish early: the controller pads the remaining points to K with v=0.
        if self.label.drawing_keypoints and self.label.current_keypoints:
            self.label.finishKeypointsRequested.emit()
            return True
        return False

    def on_escape(self) -> bool:
        if self.label.drawing_keypoints or self.label.current_keypoints:
            self.discard()
            return True
        return False

    def on_backspace(self) -> bool:
        """Remove the last placed point ('go back' in the ordered walk)."""
        if self.label.current_keypoints:
            self.label.current_keypoints.pop()
            self.label.keypoint_next_index = len(self.label.current_keypoints)
            if not self.label.current_keypoints:
                self.label.drawing_keypoints = False
            self.label.update()
            return True
        return False

    # --------------------------------------------------------------- overlay
    def paint_overlay(self, painter) -> None:
        # Only render while the keypoint tool is the active one (paintEvent calls
        # every handler's overlay); an inactive tool must draw nothing. Gating on
        # the active tool — not on drawing_keypoints — lets the "next" hint show
        # before the first point is placed (senior-review P2).
        schema = self._schema()
        if schema is None or self.label.current_tool != "keypoint":
            return
        placed = self.label.current_keypoints

        painter.save()
        painter.translate(self.label.offset_x, self.label.offset_y)
        painter.scale(self.label.zoom_factor, self.label.zoom_factor)

        class_name = self.label._ctx.current_class()
        color = self.label.class_colors.get(class_name, QColor(Qt.GlobalColor.red))
        r = 5 * self.label.ui_scale / self.label.zoom_factor

        # Skeleton among already-placed points (both endpoints placed + v>0).
        painter.setPen(QPen(color, self.label._pen_w(1.5), Qt.PenStyle.SolidLine))
        for a, b in schema.get("skeleton", []):
            if a < len(placed) and b < len(placed) and placed[a][2] > 0 and placed[b][2] > 0:
                painter.drawLine(
                    QPointF(float(placed[a][0]), float(placed[a][1])),
                    QPointF(float(placed[b][0]), float(placed[b][1])),
                )

        # Point markers: filled for visible (v=2), hollow for occluded (v=1).
        painter.setFont(self.label._overlay_font(10))
        for i, (x, y, v) in enumerate(placed):
            center = QPointF(float(x), float(y))
            painter.setPen(QPen(color, self.label._pen_w(2), Qt.PenStyle.SolidLine))
            painter.setBrush(QBrush(color) if v == 2 else Qt.BrushStyle.NoBrush)
            painter.drawEllipse(center, r, r)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawText(QPointF(float(x) + r, float(y) - r), str(i + 1))

        # "next: <name>" hint at the cursor.
        names = schema.get("names", [])
        nxt = self.label.keypoint_next_index
        if self._hover is not None and nxt < len(names):
            text_color = (
                Qt.GlobalColor.white if self.label.dark_mode else Qt.GlobalColor.black
            )
            painter.setPen(QPen(text_color, self.label._pen_w(2), Qt.PenStyle.SolidLine))
            painter.drawText(
                QPointF(float(self._hover[0]) + r, float(self._hover[1]) + r),
                f"next: {names[nxt]}",
            )

        painter.restore()

    # ------------------------------------------------------------- lifecycle
    def has_unsaved_state(self) -> bool:
        return self.label.drawing_keypoints and bool(self.label.current_keypoints)

    def commit(self) -> None:
        if self.has_unsaved_state():
            self.label.finishKeypointsRequested.emit()

    def discard(self) -> None:
        self.label.current_keypoints = []
        self.label.drawing_keypoints = False
        self.label.keypoint_next_index = 0
        self._hover = None
