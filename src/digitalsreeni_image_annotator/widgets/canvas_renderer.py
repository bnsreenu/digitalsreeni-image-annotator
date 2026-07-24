"""
Canvas rendering for ImageLabel (issue #46 split).

``CanvasRenderer`` owns the paint-layer drawing routines for the annotation
canvas: committed annotations, keypoint/pose instances, the selection overlay,
the rubber-band selection rect, the in-progress editing polygon, temp
(DINO/YOLO) annotations, the SAM bbox, the paint/eraser size indicator, and the
overlay painter helpers (pen width, overlay font, centroid).

All canvas STATE (annotations, zoom/offset, class colours, selection, temp
annotations, SAM state, …) lives on the ImageLabel; CanvasRenderer reads it via
``self.label`` and never owns any of it. ``paintEvent`` stays on ImageLabel and
calls into these methods in the exact same layer order as before the split.
"""

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QBrush, QColor, QFont, QPen, QPolygonF


class CanvasRenderer:
    """Draws the ImageLabel canvas layers. State lives on ``self.label``."""

    # Selection highlight: a semi-transparent selection-blue, drawn as the
    # dashed bounding-box marquee (handles use the opaque variant). Class-
    # colour-independent, so it never vanishes the way the old red-on-red
    # highlight did. Re-exported on ImageLabel as ImageLabel._SELECTION_COLOR.
    _SELECTION_COLOR = QColor(0, 120, 215, 220)

    def __init__(self, image_label):
        self.label = image_label

    def _pen_w(self, base):
        """Overlay pen width: ui-scaled, zoom-compensated (constant on screen)."""
        return base * self.label.ui_scale / self.label.zoom_factor

    def _overlay_font(self, base=12):
        """Overlay label font: ui-scaled, zoom-compensated (constant on screen)."""
        return QFont("Arial", max(1, int(base * self.label.ui_scale / self.label.zoom_factor)))

    def draw_temp_annotations(self, painter):
        painter.save()
        painter.translate(self.label.offset_x, self.label.offset_y)
        painter.scale(self.label.zoom_factor, self.label.zoom_factor)

        for annotation in self.label.temp_annotations:
            color = QColor(255, 165, 0, 128)  # Semi-transparent orange
            painter.setPen(QPen(color, self._pen_w(2), Qt.PenStyle.DashLine))
            painter.setBrush(QBrush(color))

            # Prefer segmentation polygon over bbox when both are present
            # (DINO+SAM temp annotations carry both — the polygon is the mask).
            points = None
            if "segmentation" in annotation:
                points = [
                    QPointF(float(x), float(y))
                    for x, y in zip(
                        annotation["segmentation"][0::2],
                        annotation["segmentation"][1::2],
                    )
                ]
                painter.drawPolygon(QPolygonF(points))
            elif "bbox" in annotation:
                x, y, w, h = annotation["bbox"]
                painter.drawRect(QRectF(x, y, w, h))

            # Draw label and score
            painter.setFont(self._overlay_font())
            label = f"{annotation['category_name']} {annotation['score']:.2f}"
            if points is not None:
                centroid = self.calculate_centroid(points)
                if centroid:
                    painter.drawText(centroid, label)
            elif "bbox" in annotation:
                x, y, _, _ = annotation["bbox"]
                painter.drawText(QPointF(x, y - 5), label)

        painter.restore()

    def draw_tool_size_indicator(self, painter):
        if self.label.current_tool in ["paint_brush", "eraser"] and hasattr(
            self.label, "cursor_pos"
        ):
            painter.save()
            painter.translate(self.label.offset_x, self.label.offset_y)
            painter.scale(self.label.zoom_factor, self.label.zoom_factor)

            if self.label.current_tool == "paint_brush":
                size = self.label._ctx.paint_brush_size()
                color = QColor(255, 0, 0, 128)  # Semi-transparent red
            else:  # eraser
                size = self.label._ctx.eraser_size()
                color = QColor(0, 0, 255, 128)  # Semi-transparent blue

            # Draw filled circle with lower opacity
            painter.setOpacity(0.3)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(color)
            painter.drawEllipse(
                QPointF(self.label.cursor_pos[0], self.label.cursor_pos[1]), size, size
            )

            # Draw circle outline with full opacity
            painter.setOpacity(1.0)
            painter.setPen(QPen(color.darker(150), self._pen_w(1), Qt.PenStyle.SolidLine))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(
                QPointF(self.label.cursor_pos[0], self.label.cursor_pos[1]), size, size
            )

            # Draw size text
            # Reset the transform to ensure text is drawn at screen coordinates
            painter.resetTransform()
            font = QFont()
            # Screen-space text (transform was reset above): scale with
            # the UI font setting only, not with image zoom.
            font.setPointSize(max(1, int(10 * self.label.ui_scale)))
            painter.setFont(font)
            painter.setPen(QPen(Qt.GlobalColor.black))  # Use black color for better visibility

            # Convert cursor position back to screen coordinates
            screen_x = self.label.cursor_pos[0] * self.label.zoom_factor + self.label.offset_x
            screen_y = self.label.cursor_pos[1] * self.label.zoom_factor + self.label.offset_y

            # Position text above the circle
            text_rect = QRectF(
                screen_x + (size * self.label.zoom_factor),
                screen_y - (size * self.label.zoom_factor),
                100,
                20,
            )

            text = f"Size: {size}"
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, text)

            painter.restore()

    def draw_sam_bbox(self, painter):
        painter.save()
        painter.translate(self.label.offset_x, self.label.offset_y)
        painter.scale(self.label.zoom_factor, self.label.zoom_factor)
        painter.setPen(QPen(Qt.GlobalColor.red, self._pen_w(2), Qt.PenStyle.SolidLine))
        x1, y1, x2, y2 = self.label.sam_bbox
        painter.drawRect(QRectF(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1)))
        painter.restore()

    def draw_selection_rect(self, painter):
        """Draw the idle-mode rubber-band selection rectangle (issue #75).

        A single dashed selection-blue rect with a faint fill — same restrained
        style as the selection outline (not red, which clashes with class colours)."""
        painter.save()
        painter.translate(self.label.offset_x, self.label.offset_y)
        painter.scale(self.label.zoom_factor, self.label.zoom_factor)
        x0, y0, x1, y1 = self.label.selection_rect
        rect = QRectF(min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0))
        fill = QColor(self._SELECTION_COLOR)
        fill.setAlphaF(0.10)
        painter.setBrush(QBrush(fill))
        painter.setPen(QPen(self._SELECTION_COLOR, self._pen_w(1), Qt.PenStyle.DashLine))
        painter.drawRect(rect)
        painter.restore()

    def draw_annotations(self, painter):
        """Draw all annotations on the image."""
        if not self.label.original_pixmap:
            return

        painter.save()
        painter.translate(self.label.offset_x, self.label.offset_y)
        painter.scale(self.label.zoom_factor, self.label.zoom_factor)

        for class_name, class_annotations in self.label.annotations.items():
            if not self.label._ctx.is_class_visible(class_name):
                continue

            color = self.label.class_colors.get(class_name, QColor(Qt.GlobalColor.white))
            for annotation in class_annotations:
                # Selection no longer recolours the mask (it used to turn red,
                # which was invisible on a red-class mask). The mask always
                # keeps its class colour; selection is drawn as a
                # class-colour-independent overlay in a final pass below.
                border_color = color
                fill_color = QColor(color)
                fill_color.setAlphaF(self.label.fill_opacity)

                text_color = Qt.GlobalColor.white if self.label.dark_mode else Qt.GlobalColor.black
                painter.setPen(QPen(border_color, self._pen_w(2), Qt.PenStyle.SolidLine))
                painter.setBrush(QBrush(fill_color))

                if "segmentation" in annotation:
                    segmentation = annotation["segmentation"]
                    if isinstance(segmentation, list) and len(segmentation) > 0:
                        if isinstance(segmentation[0], list):  # Multiple polygons
                            for polygon in segmentation:
                                points = [
                                    QPointF(float(x), float(y))
                                    for x, y in zip(polygon[0::2], polygon[1::2])
                                ]
                                if points:
                                    painter.drawPolygon(QPolygonF(points))
                        else:  # Single polygon
                            points = [
                                QPointF(float(x), float(y))
                                for x, y in zip(segmentation[0::2], segmentation[1::2])
                            ]
                            if points:
                                painter.drawPolygon(QPolygonF(points))

                        # Draw centroid and label
                        if points:
                            centroid = self.calculate_centroid(points)
                            if centroid:
                                painter.setFont(self._overlay_font())
                                painter.setPen(
                                    QPen(text_color, self._pen_w(2), Qt.PenStyle.SolidLine)
                                )
                                painter.drawText(
                                    centroid,
                                    f"{class_name} {annotation.get('number', '')}",
                                )

                elif "keypoints" in annotation:
                    # Pose instance (#35): skeleton + visibility-coloured points.
                    # Drawn before the bbox branch since an instance also carries
                    # a bbox (the box is resizable via the selection handles).
                    self._draw_keypoint_annotation(
                        painter, annotation, class_name, color, text_color
                    )

                elif "bbox" in annotation:
                    x, y, width, height = annotation["bbox"]
                    painter.drawRect(QRectF(x, y, width, height))
                    painter.setPen(QPen(text_color, self._pen_w(2), Qt.PenStyle.SolidLine))
                    painter.drawText(
                        QPointF(x, y), f"{class_name} {annotation.get('number', '')}"
                    )

        # Polygon-in-progress is rendered by PolygonTool.paint_overlay
        # (paintEvent calls active_tool_handler.paint_overlay).

        # Draw temporary SAM prediction
        if self.label.temp_sam_prediction:
            temp_color = QColor(255, 165, 0, 128)  # Semi-transparent orange
            painter.setPen(QPen(temp_color, self._pen_w(2), Qt.PenStyle.DashLine))
            painter.setBrush(QBrush(temp_color))

            segmentation = self.label.temp_sam_prediction["segmentation"]
            points = [
                QPointF(float(x), float(y))
                for x, y in zip(segmentation[0::2], segmentation[1::2])
            ]
            if points:
                painter.drawPolygon(QPolygonF(points))
                centroid = self.calculate_centroid(points)
                if centroid:
                    painter.setFont(self._overlay_font())
                    painter.drawText(
                        centroid, f"SAM: {self.label.temp_sam_prediction['score']:.2f}"
                    )

        # Selection overlay — drawn LAST so it sits on top of every mask's
        # fill, and in a class-colour-independent style so it's recognisable
        # regardless of the selected mask's colour (issue #75 follow-up).
        for annotation in self.label.highlighted_annotations:
            self._draw_selection_overlay(painter, annotation)

        painter.restore()

    def _draw_keypoint_annotation(self, painter, annotation, class_name, color, text_color):
        """Render a committed pose instance (#35): a faint instance box, the
        skeleton edges (between labelled points), visibility-coloured markers
        (filled = visible v2, hollow = occluded v1, v0 skipped), and the label.
        Marker/skeleton geometry matches the in-progress KeypointTool overlay so
        placing and reviewing look the same."""
        kps = annotation.get("keypoints") or []
        pts = list(zip(kps[0::3], kps[1::3], kps[2::3]))
        schema = self.label._ctx.keypoint_schema(class_name) if self.label._ctx else None
        r = 4 * self.label.ui_scale / self.label.zoom_factor

        # Faint instance box so the resizable bounds are visible.
        bbox = annotation.get("bbox")
        if bbox:
            x, y, w, h = bbox
            painter.setPen(QPen(color, self._pen_w(1), Qt.PenStyle.DotLine))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(QRectF(float(x), float(y), float(w), float(h)))

        if schema:
            painter.setPen(QPen(color, self._pen_w(1.5), Qt.PenStyle.SolidLine))
            for a, b in schema.get("skeleton", []):
                if a < len(pts) and b < len(pts) and pts[a][2] > 0 and pts[b][2] > 0:
                    painter.drawLine(
                        QPointF(float(pts[a][0]), float(pts[a][1])),
                        QPointF(float(pts[b][0]), float(pts[b][1])),
                    )

        for x, y, v in pts:
            if v <= 0:
                continue  # not labelled — nothing to draw
            painter.setPen(QPen(color, self._pen_w(2), Qt.PenStyle.SolidLine))
            painter.setBrush(QBrush(color) if v == 2 else Qt.BrushStyle.NoBrush)
            painter.drawEllipse(QPointF(float(x), float(y)), r, r)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        # Instance label at the box top-left (fallback to first point).
        if bbox:
            anchor = QPointF(float(bbox[0]), float(bbox[1]))
        elif pts:
            anchor = QPointF(float(pts[0][0]), float(pts[0][1]))
        else:
            return
        painter.setFont(self._overlay_font())
        painter.setPen(QPen(text_color, self._pen_w(2), Qt.PenStyle.SolidLine))
        painter.drawText(anchor, f"{class_name} {annotation.get('number', '')}")

    def _draw_selection_overlay(self, painter, annotation):
        """Mark a selected annotation the way the sibling open-garden-planner
        app does: a dashed selection-blue bounding box plus bright square
        handles at the 4 corners and 4 edge midpoints. Class-colour-independent
        and clearly visible regardless of the mask's own colour."""
        if not self.label._is_class_pickable(annotation.get("category_name")):
            return  # don't draw selection chrome over a hidden mask
        bb = self.label._annotation_bbox(annotation)
        if bb is None:
            return
        x0, y0, x1, y1 = bb
        rect = QRectF(x0, y0, x1 - x0, y1 - y0)

        # Dashed bounding-box marquee.
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(self._SELECTION_COLOR, self._pen_w(1.5), Qt.PenStyle.DashLine))
        painter.drawRect(rect)

        # Handle squares — opaque blue with a white casing so they read on any
        # background; fixed on-screen size (zoom-compensated). For a bbox these
        # are grab targets for resize (issue #40); for a polygon they are visual
        # markers only (vertex editing happens via double-click).
        half = 4 * self.label.ui_scale / self.label.zoom_factor
        painter.setPen(QPen(Qt.GlobalColor.white, self._pen_w(1), Qt.PenStyle.SolidLine))
        painter.setBrush(QBrush(QColor(0, 120, 215)))
        for hx, hy in self.label._bbox_handle_points(bb).values():
            painter.drawRect(QRectF(hx - half, hy - half, 2 * half, 2 * half))

    def draw_editing_polygon(self, painter):
        """Draw the polygon being edited."""
        painter.save()
        painter.translate(self.label.offset_x, self.label.offset_y)
        painter.scale(self.label.zoom_factor, self.label.zoom_factor)

        points = [
            QPointF(float(x), float(y))
            for x, y in zip(
                self.label.editing_polygon["segmentation"][0::2],
                self.label.editing_polygon["segmentation"][1::2],
            )
        ]
        color = self.label.class_colors.get(
            self.label.editing_polygon["category_name"], QColor(Qt.GlobalColor.white)
        )
        fill_color = QColor(color)
        fill_color.setAlphaF(self.label.fill_opacity)

        painter.setPen(QPen(color, self._pen_w(2), Qt.PenStyle.SolidLine))
        painter.setBrush(QBrush(fill_color))
        painter.drawPolygon(QPolygonF(points))  # Changed QPolygon to QPolygonF - Sreeni

        for i, point in enumerate(points):
            if i == self.label.hover_point_index:
                painter.setBrush(QColor(255, 0, 0))
            else:
                painter.setBrush(QColor(0, 255, 0))
            r = 5 * self.label.ui_scale / self.label.zoom_factor
            painter.drawEllipse(point, r, r)

        painter.restore()

    def calculate_centroid(self, points):
        """Calculate the centroid of a polygon."""
        if not points:
            return None
        x_coords = [point.x() for point in points]
        y_coords = [point.y() for point in points]
        centroid_x = sum(x_coords) / len(points)
        centroid_y = sum(y_coords) / len(points)
        return QPointF(centroid_x, centroid_y)
