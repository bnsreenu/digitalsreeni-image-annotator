"""
ImageLabel module for the Image Annotator application.

This module contains the ImageLabel class, which is responsible for
displaying the image and handling annotation interactions.

@DigitalSreeni
Dr. Sreenivas Bhattiprolu
"""

import os
import warnings

import cv2
import numpy as np
from PIL import Image
from PyQt5.QtCore import QPoint, QPointF, QRectF, QSize, Qt
from PyQt5.QtGui import (
    QBrush,
    QColor,
    QFont,
    QImage,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
    QPolygon,
    QPolygonF,
    QWheelEvent,
)
from PyQt5.QtWidgets import QApplication, QLabel, QMessageBox

warnings.filterwarnings("ignore", category=UserWarning)


class ImageLabel(QLabel):
    """
    A custom QLabel for displaying images and handling annotations.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.annotations = {}
        self.current_annotation = []
        self.temp_point = None
        self.current_tool = None
        self.zoom_factor = 1.0
        self.class_colors = {}
        self.class_visibility = {}
        self.start_point = None
        self.end_point = None
        self.highlighted_annotations = []
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.original_pixmap = None
        self.scaled_pixmap = None
        self.pan_start_pos = None
        self.main_window = None
        self.offset_x = 0
        self.offset_y = 0
        self.drawing_polygon = False
        self.editing_polygon = None
        self.editing_point_index = None
        self.hover_point_index = None
        self.fill_opacity = 0.3
        self.drawing_rectangle = False
        self.current_rectangle = None
        self.bit_depth = None
        self.image_path = None
        self.dark_mode = False

        self.paint_mask = None
        self.eraser_mask = None
        self.temp_paint_mask = None
        self.is_painting = False
        self.temp_eraser_mask = None
        self.is_erasing = False
        self.cursor_pos = None

        # SAM
        self.sam_magic_wand_active = False
        self.sam_bbox = None
        self.drawing_sam_bbox = False
        self.temp_sam_prediction = None
        self.temp_annotations = []

        # --- New for SAM modes ---
        self.sam_box_active = False
        self.sam_points_active = False
        self.sam_positive_points = []
        self.sam_negative_points = []

    def set_main_window(self, main_window):
        self.main_window = main_window

    def set_dark_mode(self, is_dark):
        self.dark_mode = is_dark
        self.update()

    def setPixmap(self, pixmap):
        """Set the pixmap and update the scaled version."""
        if isinstance(pixmap, QImage):
            pixmap = QPixmap.fromImage(pixmap)
        self.original_pixmap = pixmap
        self.update_scaled_pixmap()

    def detect_bit_depth(self):
        """Detect and store the actual image bit depth using PIL."""
        if self.image_path and os.path.exists(self.image_path):
            with Image.open(self.image_path) as img:
                if img.mode == "1":
                    self.bit_depth = 1
                elif img.mode == "L":
                    self.bit_depth = 8
                elif img.mode == "I;16":
                    self.bit_depth = 16
                elif img.mode in ["RGB", "HSV"]:
                    self.bit_depth = 24
                elif img.mode in ["RGBA", "CMYK"]:
                    self.bit_depth = 32
                else:
                    self.bit_depth = img.bits

                if self.main_window:
                    self.main_window.update_image_info()

    def update_scaled_pixmap(self):
        if self.original_pixmap and not self.original_pixmap.isNull():
            scaled_size = self.original_pixmap.size() * self.zoom_factor
            self.scaled_pixmap = self.original_pixmap.scaled(
                scaled_size.width(),
                scaled_size.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            super().setPixmap(self.scaled_pixmap)
            self.setMinimumSize(self.scaled_pixmap.size())
            self.update_offset()
        else:
            self.scaled_pixmap = None
            super().setPixmap(QPixmap())
            self.setMinimumSize(QSize(0, 0))

    def update_offset(self):
        """Update the offset for centered image display."""
        if self.scaled_pixmap:
            self.offset_x = int((self.width() - self.scaled_pixmap.width()) / 2)
            self.offset_y = int((self.height() - self.scaled_pixmap.height()) / 2)

    def reset_annotation_state(self):
        """Reset the annotation state."""
        self.temp_point = None
        self.start_point = None
        self.end_point = None

    def clear_current_annotation(self):
        """Clear the current annotation."""
        self.current_annotation = []

    def resizeEvent(self, event):
        """Handle resize events."""
        super().resizeEvent(event)
        self.update_offset()

    def start_painting(self, pos):
        if self.temp_paint_mask is None:
            self.temp_paint_mask = np.zeros(
                (self.original_pixmap.height(), self.original_pixmap.width()),
                dtype=np.uint8,
            )
        self.is_painting = True
        self.continue_painting(pos)

    def continue_painting(self, pos):
        if not self.is_painting:
            return
        brush_size = self.main_window.paint_brush_size
        cv2.circle(
            self.temp_paint_mask, (int(pos[0]), int(pos[1])), brush_size, 255, -1
        )
        self.update()

    def finish_painting(self):
        if not self.is_painting:
            return
        self.is_painting = False
        # Don't commit the annotation yet, just keep the temp_paint_mask

    def commit_paint_annotation(self):
        if self.temp_paint_mask is not None and self.main_window.current_class:
            class_name = self.main_window.current_class
            contours, _ = cv2.findContours(
                self.temp_paint_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for contour in contours:
                if cv2.contourArea(contour) > 10:  # Minimum area threshold
                    segmentation = contour.flatten().tolist()
                    new_annotation = {
                        "segmentation": segmentation,
                        "category_id": self.main_window.class_mapping[class_name],
                        "category_name": class_name,
                    }
                    self.annotations.setdefault(class_name, []).append(new_annotation)
                    self.main_window.add_annotation_to_list(new_annotation)
            self.temp_paint_mask = None
            self.main_window.save_current_annotations()
            self.main_window.update_slice_list_colors()
            self.update()

    def discard_paint_annotation(self):
        self.temp_paint_mask = None
        self.update()

    def start_erasing(self, pos):
        if self.temp_eraser_mask is None:
            self.temp_eraser_mask = np.zeros(
                (self.original_pixmap.height(), self.original_pixmap.width()),
                dtype=np.uint8,
            )
        self.is_erasing = True
        self.continue_erasing(pos)

    def continue_erasing(self, pos):
        if not self.is_erasing:
            return
        eraser_size = self.main_window.eraser_size
        cv2.circle(
            self.temp_eraser_mask, (int(pos[0]), int(pos[1])), eraser_size, 255, -1
        )
        self.update()

    def finish_erasing(self):
        if not self.is_erasing:
            return
        self.is_erasing = False
        # Don't commit the eraser changes yet, just keep the temp_eraser_mask

    def commit_eraser_changes(self):
        if self.temp_eraser_mask is not None:
            eraser_mask = self.temp_eraser_mask.astype(bool)
            current_name = (
                self.main_window.current_slice or self.main_window.image_file_name
            )
            annotations_changed = False

            for class_name, annotations in self.annotations.items():
                updated_annotations = []
                max_number = max([ann.get("number", 0) for ann in annotations] + [0])
                for annotation in annotations:
                    if "segmentation" in annotation:
                        points = (
                            np.array(annotation["segmentation"])
                            .reshape(-1, 2)
                            .astype(int)
                        )
                        mask = np.zeros_like(self.temp_eraser_mask)
                        cv2.fillPoly(mask, [points], 255)
                        mask = mask.astype(bool)
                        mask[eraser_mask] = False
                        contours, _ = cv2.findContours(
                            mask.astype(np.uint8),
                            cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE,
                        )
                        for i, contour in enumerate(contours):
                            if cv2.contourArea(contour) > 10:  # Minimum area threshold
                                new_segmentation = contour.flatten().tolist()
                                new_annotation = annotation.copy()
                                new_annotation["segmentation"] = new_segmentation
                                if i == 0:
                                    new_annotation["number"] = annotation.get(
                                        "number", max_number + 1
                                    )
                                else:
                                    max_number += 1
                                    new_annotation["number"] = max_number
                                updated_annotations.append(new_annotation)
                        if len(contours) > 1:
                            annotations_changed = True
                    else:
                        updated_annotations.append(annotation)
                self.annotations[class_name] = updated_annotations

            self.temp_eraser_mask = None

            # Update the all_annotations dictionary in the main window
            self.main_window.all_annotations[current_name] = self.annotations

            # Call update_annotation_list directly
            self.main_window.update_annotation_list()

            self.main_window.save_current_annotations()
            self.main_window.update_slice_list_colors()
            self.update()

            # print(f"Eraser changes committed. Annotations changed: {annotations_changed}")
            # print(f"Current annotations: {self.annotations}")

    def discard_eraser_changes(self):
        self.temp_eraser_mask = None
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.scaled_pixmap:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            # Draw the image
            painter.drawPixmap(
                int(self.offset_x), int(self.offset_y), self.scaled_pixmap
            )
            # Draw annotations
            self.draw_annotations(painter)
            # Draw other elements
            if self.editing_polygon:
                self.draw_editing_polygon(painter)
            if self.drawing_rectangle and self.current_rectangle:
                self.draw_current_rectangle(painter)
            if self.sam_magic_wand_active and self.sam_bbox:
                self.draw_sam_bbox(painter)
            # --- Draw for SAM-box mode ---
            if self.sam_box_active and self.sam_bbox:
                self.draw_sam_bbox(painter)
            # --- Draw for SAM-points mode ---
            if self.sam_points_active:
                painter.save()
                painter.translate(self.offset_x, self.offset_y)
                painter.scale(self.zoom_factor, self.zoom_factor)
                for pt in self.sam_positive_points:
                    painter.setPen(QPen(Qt.green, 6 / self.zoom_factor, Qt.SolidLine))
                    painter.setBrush(QBrush(Qt.green))
                    painter.drawEllipse(QPointF(pt[0], pt[1]), 4, 4)
                for pt in self.sam_negative_points:
                    painter.setPen(QPen(Qt.red, 6 / self.zoom_factor, Qt.SolidLine))
                    painter.setBrush(QBrush(Qt.red))
                    painter.drawEllipse(QPointF(pt[0], pt[1]), 4, 4)
                painter.restore()
            # Draw temporary paint mask
            if self.temp_paint_mask is not None:
                self.draw_temp_paint_mask(painter)
            if self.temp_eraser_mask is not None:
                self.draw_temp_eraser_mask(painter)
            self.draw_tool_size_indicator(painter)
            if self.temp_annotations:
                self.draw_temp_annotations(painter)
            painter.end()

    def draw_temp_annotations(self, painter):
        painter.save()
        painter.translate(self.offset_x, self.offset_y)
        painter.scale(self.zoom_factor, self.zoom_factor)

        for annotation in self.temp_annotations:
            color = QColor(255, 165, 0, 128)  # Semi-transparent orange
            painter.setPen(QPen(color, 2 / self.zoom_factor, Qt.DashLine))
            painter.setBrush(QBrush(color))

            if "bbox" in annotation:
                x, y, w, h = annotation["bbox"]
                painter.drawRect(QRectF(x, y, w, h))
            elif "segmentation" in annotation:
                points = [
                    QPointF(float(x), float(y))
                    for x, y in zip(
                        annotation["segmentation"][0::2],
                        annotation["segmentation"][1::2],
                    )
                ]
                painter.drawPolygon(QPolygonF(points))

            # Draw label and score
            painter.setFont(QFont("Arial", int(12 / self.zoom_factor)))
            label = f"{annotation['category_name']} {annotation['score']:.2f}"
            if "bbox" in annotation:
                x, y, _, _ = annotation["bbox"]
                painter.drawText(QPointF(x, y - 5), label)
            elif "segmentation" in annotation:
                centroid = self.calculate_centroid(points)
                if centroid:
                    painter.drawText(centroid, label)

        painter.restore()

    def accept_temp_annotations(self):
        for annotation in self.temp_annotations:
            class_name = annotation["category_name"]

            # Check if the class exists, if not, add it
            if class_name not in self.main_window.class_mapping:
                self.main_window.add_class(class_name)

            if class_name not in self.annotations:
                self.annotations[class_name] = []

            del annotation["temp"]
            del annotation[
                "score"
            ]  # Remove the score as it's not needed in the final annotation
            self.annotations[class_name].append(annotation)
            self.main_window.add_annotation_to_list(annotation)

        self.temp_annotations.clear()
        self.main_window.save_current_annotations()
        self.main_window.update_slice_list_colors()
        self.update()

    def discard_temp_annotations(self):
        self.temp_annotations.clear()
        self.update()

    def draw_temp_paint_mask(self, painter):
        if self.temp_paint_mask is not None:
            painter.save()
            painter.translate(self.offset_x, self.offset_y)
            painter.scale(self.zoom_factor, self.zoom_factor)

            mask_image = QImage(
                self.temp_paint_mask.data,
                self.temp_paint_mask.shape[1],
                self.temp_paint_mask.shape[0],
                self.temp_paint_mask.shape[1],
                QImage.Format_Grayscale8,
            )
            mask_pixmap = QPixmap.fromImage(mask_image)
            painter.setOpacity(0.5)
            painter.drawPixmap(0, 0, mask_pixmap)
            painter.setOpacity(1.0)

            painter.restore()

    def draw_temp_eraser_mask(self, painter):
        if self.temp_eraser_mask is not None:
            painter.save()
            painter.translate(self.offset_x, self.offset_y)
            painter.scale(self.zoom_factor, self.zoom_factor)

            mask_image = QImage(
                self.temp_eraser_mask.data,
                self.temp_eraser_mask.shape[1],
                self.temp_eraser_mask.shape[0],
                self.temp_eraser_mask.shape[1],
                QImage.Format_Grayscale8,
            )
            mask_pixmap = QPixmap.fromImage(mask_image)
            painter.setOpacity(0.5)
            painter.drawPixmap(0, 0, mask_pixmap)
            painter.setOpacity(1.0)

            painter.restore()

    def draw_tool_size_indicator(self, painter):
        if self.current_tool in ["paint_brush", "eraser"] and hasattr(
            self, "cursor_pos"
        ):
            painter.save()
            painter.translate(self.offset_x, self.offset_y)
            painter.scale(self.zoom_factor, self.zoom_factor)

            if self.current_tool == "paint_brush":
                size = self.main_window.paint_brush_size
                color = QColor(255, 0, 0, 128)  # Semi-transparent red
            else:  # eraser
                size = self.main_window.eraser_size
                color = QColor(0, 0, 255, 128)  # Semi-transparent blue

            # Draw filled circle with lower opacity
            painter.setOpacity(0.3)
            painter.setPen(Qt.NoPen)
            painter.setBrush(color)
            painter.drawEllipse(
                QPointF(self.cursor_pos[0], self.cursor_pos[1]), size, size
            )

            # Draw circle outline with full opacity
            painter.setOpacity(1.0)
            painter.setPen(QPen(color.darker(150), 1 / self.zoom_factor, Qt.SolidLine))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(
                QPointF(self.cursor_pos[0], self.cursor_pos[1]), size, size
            )

            # Draw size text
            # Reset the transform to ensure text is drawn at screen coordinates
            painter.resetTransform()
            font = QFont()
            font.setPointSize(10)
            painter.setFont(font)
            painter.setPen(QPen(Qt.black))  # Use black color for better visibility

            # Convert cursor position back to screen coordinates
            screen_x = self.cursor_pos[0] * self.zoom_factor + self.offset_x
            screen_y = self.cursor_pos[1] * self.zoom_factor + self.offset_y

            # Position text above the circle
            text_rect = QRectF(
                screen_x + (size * self.zoom_factor),
                screen_y - (size * self.zoom_factor),
                100,
                20,
            )

            text = f"Size: {size}"
            painter.drawText(text_rect, Qt.AlignLeft | Qt.AlignVCenter, text)

            painter.restore()

    def draw_paint_mask(self, painter):
        if self.paint_mask is not None:
            mask_image = QImage(
                self.paint_mask.data,
                self.paint_mask.shape[1],
                self.paint_mask.shape[0],
                self.paint_mask.shape[1],
                QImage.Format_Grayscale8,
            )
            mask_pixmap = QPixmap.fromImage(mask_image)
            painter.setOpacity(0.5)
            painter.drawPixmap(
                self.offset_x,
                self.offset_y,
                mask_pixmap.scaled(self.scaled_pixmap.size()),
            )
            painter.setOpacity(1.0)

    def draw_eraser_mask(self, painter):
        if self.eraser_mask is not None:
            mask_image = QImage(
                self.eraser_mask.data,
                self.eraser_mask.shape[1],
                self.eraser_mask.shape[0],
                self.eraser_mask.shape[1],
                QImage.Format_Grayscale8,
            )
            mask_pixmap = QPixmap.fromImage(mask_image)
            painter.setOpacity(0.5)
            painter.drawPixmap(
                self.offset_x,
                self.offset_y,
                mask_pixmap.scaled(self.scaled_pixmap.size()),
            )
            painter.setOpacity(1.0)

    def draw_sam_bbox(self, painter):
        painter.save()
        painter.translate(self.offset_x, self.offset_y)
        painter.scale(self.zoom_factor, self.zoom_factor)
        painter.setPen(QPen(Qt.red, 2 / self.zoom_factor, Qt.SolidLine))
        x1, y1, x2, y2 = self.sam_bbox
        painter.drawRect(QRectF(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1)))
        painter.restore()

    def clear_temp_sam_prediction(self):
        self.temp_sam_prediction = None
        self.update()

    def check_unsaved_changes(self):
        if self.temp_paint_mask is not None or self.temp_eraser_mask is not None:
            reply = QMessageBox.question(
                self.main_window,
                "Unsaved Changes",
                "You have unsaved changes. Do you want to save them?",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
            )
            if reply == QMessageBox.Yes:
                if self.temp_paint_mask is not None:
                    self.commit_paint_annotation()
                if self.temp_eraser_mask is not None:
                    self.commit_eraser_changes()
                return True
            elif reply == QMessageBox.No:
                self.discard_paint_annotation()
                self.discard_eraser_changes()
                return True
            else:  # Cancel
                return False
        return True  # No unsaved changes

    def clear(self):
        super().clear()
        self.annotations.clear()
        self.current_annotation.clear()
        self.temp_point = None
        self.current_tool = None
        self.start_point = None
        self.end_point = None
        self.highlighted_annotations.clear()
        self.original_pixmap = None
        self.scaled_pixmap = None
        self.editing_polygon = None
        self.editing_point_index = None
        self.hover_point_index = None
        self.current_rectangle = None
        self.sam_bbox = None
        self.temp_sam_prediction = None
        self.update()

    def set_class_visibility(self, class_name, is_visible):
        self.class_visibility[class_name] = is_visible

    def draw_annotations(self, painter):
        """Draw all annotations on the image."""
        if not self.original_pixmap:
            return

        painter.save()
        painter.translate(self.offset_x, self.offset_y)
        painter.scale(self.zoom_factor, self.zoom_factor)

        for class_name, class_annotations in self.annotations.items():
            if not self.main_window.is_class_visible(class_name):
                continue

            color = self.class_colors.get(class_name, QColor(Qt.white))
            for annotation in class_annotations:
                if annotation in self.highlighted_annotations:
                    border_color = Qt.red
                    fill_color = QColor(Qt.red)
                else:
                    border_color = color
                    fill_color = QColor(color)

                fill_color.setAlphaF(self.fill_opacity)

                text_color = Qt.white if self.dark_mode else Qt.black
                painter.setPen(QPen(border_color, 2 / self.zoom_factor, Qt.SolidLine))
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
                                painter.setFont(
                                    QFont("Arial", int(12 / self.zoom_factor))
                                )
                                painter.setPen(
                                    QPen(text_color, 2 / self.zoom_factor, Qt.SolidLine)
                                )
                                painter.drawText(
                                    centroid,
                                    f"{class_name} {annotation.get('number', '')}",
                                )

                elif "bbox" in annotation:
                    x, y, width, height = annotation["bbox"]
                    painter.drawRect(QRectF(x, y, width, height))
                    painter.setPen(QPen(text_color, 2 / self.zoom_factor, Qt.SolidLine))
                    painter.drawText(
                        QPointF(x, y), f"{class_name} {annotation.get('number', '')}"
                    )

        if self.current_annotation:
            painter.setPen(QPen(Qt.red, 2 / self.zoom_factor, Qt.SolidLine))
            points = [QPointF(float(x), float(y)) for x, y in self.current_annotation]
            if len(points) > 1:
                painter.drawPolyline(QPolygonF(points))
            for point in points:
                painter.drawEllipse(point, 5 / self.zoom_factor, 5 / self.zoom_factor)
            if self.temp_point:
                painter.drawLine(
                    points[-1],
                    QPointF(float(self.temp_point[0]), float(self.temp_point[1])),
                )

        # Draw temporary SAM prediction
        if self.temp_sam_prediction:
            temp_color = QColor(255, 165, 0, 128)  # Semi-transparent orange
            painter.setPen(QPen(temp_color, 2 / self.zoom_factor, Qt.DashLine))
            painter.setBrush(QBrush(temp_color))

            segmentation = self.temp_sam_prediction["segmentation"]
            points = [
                QPointF(float(x), float(y))
                for x, y in zip(segmentation[0::2], segmentation[1::2])
            ]
            if points:
                painter.drawPolygon(QPolygonF(points))
                centroid = self.calculate_centroid(points)
                if centroid:
                    painter.setFont(QFont("Arial", int(12 / self.zoom_factor)))
                    painter.drawText(
                        centroid, f"SAM: {self.temp_sam_prediction['score']:.2f}"
                    )

        painter.restore()

    def draw_current_rectangle(self, painter):
        """Draw the current rectangle being created."""
        if not self.current_rectangle:
            return

        painter.save()
        painter.translate(self.offset_x, self.offset_y)
        painter.scale(self.zoom_factor, self.zoom_factor)

        x1, y1, x2, y2 = self.current_rectangle
        color = self.class_colors.get(self.main_window.current_class, QColor(Qt.red))
        painter.setPen(QPen(color, 2 / self.zoom_factor, Qt.SolidLine))
        painter.drawRect(QRectF(float(x1), float(y1), float(x2 - x1), float(y2 - y1)))

        painter.restore()

    def get_rectangle_from_points(self):
        """Get rectangle coordinates from start and end points."""
        if not self.start_point or not self.end_point:
            return None
        x1, y1 = self.start_point
        x2, y2 = self.end_point
        return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]

    def draw_editing_polygon(self, painter):
        """Draw the polygon being edited."""
        painter.save()
        painter.translate(self.offset_x, self.offset_y)
        painter.scale(self.zoom_factor, self.zoom_factor)

        points = [
            QPointF(float(x), float(y))
            for x, y in zip(
                self.editing_polygon["segmentation"][0::2],
                self.editing_polygon["segmentation"][1::2],
            )
        ]
        color = self.class_colors.get(
            self.editing_polygon["category_name"], QColor(Qt.white)
        )
        fill_color = QColor(color)
        fill_color.setAlphaF(self.fill_opacity)

        painter.setPen(QPen(color, 2 / self.zoom_factor, Qt.SolidLine))
        painter.setBrush(QBrush(fill_color))
        painter.drawPolygon(QPolygonF(points))  # Changed QPolygon to QPolygonF - Sreeni

        for i, point in enumerate(points):
            if i == self.hover_point_index:
                painter.setBrush(QColor(255, 0, 0))
            else:
                painter.setBrush(QColor(0, 255, 0))
            painter.drawEllipse(point, 5 / self.zoom_factor, 5 / self.zoom_factor)

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

    def set_zoom(self, zoom_factor):
        """Set the zoom factor and update the display."""
        self.zoom_factor = zoom_factor
        self.update_scaled_pixmap()
        self.update()

    def wheelEvent(self, event: QWheelEvent):
        if event.modifiers() == Qt.ControlModifier:
            delta = event.angleDelta().y()
            if delta > 0:
                self.main_window.zoom_in()
            else:
                self.main_window.zoom_out()
            event.accept()
        else:
            super().wheelEvent(event)

    def mousePressEvent(self, event: QMouseEvent):
        if not self.original_pixmap:
            return
        if event.modifiers() == Qt.ControlModifier and event.button() == Qt.LeftButton:
            self.pan_start_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
        else:
            pos = self.get_image_coordinates(event.pos())
            # --- Immediate SAM-points prediction on click ---
            if self.sam_points_active:
                print(f"sam_points_active: {self.sam_points_active}")
                if event.button() == Qt.LeftButton:
                    self.sam_positive_points.append(pos)
                    self.update()
                    self.main_window.apply_sam_prediction()
                    print(f"sam_positive_points: {self.sam_positive_points}")
                    return
                elif event.button() == Qt.RightButton:
                    self.sam_negative_points.append(pos)
                    self.update()
                    self.main_window.apply_sam_prediction()
                    print(f"sam_negative_points: {self.sam_negative_points}")
                    return
            # --- End immediate SAM-points handling ---

            if event.button() == Qt.LeftButton:
                if self.sam_box_active:
                    self.sam_bbox = [pos[0], pos[1], pos[0], pos[1]]
                    self.drawing_sam_bbox = True
                elif self.sam_magic_wand_active:
                    self.sam_bbox = [pos[0], pos[1], pos[0], pos[1]]
                    self.drawing_sam_bbox = True
                elif self.editing_polygon:
                    self.handle_editing_click(pos, event)
                elif self.current_tool == "polygon":
                    if not self.drawing_polygon:
                        self.drawing_polygon = True
                        self.current_annotation = []
                    self.current_annotation.append(pos)
                elif self.current_tool == "rectangle":
                    self.start_point = pos
                    self.end_point = pos
                    self.drawing_rectangle = True
                    self.current_rectangle = None
                elif self.current_tool == "paint_brush":
                    self.start_painting(pos)
                elif self.current_tool == "eraser":
                    self.start_erasing(pos)
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent):
        if not self.original_pixmap:
            return
        self.cursor_pos = self.get_image_coordinates(event.pos())
        if event.modifiers() == Qt.ControlModifier and event.buttons() == Qt.LeftButton:
            if self.pan_start_pos:
                delta = event.pos() - self.pan_start_pos
                scrollbar_h = self.main_window.scroll_area.horizontalScrollBar()
                scrollbar_v = self.main_window.scroll_area.verticalScrollBar()
                scrollbar_h.setValue(scrollbar_h.value() - delta.x())
                scrollbar_v.setValue(scrollbar_v.value() - delta.y())
                self.pan_start_pos = event.pos()
            event.accept()
        else:
            pos = self.cursor_pos
            if (
                self.sam_box_active
                and self.drawing_sam_bbox
                and self.sam_bbox is not None
            ):
                self.sam_bbox[2] = pos[0]
                self.sam_bbox[3] = pos[1]
            elif (
                self.sam_magic_wand_active
                and self.drawing_sam_bbox
                and self.sam_bbox is not None
            ):
                self.sam_bbox[2] = pos[0]
                self.sam_bbox[3] = pos[1]
            elif self.editing_polygon:
                self.handle_editing_move(pos)
            elif self.current_tool == "polygon" and self.current_annotation:
                self.temp_point = pos
            elif self.current_tool == "rectangle" and self.drawing_rectangle:
                self.end_point = pos
                self.current_rectangle = self.get_rectangle_from_points()
            elif (
                self.current_tool == "paint_brush" and event.buttons() == Qt.LeftButton
            ):
                self.continue_painting(pos)
            elif self.current_tool == "eraser" and event.buttons() == Qt.LeftButton:
                self.continue_erasing(pos)
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if not self.original_pixmap:
            return
        if event.modifiers() == Qt.ControlModifier and event.button() == Qt.LeftButton:
            self.pan_start_pos = None
            self.setCursor(Qt.ArrowCursor)
            event.accept()
        else:
            pos = self.get_image_coordinates(event.pos())
            if event.button() == Qt.LeftButton:
                if (
                    self.sam_box_active
                    and self.drawing_sam_bbox
                    and self.sam_bbox is not None
                ):
                    self.sam_bbox[2] = pos[0]
                    self.sam_bbox[3] = pos[1]
                    self.drawing_sam_bbox = False
                    self.main_window.apply_sam_prediction()
                elif (
                    self.sam_magic_wand_active
                    and self.drawing_sam_bbox
                    and self.sam_bbox is not None
                ):
                    self.sam_bbox[2] = pos[0]
                    self.sam_bbox[3] = pos[1]
                    self.drawing_sam_bbox = False
                    self.main_window.apply_sam_prediction()
                elif self.editing_polygon:
                    self.editing_point_index = None
                elif self.current_tool == "rectangle" and self.drawing_rectangle:
                    self.drawing_rectangle = False
                    if self.current_rectangle:
                        self.main_window.finish_rectangle()
                elif self.current_tool == "paint_brush":
                    self.finish_painting()
                elif self.current_tool == "eraser":
                    self.finish_erasing()
            self.update()

    def mouseDoubleClickEvent(self, event):
        if not self.pixmap():
            return
        pos = self.get_image_coordinates(event.pos())
        if event.button() == Qt.LeftButton:
            if self.drawing_polygon and len(self.current_annotation) > 2:
                self.finish_polygon()
            else:
                self.clear_current_annotation()
                annotation = self.start_polygon_edit(pos)
                if annotation:
                    self.main_window.select_annotation_in_list(annotation)
        self.update()

    def get_image_coordinates(self, pos):
        if not self.scaled_pixmap:
            return (0, 0)
        x = (pos.x() - self.offset_x) / self.zoom_factor
        y = (pos.y() - self.offset_y) / self.zoom_factor
        return (int(x), int(y))

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            if self.temp_annotations:
                self.accept_temp_annotations()
            elif self.temp_sam_prediction:
                self.main_window.accept_sam_prediction()
            elif self.editing_polygon:
                self.editing_polygon = None
                self.editing_point_index = None
                self.hover_point_index = None
                self.main_window.enable_tools()
                self.main_window.update_annotation_list()
            elif self.current_tool == "polygon" and self.drawing_polygon:
                self.finish_polygon()
            elif self.current_tool == "paint_brush":
                self.commit_paint_annotation()
            elif self.current_tool == "eraser":
                self.commit_eraser_changes()
            else:
                self.finish_current_annotation()
        elif event.key() == Qt.Key_Escape:
            if self.sam_points_active:
                self.sam_positive_points = []
                self.sam_negative_points = []
                self.clear_temp_sam_prediction()
                self.update()
            elif self.temp_annotations:
                self.discard_temp_annotations()
            elif self.sam_magic_wand_active:
                self.sam_bbox = None
                self.clear_temp_sam_prediction()
            elif self.editing_polygon:
                self.editing_polygon = None
                self.editing_point_index = None
                self.hover_point_index = None
                self.main_window.enable_tools()
            elif self.current_tool == "paint_brush":
                self.discard_paint_annotation()
            elif self.current_tool == "eraser":
                self.discard_eraser_changes()
            else:
                self.cancel_current_annotation()
        elif event.key() == Qt.Key_Delete:
            if self.editing_polygon:
                self.main_window.delete_selected_annotations()
                self.editing_polygon = None
                self.editing_point_index = None
                self.hover_point_index = None
                self.main_window.enable_tools()
                self.update()
        elif event.key() == Qt.Key_Minus:
            if self.current_tool == "paint_brush":
                self.main_window.paint_brush_size = max(
                    1, self.main_window.paint_brush_size - 1
                )
                print(f"Paint brush size: {self.main_window.paint_brush_size}")
            elif self.current_tool == "eraser":
                self.main_window.eraser_size = max(1, self.main_window.eraser_size - 1)
                print(f"Eraser size: {self.main_window.eraser_size}")
        elif event.key() == Qt.Key_Equal:
            if self.current_tool == "paint_brush":
                self.main_window.paint_brush_size += 1
                print(f"Paint brush size: {self.main_window.paint_brush_size}")
            elif self.current_tool == "eraser":
                self.main_window.eraser_size += 1
                print(f"Eraser size: {self.main_window.eraser_size}")
        self.update()

    def cancel_current_annotation(self):
        """Cancel the current annotation being created."""
        if self.current_tool == "polygon" and self.current_annotation:
            self.current_annotation = []
            self.temp_point = None
            self.drawing_polygon = False
            self.update()

    def finish_current_annotation(self):
        """Finish the current annotation being created."""
        if self.current_tool == "polygon" and len(self.current_annotation) > 2:
            if self.main_window:
                self.main_window.finish_polygon()

    def finish_polygon(self):
        """Finish the current polygon annotation."""
        if self.drawing_polygon and len(self.current_annotation) > 2:
            self.drawing_polygon = False
            if self.main_window:
                self.main_window.finish_polygon()

    def start_polygon_edit(self, pos):
        for class_name, annotations in self.annotations.items():
            for annotation in annotations:
                if "segmentation" in annotation:
                    points = [
                        QPoint(int(x), int(y))
                        for x, y in zip(
                            annotation["segmentation"][0::2],
                            annotation["segmentation"][1::2],
                        )
                    ]
                    if self.point_in_polygon(pos, points):
                        self.editing_polygon = annotation
                        self.current_tool = None
                        self.main_window.disable_tools()
                        self.main_window.reset_tool_buttons()
                        return annotation
        return None

    def handle_editing_click(self, pos, event):
        """Handle clicks during polygon editing."""
        points = [
            QPoint(int(x), int(y))
            for x, y in zip(
                self.editing_polygon["segmentation"][0::2],
                self.editing_polygon["segmentation"][1::2],
            )
        ]
        for i, point in enumerate(points):
            if self.distance(pos, point) < 10 / self.zoom_factor:
                if event.modifiers() & Qt.ShiftModifier:
                    # Delete point
                    del self.editing_polygon["segmentation"][i * 2 : i * 2 + 2]
                else:
                    # Start moving point
                    self.editing_point_index = i
                return
        # Add new point
        for i in range(len(points)):
            if self.point_on_line(pos, points[i], points[(i + 1) % len(points)]):
                self.editing_polygon["segmentation"][i * 2 + 2 : i * 2 + 2] = [
                    pos[0],
                    pos[1],
                ]
                self.editing_point_index = i + 1
                return

    def handle_editing_move(self, pos):
        """Handle mouse movement during polygon editing."""
        points = [
            QPoint(int(x), int(y))
            for x, y in zip(
                self.editing_polygon["segmentation"][0::2],
                self.editing_polygon["segmentation"][1::2],
            )
        ]
        self.hover_point_index = None
        for i, point in enumerate(points):
            if self.distance(pos, point) < 10 / self.zoom_factor:
                self.hover_point_index = i
                break
        if self.editing_point_index is not None:
            self.editing_polygon["segmentation"][self.editing_point_index * 2] = pos[0]
            self.editing_polygon["segmentation"][self.editing_point_index * 2 + 1] = (
                pos[1]
            )

    def exit_editing_mode(self):
        self.editing_polygon = None
        self.editing_point_index = None
        self.hover_point_index = None
        self.update()

    @staticmethod
    def point_in_polygon(point, polygon):
        """Check if a point is inside a polygon."""
        n = len(polygon)
        inside = False
        p1x, p1y = polygon[0].x(), polygon[0].y()
        for i in range(n + 1):
            p2x, p2y = polygon[i % n].x(), polygon[i % n].y()
            if point[1] > min(p1y, p2y):
                if point[1] <= max(p1y, p2y):
                    if point[0] <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (point[1] - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or point[0] <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    @staticmethod
    def point_to_tuple(point):
        """Convert QPoint to tuple."""
        if isinstance(point, QPoint):
            return (point.x(), point.y())
        return point

    @staticmethod
    def distance(p1, p2):
        """Calculate distance between two points."""
        p1 = ImageLabel.point_to_tuple(p1)
        p2 = ImageLabel.point_to_tuple(p2)
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

    @staticmethod
    def point_on_line(p, start, end):
        """Check if a point is on a line segment."""
        p = ImageLabel.point_to_tuple(p)
        start = ImageLabel.point_to_tuple(start)
        end = ImageLabel.point_to_tuple(end)
        d1 = ImageLabel.distance(p, start)
        d2 = ImageLabel.distance(p, end)
        line_length = ImageLabel.distance(start, end)
        buffer = 0.1  # Adjust this value for more or less strict "on-line" detection
        return abs(d1 + d2 - line_length) < buffer
