"""
ImageLabel module for the Image Annotator application.

This module contains the ImageLabel class, which is responsible for
displaying the image and handling annotation interactions.

@DigitalSreeni
Dr. Sreenivas Bhattiprolu
"""

from PyQt5.QtWidgets import QLabel, QApplication
from PyQt5.QtGui import (QPainter, QPen, QColor, QFont, QPolygonF, QBrush, QPolygon,
                         QPixmap, QImage, QWheelEvent, QMouseEvent, QKeyEvent)
from PyQt5.QtCore import Qt, QPoint, QPointF, QRectF, QSize
from PIL import Image
import os
import warnings
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
        self.editing_polygon = None
        self.editing_point_index = None
        self.hover_point_index = None
        self.fill_opacity = 0.3
        self.drawing_rectangle = False
        self.current_rectangle = None
        self.bit_depth = None
        self.image_path = None
        self.dark_mode = False
        
        #SAM2
        self.sam_magic_wand_active = False
        self.sam_bbox = None
        self.drawing_sam_bbox = False
        self.temp_sam_prediction = None


        
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
                if img.mode == '1':
                    self.bit_depth = 1
                elif img.mode == 'L':
                    self.bit_depth = 8
                elif img.mode == 'I;16':
                    self.bit_depth = 16
                elif img.mode in ['RGB', 'HSV']:
                    self.bit_depth = 24
                elif img.mode in ['RGBA', 'CMYK']:
                    self.bit_depth = 32
                else:
                    self.bit_depth = img.bits
                
                if self.main_window:
                    self.main_window.update_image_info()

    def update_scaled_pixmap(self):
        """Update the scaled pixmap based on the zoom factor."""
        if self.original_pixmap:
            scaled_size = self.original_pixmap.size() * self.zoom_factor
            self.scaled_pixmap = self.original_pixmap.scaled(
                scaled_size.width(),
                scaled_size.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            super().setPixmap(self.scaled_pixmap)
            self.setMinimumSize(self.scaled_pixmap.size())
            self.update_offset()

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

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.scaled_pixmap:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            
            # Draw the image
            painter.drawPixmap(int(self.offset_x), int(self.offset_y), self.scaled_pixmap)
            
            # Draw annotations
            self.draw_annotations(painter)
            if self.editing_polygon:
                self.draw_editing_polygon(painter)
            if self.drawing_rectangle and self.current_rectangle:
                self.draw_current_rectangle(painter)
            if self.sam_magic_wand_active and self.sam_bbox:
                self.draw_sam_bbox(painter)

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


    def draw_annotations(self, painter):
        """Draw all annotations on the image."""
        if not self.original_pixmap:
            return

        painter.save()
        painter.translate(self.offset_x, self.offset_y)
        painter.scale(self.zoom_factor, self.zoom_factor)

        for class_name, class_annotations in self.annotations.items():
            color = self.class_colors.get(class_name, QColor(Qt.white))
            for i, annotation in enumerate(class_annotations, start=1):
                if annotation in self.highlighted_annotations:
                    border_color = Qt.red
                    fill_color = QColor(Qt.red)
                else:
                    border_color = color
                    fill_color = QColor(color)
                
                fill_color.setAlphaF(self.fill_opacity)
                
                text_color = Qt.white if self.dark_mode else Qt.black
                painter.setPen(QPen(text_color, 2 / self.zoom_factor, Qt.SolidLine))
                painter.setPen(QPen(border_color, 2 / self.zoom_factor, Qt.SolidLine))
                painter.setBrush(QBrush(fill_color))

                if "segmentation" in annotation:
                    segmentation = annotation["segmentation"]
                    if isinstance(segmentation, list) and len(segmentation) > 0:
                        if isinstance(segmentation[0], list):  # Multiple polygons
                            for polygon in segmentation:
                                points = [QPointF(float(x), float(y)) for x, y in zip(polygon[0::2], polygon[1::2])]
                                if points:
                                    painter.drawPolygon(QPolygonF(points))
                        else:  # Single polygon
                            points = [QPointF(float(x), float(y)) for x, y in zip(segmentation[0::2], segmentation[1::2])]
                            if points:
                                painter.drawPolygon(QPolygonF(points))
                        
                        # Draw centroid and label
                        if points:
                            centroid = self.calculate_centroid(points)
                            if centroid:
                                painter.setFont(QFont("Arial", int(12 / self.zoom_factor)))
                                painter.drawText(centroid, str(i))
                elif "bbox" in annotation:
                    x, y, width, height = annotation["bbox"]
                    painter.drawRect(QRectF(x, y, width, height))
                    painter.drawText(QPointF(x, y), str(i))

        if self.current_annotation:
            painter.setPen(QPen(Qt.red, 2 / self.zoom_factor, Qt.SolidLine))
            points = [QPointF(float(x), float(y)) for x, y in self.current_annotation]
            if len(points) > 1:
                painter.drawPolyline(QPolygonF(points))  # Changed QPolygon to QPolygonF - Sreeni
            for point in points:
                painter.drawEllipse(point, 5 / self.zoom_factor, 5 / self.zoom_factor)
            if self.temp_point:
                painter.drawLine(points[-1], QPointF(float(self.temp_point[0]), float(self.temp_point[1])))
                
                # Draw temporary SAM prediction
        if self.temp_sam_prediction:
            temp_color = QColor(255, 165, 0, 128)  # Semi-transparent orange
            painter.setPen(QPen(temp_color, 2 / self.zoom_factor, Qt.DashLine))
            painter.setBrush(QBrush(temp_color))
            
            segmentation = self.temp_sam_prediction["segmentation"]
            points = [QPointF(float(x), float(y)) for x, y in zip(segmentation[0::2], segmentation[1::2])]
            if points:
                painter.drawPolygon(QPolygonF(points))
                centroid = self.calculate_centroid(points)
                if centroid:
                    painter.setFont(QFont("Arial", int(12 / self.zoom_factor)))
                    painter.drawText(centroid, f"SAM: {self.temp_sam_prediction['score']:.2f}")

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

        points = [QPointF(float(x), float(y)) for x, y in zip(self.editing_polygon["segmentation"][0::2], self.editing_polygon["segmentation"][1::2])]
        color = self.class_colors.get(self.editing_polygon["category_name"], QColor(Qt.white))
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
        """Handle wheel events for zooming."""
        if event.modifiers() == Qt.ControlModifier and self.main_window:
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
            if event.button() == Qt.LeftButton:
                if self.sam_magic_wand_active:
                    self.sam_bbox = [pos[0], pos[1], pos[0], pos[1]]
                    self.drawing_sam_bbox = True
                elif self.editing_polygon:
                    self.handle_editing_click(pos, event)
                elif self.current_tool == "polygon":
                    self.current_annotation.append(pos)
                    self.main_window.finish_polygon_button.setEnabled(True)
                elif self.current_tool == "rectangle":
                    self.start_point = pos
                    self.end_point = pos
                    self.drawing_rectangle = True
                    self.current_rectangle = None
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent):
        if not self.original_pixmap:
            return
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
            pos = self.get_image_coordinates(event.pos())
            if self.sam_magic_wand_active and self.drawing_sam_bbox:
                self.sam_bbox[2] = pos[0]
                self.sam_bbox[3] = pos[1]
            elif self.editing_polygon:
                self.handle_editing_move(pos)
            elif self.current_tool == "polygon" and self.current_annotation:
                self.temp_point = pos
            elif self.current_tool == "rectangle" and self.drawing_rectangle:
                self.end_point = pos
                self.current_rectangle = self.get_rectangle_from_points()
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
                if self.sam_magic_wand_active and self.drawing_sam_bbox:
                    self.sam_bbox[2] = pos[0]
                    self.sam_bbox[3] = pos[1]
                    self.drawing_sam_bbox = False
                    self.main_window.apply_sam2_prediction()
                elif self.editing_polygon:
                    self.editing_point_index = None
                elif self.current_tool == "rectangle" and self.drawing_rectangle:
                    self.drawing_rectangle = False
                    if self.current_rectangle:
                        self.main_window.finish_rectangle()
            self.update()
            

    def mouseDoubleClickEvent(self, event):
        """Handle mouse double click events."""
        if not self.original_pixmap:
            return
        pos = self.get_image_coordinates(event.pos())
        if event.button() == Qt.LeftButton:
            self.start_polygon_edit(pos)
        self.update()

    def get_image_coordinates(self, pos):
        if not self.scaled_pixmap:
            return (0, 0)
        x = (pos.x() - self.offset_x) / self.zoom_factor
        y = (pos.y() - self.offset_y) / self.zoom_factor
        return (int(x), int(y))

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            if self.temp_sam_prediction:
                self.main_window.accept_sam_prediction()
            elif self.editing_polygon:
                self.editing_polygon = None
                self.editing_point_index = None
                self.hover_point_index = None
                self.main_window.enable_tools()
                self.main_window.update_annotation_list()
            else:
                self.finish_current_annotation()
        elif event.key() == Qt.Key_Escape:
            if self.sam_magic_wand_active:
                self.sam_magic_wand_active = False
                self.sam_bbox = None
                self.clear_temp_sam_prediction()  # Use the new method
                self.setCursor(Qt.ArrowCursor)
                if self.main_window:
                    self.main_window.sam_magic_wand_button.setChecked(False)
            elif self.editing_polygon:
                self.editing_polygon = None
                self.editing_point_index = None
                self.hover_point_index = None
                self.main_window.enable_tools()
            else:
                self.cancel_current_annotation()
        self.update()


    def cancel_current_annotation(self):
        """Cancel the current annotation being created."""
        if self.current_tool == "polygon" and self.current_annotation:
            self.current_annotation = []
            self.temp_point = None
            self.update()
            if self.main_window:
                self.main_window.finish_polygon_button.setEnabled(False)

    def finish_current_annotation(self):
        """Finish the current annotation being created."""
        if self.current_tool == "polygon" and len(self.current_annotation) > 2:
            if self.main_window:
                self.main_window.finish_polygon()


    def start_polygon_edit(self, pos):
            """Start editing a polygon."""
            for class_name, annotations in self.annotations.items():
                for i, annotation in enumerate(annotations):
                    if "segmentation" in annotation:
                        points = [QPoint(int(x), int(y)) for x, y in zip(annotation["segmentation"][0::2], annotation["segmentation"][1::2])]
                        if self.point_in_polygon(pos, points):
                            self.editing_polygon = annotation
                            self.current_tool = None
                            self.main_window.disable_tools()
                            return

    def handle_editing_click(self, pos, event):
        """Handle clicks during polygon editing."""
        points = [QPoint(int(x), int(y)) for x, y in zip(self.editing_polygon["segmentation"][0::2], self.editing_polygon["segmentation"][1::2])]
        for i, point in enumerate(points):
            if self.distance(pos, point) < 10 / self.zoom_factor:
                if event.modifiers() & Qt.ShiftModifier:
                    # Delete point
                    del self.editing_polygon["segmentation"][i*2:i*2+2]
                else:
                    # Start moving point
                    self.editing_point_index = i
                return
        # Add new point
        for i in range(len(points)):
            if self.point_on_line(pos, points[i], points[(i+1) % len(points)]):
                self.editing_polygon["segmentation"][i*2+2:i*2+2] = [pos[0], pos[1]]
                self.editing_point_index = i + 1
                return

    def handle_editing_move(self, pos):
        """Handle mouse movement during polygon editing."""
        points = [QPoint(int(x), int(y)) for x, y in zip(self.editing_polygon["segmentation"][0::2], self.editing_polygon["segmentation"][1::2])]
        self.hover_point_index = None
        for i, point in enumerate(points):
            if self.distance(pos, point) < 10 / self.zoom_factor:
                self.hover_point_index = i
                break
        if self.editing_point_index is not None:
            self.editing_polygon["segmentation"][self.editing_point_index*2] = pos[0]
            self.editing_polygon["segmentation"][self.editing_point_index*2+1] = pos[1]

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
        return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

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