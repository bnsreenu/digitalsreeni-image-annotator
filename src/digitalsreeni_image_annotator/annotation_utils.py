from PyQt5.QtWidgets import QListWidgetItem
from PyQt5.QtGui import QColor
from PyQt5.QtCore import Qt

class AnnotationUtils:
    @staticmethod
    def update_annotation_list(self, image_name=None):
        self.annotation_list.clear()
        current_name = image_name or self.current_slice or self.image_file_name
        annotations = self.all_annotations.get(current_name, {})
        for class_name, class_annotations in annotations.items():
            color = self.image_label.class_colors.get(class_name, QColor(Qt.white))
            for i, annotation in enumerate(class_annotations, start=1):
                item_text = f"{class_name} - {i}"
                item = QListWidgetItem(item_text)
                item.setData(Qt.UserRole, annotation)
                item.setForeground(color)
                self.annotation_list.addItem(item)

    @staticmethod
    def update_slice_list_colors(self):
        for i in range(self.slice_list.count()):
            item = self.slice_list.item(i)
            slice_name = item.text()
            if slice_name in self.all_annotations and any(self.all_annotations[slice_name].values()):
                item.setForeground(QColor(Qt.green))
            else:
                item.setForeground(QColor(Qt.black) if not self.dark_mode else QColor(Qt.white))

    @staticmethod
    def update_annotation_list_colors(self, class_name=None, color=None):
        for i in range(self.annotation_list.count()):
            item = self.annotation_list.item(i)
            annotation = item.data(Qt.UserRole)
            if class_name is None or annotation['category_name'] == class_name:
                item_color = color if class_name else self.image_label.class_colors.get(annotation['category_name'], QColor(Qt.white))
                item.setForeground(item_color)

    @staticmethod
    def load_image_annotations(self):
        self.image_label.annotations.clear()
        current_name = self.current_slice or self.image_file_name
        if current_name in self.all_annotations:
            self.image_label.annotations = self.all_annotations[current_name].copy()
        self.image_label.update()

    @staticmethod
    def save_current_annotations(self):
        current_name = self.current_slice or self.image_file_name
        if current_name:
            if self.image_label.annotations:
                self.all_annotations[current_name] = self.image_label.annotations.copy()
            elif current_name in self.all_annotations:
                del self.all_annotations[current_name]
        AnnotationUtils.update_slice_list_colors(self)

    @staticmethod
    def add_annotation_to_list(self, annotation):
        class_name = annotation['category_name']
        color = self.image_label.class_colors.get(class_name, QColor(Qt.white))
        annotations = self.image_label.annotations.get(class_name, [])
        item_text = f"{class_name} - {len(annotations)}"
        item = QListWidgetItem(item_text)
        item.setData(Qt.UserRole, annotation)
        item.setForeground(color)
        self.annotation_list.addItem(item)