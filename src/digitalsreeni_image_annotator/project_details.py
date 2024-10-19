from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QTextEdit, QPushButton, QLabel, 
                             QDialogButtonBox, QScrollArea, QWidget)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
import os
from datetime import datetime

class ProjectDetailsDialog(QDialog):
    def __init__(self, parent=None, stats_dialog=None):
        super().__init__(parent)
        self.parent = parent
        self.stats_dialog = stats_dialog
        self.setWindowTitle("Project Details")
        self.setModal(True)
        self.setMinimumSize(600, 800)  # Set initial size
        self.original_notes = parent.project_notes if parent else ""
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        # Helper function to create bold labels
        def bold_label(text):
            label = QLabel(text)
            font = label.font()
            font.setBold(True)
            label.setFont(font)
            return label

        # Helper function to format datetime
        def format_datetime(date_string):
            try:
                dt = datetime.fromisoformat(date_string)
                return dt.strftime("%Y-%m-%d %H:%M:%S")
            except ValueError:
                return date_string  # Return original string if parsing fails

        # Project metadata
        scroll_layout.addWidget(bold_label("Project:"))
        scroll_layout.addWidget(QLabel(os.path.basename(self.parent.current_project_file)))
        scroll_layout.addWidget(bold_label("Creation Date:"))
        scroll_layout.addWidget(QLabel(format_datetime(getattr(self.parent, 'project_creation_date', 'N/A'))))
        scroll_layout.addWidget(bold_label("Last Modified:"))
        scroll_layout.addWidget(QLabel(format_datetime(getattr(self.parent, 'last_modified', 'N/A'))))

        # Image information
        image_count = len(self.parent.all_images)
        scroll_layout.addWidget(bold_label(f"Total Images: {image_count}"))
        
        # List image file names
        scroll_layout.addWidget(bold_label("Image Files:"))
        image_names = [f"• {os.path.basename(path)}" for path in self.parent.image_paths.values()]
        image_list = QLabel("\n".join(image_names))
        image_list.setWordWrap(True)
        scroll_layout.addWidget(image_list)

        # Multi-dimensional image information
        multi_slice_images = [img for img in self.parent.all_images if img.get('is_multi_slice', False)]
        if multi_slice_images:
            scroll_layout.addWidget(bold_label(f"Multi-dimensional Images: {len(multi_slice_images)}"))
            for img in multi_slice_images:
                slice_count = len(img.get('slices', []))
                scroll_layout.addWidget(QLabel(f"• {os.path.basename(img['file_name'])}: {slice_count} slices"))

        # Annotation information
        class_names = list(self.parent.class_mapping.keys())
        scroll_layout.addWidget(bold_label("Classes:"))
        class_list = QLabel("\n".join([f"• {name}" for name in class_names]))
        class_list.setWordWrap(True)
        scroll_layout.addWidget(class_list)

        # Add annotation statistics
        if self.stats_dialog:
            scroll_layout.addWidget(bold_label("Annotation Statistics:"))
            stats_text = self.stats_dialog.text_browser.toPlainText()
            stats_lines = stats_text.split('\n')
            formatted_stats = []
            for line in stats_lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    formatted_stats.append(f"<p><b>{key}:</b>{value}</p>")
                else:
                    formatted_stats.append(f"<p>{line}</p>")
            stats_label = QLabel("".join(formatted_stats))
            stats_label.setTextFormat(Qt.RichText)
            stats_label.setWordWrap(True)
            scroll_layout.addWidget(stats_label)

        scroll_area.setWidget(scroll_content)
        layout.addWidget(scroll_area)

        # Project notes
        layout.addWidget(bold_label("Project Notes:"))
        self.notes_edit = QTextEdit()
        self.notes_edit.setPlainText(getattr(self.parent, 'project_notes', ''))
        layout.addWidget(self.notes_edit)

        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def get_notes(self):
        return self.notes_edit.toPlainText()

    def were_changes_made(self):
        return self.get_notes() != self.original_notes