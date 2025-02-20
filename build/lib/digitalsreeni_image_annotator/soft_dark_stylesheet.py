# soft_dark_stylesheet.py

soft_dark_stylesheet = """
QWidget {
    background-color: #2F2F2F;
    color: #E0E0E0;
    font-family: Arial, sans-serif;
}

QMainWindow {
    background-color: #2A2A2A;
}

QPushButton {
    background-color: #4A4A4A;
    border: 1px solid #5E5E5E;
    padding: 5px 10px;
    border-radius: 3px;
    color: #E0E0E0;
}

QPushButton:hover {
    background-color: #545454;
}

QPushButton:pressed {
    background-color: #404040;
}

QPushButton:checked {
    background-color: #606060;
    border: 2px solid #808080;
    color: #FFFFFF;
}

QListWidget, QTreeWidget {
    background-color: #3A3A3A;
    border: 1px solid #4A4A4A;
    border-radius: 3px;
    color: #E0E0E0;
}

QListWidget::item, QTreeWidget::item {
    color: #E0E0E0;  
}

QListWidget::item:selected, QTreeWidget::item:selected {
    background-color: #4A4A4A;
    color: #FFFFFF;  /* Make selected items a bit brighter */
}

QLabel {
    color: #E0E0E0;
}

QLabel.section-header {
    font-weight: bold;
    font-size: 14px;
    padding: 5px 0;
    color: #FFFFFF;  /* Bright white color for better visibility in dark mode */
}

QLineEdit, QTextEdit, QPlainTextEdit {
    background-color: #3A3A3A;
    border: 1px solid #4A4A4A;
    color: #E0E0E0;
    padding: 2px;
    border-radius: 3px;
}

QSlider::groove:horizontal {
    background: #4A4A4A;
    height: 8px;
    border-radius: 4px;
}

QSlider::handle:horizontal {
    background: #6A6A6A;
    width: 18px;
    margin-top: -5px;
    margin-bottom: -5px;
    border-radius: 9px;
}

QSlider::handle:horizontal:hover {
    background: #7A7A7A;
}

QScrollBar:vertical, QScrollBar:horizontal {
    background-color: #3A3A3A;
    width: 12px;
    height: 12px;
}

QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
    background-color: #5A5A5A;
    border-radius: 6px;
    min-height: 20px;
}

QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover {
    background-color: #6A6A6A;
}

QScrollBar::add-line, QScrollBar::sub-line {
    background: none;
}

QMenuBar {
    background-color: #2F2F2F;
}

QMenuBar::item {
    padding: 5px 10px;
    background-color: transparent;
}

QMenuBar::item:selected {
    background-color: #3A3A3A;
}

QMenu {
    background-color: #2F2F2F;
    border: 1px solid #3A3A3A;
}

QMenu::item {
    padding: 5px 20px 5px 20px;
}

QMenu::item:selected {
    background-color: #3A3A3A;
}

QToolTip {
    background-color: #2F2F2F;
    color: #E0E0E0;
    border: 1px solid #3A3A3A;
}

QStatusBar {
    background-color: #2A2A2A;
    color: #B0B0B0;
}

QListWidget::item {
    color: none;
}
"""