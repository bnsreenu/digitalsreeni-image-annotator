from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton, 
                             QDateEdit, QLabel, QListWidget, QDialogButtonBox, QFormLayout,
                             QFileDialog, QMessageBox)
from PyQt5.QtCore import Qt, QDate
import os
import json
from datetime import datetime

class ProjectSearchDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setWindowTitle("Search Projects")
        self.setModal(True)
        self.setMinimumSize(600, 400)
        self.search_directory = ""
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        # Search criteria
        form_layout = QFormLayout()
        self.keyword_edit = QLineEdit()
        self.keyword_edit.setPlaceholderText("Enter search query (e.g., monkey AND dog AND (project_animals OR project_zoo))")
        form_layout.addRow("Search Query:", self.keyword_edit)

        self.start_date = QDateEdit()
        self.start_date.setCalendarPopup(True)
        self.start_date.setDate(QDate.currentDate().addYears(-1))
        form_layout.addRow("Start Date:", self.start_date)

        self.end_date = QDateEdit()
        self.end_date.setCalendarPopup(True)
        self.end_date.setDate(QDate.currentDate())
        form_layout.addRow("End Date:", self.end_date)

        layout.addLayout(form_layout)

        # Directory selection
        dir_layout = QHBoxLayout()
        self.dir_edit = QLineEdit()
        dir_layout.addWidget(self.dir_edit)
        dir_button = QPushButton("Browse")
        dir_button.clicked.connect(self.browse_directory)
        dir_layout.addWidget(dir_button)
        layout.addLayout(dir_layout)

        # Search button
        search_button = QPushButton("Search")
        search_button.clicked.connect(self.perform_search)
        layout.addWidget(search_button)

        # Results list
        self.results_list = QListWidget()
        self.results_list.itemDoubleClicked.connect(self.open_selected_project)
        layout.addWidget(self.results_list)

        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Close)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def browse_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Directory to Search")
        if directory:
            self.search_directory = directory
            self.dir_edit.setText(directory)

    def perform_search(self):
        if not self.search_directory:
            QMessageBox.warning(self, "No Directory", "Please select a directory to search.")
            return

        query = self.keyword_edit.text()
        start_date = self.start_date.date().toPyDate()
        end_date = self.end_date.date().toPyDate()

        self.results_list.clear()

        for root, dirs, files in os.walk(self.search_directory):
            for filename in files:
                if filename.endswith('.iap'):
                    project_path = os.path.join(root, filename)
                    try:
                        with open(project_path, 'r') as f:
                            project_data = json.load(f)
                        
                        if self.project_matches(project_data, query, start_date, end_date):
                            self.results_list.addItem(project_path)
                    except Exception as e:
                        print(f"Error reading project file {filename}: {str(e)}")

        if self.results_list.count() == 0:
            QMessageBox.information(self, "Search Results", "No matching projects found.")
        else:
            QMessageBox.information(self, "Search Results", f"{self.results_list.count()} matching projects found.")

    def project_matches(self, project_data, query, start_date, end_date):
        # Check date range
        creation_date = project_data.get('creation_date', '')
        if creation_date:
            try:
                creation_date = datetime.fromisoformat(creation_date).date()
                if creation_date < start_date or creation_date > end_date:
                    return False
            except ValueError:
                print(f"Invalid date format in project: {creation_date}")

        if not query:
            return True

        return self.evaluate_query(query.lower(), project_data)

    def term_matches(self, term, project_data):
        # Search in project name
        if term in os.path.basename(project_data.get('current_project_file', '')).lower():
            return True
        
        # Search in classes
        if any(term in class_info['name'].lower() for class_info in project_data.get('classes', [])):
            return True
        
        # Search in image names
        if any(term in img['file_name'].lower() for img in project_data.get('images', [])):
            return True
        
        # Search in project notes
        if term in project_data.get('notes', '').lower():
            return True
        
        return False


    def evaluate_query(self, query, project_data):
        tokens = self.tokenize_query(query)
        return self.evaluate_tokens(tokens, project_data)

    def tokenize_query(self, query):
        tokens = []
        current_token = ""
        for char in query:
            if char in '()':
                if current_token:
                    tokens.append(current_token)
                    current_token = ""
                tokens.append(char)
            elif char.isspace():
                if current_token:
                    tokens.append(current_token)
                    current_token = ""
            else:
                current_token += char
        if current_token:
            tokens.append(current_token)
        return tokens

    def evaluate_tokens(self, tokens, project_data):
        def evaluate_expression():
            nonlocal i
            result = True
            current_op = 'and'

            while i < len(tokens):
                if tokens[i] == '(':
                    i += 1
                    sub_result = evaluate_expression()
                    if current_op == 'and':
                        result = result and sub_result
                    else:
                        result = result or sub_result
                elif tokens[i] == ')':
                    return result
                elif tokens[i].lower() in ['and', 'or']:
                    current_op = tokens[i].lower()
                else:
                    term_result = self.term_matches(tokens[i], project_data)
                    if current_op == 'and':
                        result = result and term_result
                    else:
                        result = result or term_result
                i += 1
            return result

        i = 0
        return evaluate_expression()


    def keyword_matches(self, keyword, project_data):
        # Search in project name
        if keyword in os.path.basename(project_data.get('current_project_file', '')).lower().split():
            return True
        
        # Search in classes
        if any(keyword in class_info['name'].lower().split() for class_info in project_data.get('classes', [])):
            return True
        
        # Search in image names
        if any(keyword in img['file_name'].lower().split() for img in project_data.get('images', [])):
            return True
        
        # Search in project notes
        if keyword in project_data.get('notes', '').lower().split():
            return True
        
        # Search in creation date and last modified date
        if keyword in project_data.get('creation_date', '').lower().split() or keyword in project_data.get('last_modified', '').lower().split():
            return True
        
        return False

    def open_selected_project(self, item):
        project_file = item.text()
        self.parent.open_specific_project(project_file)
        self.accept()

def show_project_search(parent):
    dialog = ProjectSearchDialog(parent)
    dialog.exec_()