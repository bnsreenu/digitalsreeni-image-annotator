# Release Notes

## Version X.X.X (Insert version number)

### New Features and Enhancements

1. Project Notes and Details
   - Added ability to add and edit custom notes for each project.
   - Implemented a Project Details view, accessible from the Project menu.
   - Project summary now includes:
     - Project creation date
     - Last modified date
     - Total number of images
     - List of image file names
     - Information about multi-dimensional images (if present)
     - List of classes used in the project
     - Annotation statistics (total objects, average objects per image, class distribution)
   - Project notes are automatically saved when edited.

2. Advanced Project Search Functionality
   - Introduced a new "Search Projects" feature in the Project menu.
   - Allows searching across multiple projects based on various criteria:
     - Project name
     - Class names
     - Image file names
     - Project notes content
   - Supports complex search queries using logical operators (AND, OR) and parentheses.
   - Search results can be double-clicked to open the corresponding project.

3. UI Improvements
   - Added a new "Project Details" option in the Project menu.
   - Implemented a "Search Projects" dialog with an intuitive interface for complex queries.

### Bug Fixes and Optimizations

- Fixed an issue where the "Project details updated" message was shown even when no changes were made.
- Optimized the project loading process to properly handle project metadata.
- Improved error handling in the search functionality to prevent crashes on invalid project files.

### Developer Notes

- Refactored the project data structure to include metadata fields (creation date, last modified date, notes).
- Implemented a recursive parser for handling complex search queries with nested expressions.