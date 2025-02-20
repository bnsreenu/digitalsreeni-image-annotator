# Release Notes
## Version 0.8.9

### New Features and Enhancements
- Same as version 0.8.9 except changed the requirements file to define specific version numbers for the libraies used.
- The following bug fixes and optimizations correspond to version 0.8.10 

### Bug Fixes and Optimizations
1. **Project Corruption Prevention**
   - Fixed critical issue where projects could become corrupted if application was terminated during loading
   - Disabled auto-save functionality during project loading process
   - Enhanced project loading stability for large datasets
   - Protected project integrity when handling multiple classes and images

### Notes
- All existing tools continue to support both Windows and macOS operating systems
- Improved reliability of project file handling
- Critical update recommended for users working with large projects