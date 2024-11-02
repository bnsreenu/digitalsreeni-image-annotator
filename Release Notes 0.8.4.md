# Release Notes
## Version 0.8.4

### New Features and Enhancements

1. **Slice Registration Tool**
   - Advanced image registration tool for aligning image slices in a stack
   - Multiple registration methods supported:
     - Translation (X-Y translation only)
     - Rigid Body (translation + rotation)
     - Scaled Rotation (translation + rotation + scaling)
     - Affine (translation + rotation + scaling + shearing)
     - Bilinear (non-linear transformation)
   - Flexible reference frame options:
     - Previous frame alignment
     - First frame alignment
     - Mean of all frames
     - Mean of first N frames
     - Mean of first N frames with moving average
   - Preserves image metadata and physical dimensions
   - Outputs registered images as either single stack or individual slices
   - Maintains original pixel spacing and slice thickness information

2. **Stack Interpolation Tool**
   - Powerful tool for adjusting Z-spacing in image stacks
   - Multiple interpolation methods available:
     - Linear interpolation (fast, suitable for most cases)
     - Nearest neighbor (preserves original values)
     - Cubic interpolation (smoother results)
     - Other higher-order methods for specialized needs
   - Memory-efficient processing for large datasets
   - Preserves original bit depth and dynamic range
   - Maintains physical dimensions and metadata
   - Real-time progress tracking for long operations
   - Support for both up-sampling and down-sampling in Z dimension

3. **DICOM to TIFF Converter**
   - Comprehensive DICOM conversion tool with metadata preservation
   - Features:
     - Supports both single-slice and multi-slice DICOM files
     - Option to save as single TIFF stack or individual slices
     - Preserves important DICOM metadata:
       - Patient information
       - Acquisition parameters
       - Physical dimensions (pixel spacing, slice thickness)
     - Handles window/level adjustments
     - Maintains original bit depth
     - Exports metadata to separate JSON file for reference
     - ImageJ-compatible metadata inclusion in TIFF files
     - Progress tracking for large conversions

### Bug Fixes and Optimizations
- None

### Notes
- All new tools support both Windows and macOS operating systems
- Tools preserve original image quality and metadata
- Interface includes detailed progress information and success/error messages
- Each tool includes comprehensive error handling and input validation

