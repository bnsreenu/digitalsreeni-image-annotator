# Architecture Constraints

## Technical Constraints

| Constraint | Description | Rationale |
|------------|-------------|-----------|
| **Python 3.10+** | Minimum Python version | Required for modern type hints and library compatibility |
| **PyQt5** | GUI framework | Cross-platform, mature, rich widget set |
| **Ultralytics** | SAM integration | Simplified SAM model loading, includes PyTorch |
| **Desktop Application** | Not web-based | Direct file system access, better performance for large images |

## Organizational Constraints

| Constraint | Description |
|------------|-------------|
| **Open Source** | MIT License |
| **No Automated Tests** | Manual testing only (current state) |
| **Fork Maintenance** | Maintain compatibility with upstream changes |

## Platform Constraints

| Platform | Status | Notes |
|----------|--------|-------|
| **Windows** | ✅ Fully Supported | Primary development platform |
| **macOS** | ✅ Fully Supported | Tested and working |
| **Linux** | ⚠️ Limited Support | XCB plugin issues, minimal testing |

### Linux-Specific Issues
- PyQt5 XCB platform plugin conflicts
- Workaround: Remove `QT_QPA_PLATFORM_PLUGIN_PATH` environment variable on startup (see [main.py](../../src/digitalsreeni_image_annotator/main.py:15-19))

## Conventions

| Convention | Description |
|------------|-------------|
| **Code Style** | Follow existing PyQt5 patterns |
| **UI Modes** | Support both light and dark mode |
| **Image Paths** | Store absolute paths in project files |
| **Annotations** | Polygon (segmentation) or bbox (rectangle) format |
