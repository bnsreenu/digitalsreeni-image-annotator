# Architecture Constraints

## Technical Constraints

| Constraint | Description | Rationale |
|------------|-------------|-----------|
| **Python 3.10+** | Minimum Python version | Required for modern type hints and library compatibility |
| **PyQt6 `>=6.7,<6.12`** | GUI framework | Cross-platform, mature, rich widget set; improved Linux/XCB integration over PyQt5. The ceiling is the highest minor CI exercises, +1 — Qt 6.12 is the last release supporting Windows 10, so raising it changes the platform baseline (ADR-046) |
| **A Qt-free environment** | No second Qt on the DLL search path | Installing into an existing Conda env whose own Qt shadows the PyQt6 wheel's produces `0xc0000139` at import. Diagnose with `sreeni-cli doctor` (issue #92, ADR-046) |
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
| **Linux** | ✅ Supported | Qt6 native integration; runtime needs libxcb-cursor0 |

### Linux Runtime Requirements
- `libxcb-cursor0` (required by Qt 6, was optional under Qt 5)
- `libegl1`, `libgl1` for software rendering fallback
- `libxkbcommon-x11-0` and the standard XCB plugin set

## Conventions

| Convention | Description |
|------------|-------------|
| **Code Style** | Follow existing PyQt6 patterns (fully-qualified enum names) |
| **UI Modes** | Support both light and dark mode |
| **Image Paths** | Store absolute paths in project files |
| **Annotations** | Polygon (segmentation) or bbox (rectangle) format |
