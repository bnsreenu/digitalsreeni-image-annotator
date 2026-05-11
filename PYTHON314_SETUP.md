# Python 3.14 Setup Complete ✓

## Summary

Successfully set up the testing infrastructure for the DigitalSreeni Image Annotator project with **Python 3.14.2** support.

## What Was Fixed

### 1. Dependency Compatibility

**Problem**: Python 3.14 is bleeding-edge and requires specific dependency versions.

**Solution**:
- Updated `numpy>=2.4.0` (required for Python 3.14)
- Changed all pinned versions in [setup.py](setup.py) to flexible constraints (`>=`)
- Updated [requirements.txt](requirements.txt) with Python 3.14-compatible versions

### 2. PyTorch/Torch DLL Loading Issue

**Problem**: PyTorch has DLL compatibility issues with Python 3.14 on Windows, causing:
```
OSError: [WinError 1114] A dynamic link library (DLL) initialization routine failed.
Error loading "torch\lib\c10.dll"
```

**Solution**: Modified test imports to load modules directly by file path using `importlib.util`, bypassing the package `__init__.py` that imports torch:

```python
import importlib.util

# Import module directly by file path
module_path = os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'digitalsreeni_image_annotator', 'utils.py')
spec = importlib.util.spec_from_file_location("utils", module_path)
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)
```

This allows tests to run without loading PyTorch.

## Test Results

### All Tests Passing ✓

```bash
.venv\Scripts\python.exe -m pytest tests/unit/ -v
```

**Results:**
- ✅ **47 tests passed** (27 utils + 20 conversions)
- ✅ **100% coverage** on [utils.py](src/digitalsreeni_image_annotator/utils.py)
- ✅ **16% coverage** on [image_label.py](src/digitalsreeni_image_annotator/image_label.py) (get_image_coordinates method)
- ✅ **2% overall coverage** (baseline established)

### Test Suite Breakdown

1. **[tests/unit/test_utils.py](tests/unit/test_utils.py)** - 27 tests
   - `TestCalculateArea`: 9 tests (polygons, bboxes, edge cases)
   - `TestCalculateBbox`: 9 tests (various polygon shapes)
   - `TestNormalizeImage`: 9 tests (8-bit, 16-bit, float conversion)

2. **[tests/unit/test_conversions.py](tests/unit/test_conversions.py)** - 20 tests
   - `TestGetImageCoordinates`: 11 tests (zoom, pan, screen-to-image)
   - `TestCoordinateConversionProperties`: 6 tests (parametrized)
   - `TestEdgeCases`: 3 tests (edge cases)

3. **[tests/integration/test_export_formats.py](tests/integration/test_export_formats.py)** - 20+ tests (not yet run)
   - COCO JSON export
   - YOLO format export
   - Pascal VOC export
   - Multi-dimensional slices

## Files Modified

### Configuration Files
- [requirements.txt](requirements.txt) - Updated to `numpy>=2.4.0`, flexible versions
- [setup.py](setup.py) - Changed pinned versions to flexible constraints
- [pytest.ini](pytest.ini) - Created pytest configuration

### Test Files Created
- [tests/](tests/) - Test directory structure
  - [tests/conftest.py](tests/conftest.py) - Pytest fixtures
  - [tests/unit/test_utils.py](tests/unit/test_utils.py) - 27 utility function tests
  - [tests/unit/test_conversions.py](tests/unit/test_conversions.py) - 20 coordinate conversion tests
  - [tests/integration/test_export_formats.py](tests/integration/test_export_formats.py) - 20+ export tests

### Documentation
- [TESTING.md](TESTING.md) - Complete testing guide
- [PYTHON314_SETUP.md](PYTHON314_SETUP.md) - This file
- [.github/workflows/tests.yml](.github/workflows/tests.yml) - CI/CD pipeline

## How to Run Tests

### Using Virtual Environment

```bash
# Activate venv (Windows)
.venv\Scripts\activate

# Run all unit tests
pytest tests/unit/ -v

# Run specific test file
pytest tests/unit/test_utils.py -v

# Run with coverage
pytest tests/unit/ -v --cov=src/digitalsreeni_image_annotator --cov-report=html

# View coverage report
start htmlcov/index.html
```

### Direct Execution (without activating venv)

```bash
# Windows
.venv\Scripts\python.exe -m pytest tests/unit/ -v

# Run single test
.venv\Scripts\python.exe -m pytest tests/unit/test_utils.py::TestCalculateArea::test_polygon_area_square -v
```

## Dependencies Installed in .venv

### Core Dependencies
- PyQt5 5.15.11
- numpy 2.4.2 (Python 3.14 compatible)
- Pillow 12.1.0
- opencv-python 4.13.0.90
- shapely 2.1.2
- ultralytics 8.4.9 (with torch 2.10.0)
- scikit-image 0.26.0
- And 20+ other dependencies

### Test Dependencies
- pytest 9.0.2
- pytest-qt 4.5.0
- pytest-cov 7.0.0
- pytest-mock 3.15.1
- coverage 7.13.2

## Known Limitations

### PyTorch Integration Tests

Tests that require SAM (Segment Anything Model) or torch will currently fail due to DLL loading issues. Workarounds:

1. **Mock torch/SAM** in tests (future work)
2. **Skip torch-dependent tests** on Python 3.14 (future work)
3. **Wait for PyTorch update** with full Python 3.14 support

### CI/CD Pipeline

The GitHub Actions workflow ([.github/workflows/tests.yml](.github/workflows/tests.yml)) is configured for Python 3.10, 3.11, 3.12. To add Python 3.14:

1. Wait for PyTorch to add Windows Python 3.14 support
2. Update workflow matrix to include `'3.14'`

## Next Steps

### Milestone 1.1 Complete ✓

- ✅ Test infrastructure setup
- ✅ Python 3.14 compatibility
- ✅ 47 unit tests passing
- ✅ 100% coverage on utils.py
- ✅ CI/CD pipeline configured

### Milestone 1.2: PyQt6 Migration (Next)

1. Update all PyQt5 imports to PyQt6
2. Handle Qt enum changes (Qt.AlignLeft → Qt.AlignmentFlag.AlignLeft)
3. Update signal/slot syntax
4. Test for 100% feature parity
5. Update documentation

### Future Testing Work

1. **Add more unit tests**
   - Polygon operations (Shapely)
   - QImage ↔ NumPy conversions
   - Export format helpers

2. **Add integration tests**
   - Test export formats (when ready)
   - Test project save/load
   - Test annotation workflows

3. **Add UI tests** (pytest-qt)
   - Test annotation creation
   - Test SAM integration (when torch works)
   - Test video loading (Phase 2)

## Verification Commands

```bash
# Verify Python version
.venv\Scripts\python.exe --version
# Output: Python 3.14.2

# Verify numpy version
.venv\Scripts\python.exe -c "import numpy; print(numpy.__version__)"
# Output: 2.4.2

# Run all tests
.venv\Scripts\python.exe -m pytest tests/unit/ -v
# Output: 47 passed

# Check coverage
.venv\Scripts\python.exe -m pytest tests/unit/ --cov=src/digitalsreeni_image_annotator --cov-report=term
# Output: 2% coverage (utils.py at 100%)
```

## Resources

- [pytest documentation](https://docs.pytest.org/)
- [numpy 2.4 release notes](https://numpy.org/doc/stable/release/2.4.0-notes.html)
- [PyTorch compatibility matrix](https://pytorch.org/get-started/locally/)
- [Python 3.14 what's new](https://docs.python.org/3.14/whatsnew/3.14.html)

## Troubleshooting

### "Module not found" errors

Make sure you're using the venv Python:
```bash
.venv\Scripts\python.exe -m pytest tests/unit/ -v
```

### Torch DLL errors

This is expected with Python 3.14. Tests are designed to work around this by importing modules directly.

### Coverage warnings

If you see "already imported", this is normal due to our direct import workaround. Coverage still tracks correctly.

---

**Status**: ✅ Ready for PyQt6 Migration (Milestone 1.2)
