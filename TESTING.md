# Testing Infrastructure

## Overview

This document describes the testing infrastructure for the DigitalSreeni Image Annotator project.

## Phase 1, Milestone 1.1: Testing Infrastructure ✓

### Completed Tasks

1. **Test Dependencies** ✓
   - Added pytest, pytest-qt, pytest-cov, pytest-mock to [requirements.txt](requirements.txt)
   - Using flexible version constraints (>=) instead of pinned versions

2. **Test Directory Structure** ✓
   ```
   tests/
   ├── __init__.py
   ├── conftest.py               # Pytest configuration and fixtures
   ├── unit/
   │   ├── __init__.py
   │   ├── test_utils.py         # Tests for utility functions
   │   └── test_conversions.py   # Tests for coordinate conversions
   ├── integration/
   │   ├── __init__.py
   │   └── test_export_formats.py # Tests for COCO, YOLO, Pascal VOC exports
   └── ui/
       └── __init__.py            # UI tests (TBD)
   ```

3. **Pytest Configuration** ✓
   - Created [pytest.ini](pytest.ini) with:
     - Test discovery patterns
     - Coverage settings
     - Custom markers (unit, integration, ui, slow)

4. **Unit Tests** ✓
   - **[tests/unit/test_utils.py](tests/unit/test_utils.py)**: 25+ test cases
     - `TestCalculateArea`: 10 tests for polygon and bbox area calculations
     - `TestCalculateBbox`: 9 tests for bounding box extraction from polygons
     - `TestNormalizeImage`: 11 tests for image normalization (8-bit, 16-bit, float)

   - **[tests/unit/test_conversions.py](tests/unit/test_conversions.py)**: 20+ test cases
     - `TestGetImageCoordinates`: Screen-to-image coordinate conversion tests
     - `TestCoordinateConversionProperties`: Property-based tests with various zoom/offset values
     - `TestEdgeCases`: Edge cases (zero zoom, negative coordinates, large values)

5. **Integration Tests** ✓
   - **[tests/integration/test_export_formats.py](tests/integration/test_export_formats.py)**: 20+ test cases
     - `TestCOCOExport`: Tests for COCO JSON format export
     - `TestYOLOExport`: Tests for YOLO format export
     - `TestPascalVOCExport`: Tests for Pascal VOC XML export
     - `TestExportWithSlices`: Tests for multi-dimensional image slice export
     - `TestExportEdgeCases`: Edge case handling

6. **CI/CD Pipeline** ✓
   - Created [.github/workflows/tests.yml](.github/workflows/tests.yml)
   - Multi-platform testing: Ubuntu, Windows, macOS
   - Multi-version testing: Python 3.10, 3.11, 3.12
   - Automated coverage reporting (Codecov integration)
   - Coverage report artifacts

## Running Tests

### Prerequisites

Install test dependencies:
```bash
pip install pytest pytest-qt pytest-cov pytest-mock
```

Or install all dependencies including tests:
```bash
pip install -e .
```

### Run All Tests

```bash
pytest tests/ -v
```

### Run Specific Test Suites

```bash
# Unit tests only
pytest tests/unit -v

# Integration tests only
pytest tests/integration -v

# UI tests only (when available)
pytest tests/ui -v
```

### Run with Coverage

```bash
# Generate coverage report
pytest tests/ --cov=src/digitalsreeni_image_annotator --cov-report=term-missing --cov-report=html

# View HTML coverage report
# Open htmlcov/index.html in browser
```

### Run Specific Test Files

```bash
pytest tests/unit/test_utils.py -v
pytest tests/unit/test_conversions.py -v
pytest tests/integration/test_export_formats.py -v
```

### Run Tests by Marker

```bash
# Run only unit tests
pytest -m unit -v

# Run only integration tests
pytest -m integration -v

# Skip slow tests
pytest -m "not slow" -v
```

## Test Coverage Goals

- **Phase 1 Target**: 60% code coverage
- **Phase 2 Target**: 80% code coverage
- **Phase 3 Target**: 90% code coverage

## Known Issues

### Python 3.14 + PyTorch Compatibility

**Issue**: PyTorch (torch) has DLL loading issues with Python 3.14 on Windows, causing access violations when importing ultralytics/SAM.

**Workaround**: Tests use `importlib.util.spec_from_file_location()` to import modules directly by file path, bypassing the package `__init__.py` that imports torch. This allows unit tests to run without loading PyTorch.

**Impact**:
- ✓ Unit tests work fine (utils, conversions)
- ✓ Integration tests that don't use SAM work
- ⚠️ Tests requiring SAM/torch will need mocking or skipping until PyTorch adds full Python 3.14 support

**Dependencies updated for Python 3.14**:
- `numpy>=2.4.0` (Python 3.14 requires numpy 2.4+)
- Other dependencies use latest compatible versions

### Virtual Environment

To use the project's .venv with Python 3.14:
```bash
# Windows
.venv\Scripts\activate
.venv\Scripts\python.exe -m pytest tests/unit/ -v

# Linux/macOS
source .venv/bin/activate
python -m pytest tests/unit/ -v
```

## Next Steps

### Milestone 1.2: PyQt6 Migration

- Update all PyQt5 imports to PyQt6
- Handle Qt enum changes
- Test for feature parity
- Update documentation

### Future Testing Work

1. **Add UI Tests** (pytest-qt)
   - Test annotation creation workflows
   - Test project save/load
   - Test SAM integration
   - Test video loading (Phase 2)

2. **Add Performance Tests**
   - Benchmark critical operations
   - Video frame extraction speed
   - SAM inference latency
   - Batch processing throughput

3. **Increase Coverage**
   - Test error handling paths
   - Test edge cases in ImageLabel
   - Test multi-dimensional image handling
   - Test export format edge cases

4. **Add Regression Tests**
   - Test backward compatibility
   - Test project migration (v1 → v2)

## Resources

- [pytest documentation](https://docs.pytest.org/)
- [pytest-qt documentation](https://pytest-qt.readthedocs.io/)
- [pytest-cov documentation](https://pytest-cov.readthedocs.io/)
- [Coverage.py documentation](https://coverage.readthedocs.io/)

## Contributing

When adding new features:
1. Write tests first (TDD approach preferred)
2. Ensure all tests pass: `pytest tests/ -v`
3. Check coverage: `pytest tests/ --cov=src/digitalsreeni_image_annotator`
4. Aim for 80%+ coverage on new code
5. Add docstrings to test functions

When fixing bugs:
1. Write a failing test that reproduces the bug
2. Fix the bug
3. Verify the test passes
4. Check that other tests still pass
