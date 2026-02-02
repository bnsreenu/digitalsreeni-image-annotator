# Risks and Technical Debt

## Technical Risks

### Linux Platform Support

**Risk Level**: Medium

**Description**: Application has limited testing on Linux, known XCB plugin issues

**Impact**:
- May not work correctly on Linux systems
- Potential crashes or rendering issues
- Limits user base

**Mitigation**:
- Environment variable workaround in main.py (removes `QT_QPA_PLATFORM_PLUGIN_PATH`)
- Document Windows/macOS as primary platforms
- Community testing and feedback

**Future Action**: Dedicated Linux testing and fixes

---

### SAM 2 Large Model Crashes

**Risk Level**: Medium

**Description**: SAM 2 large model can crash application on systems with limited RAM

**Impact**:
- Application termination
- Loss of unsaved work
- Poor user experience

**Mitigation**:
- Documentation recommends tiny/small models
- UI warns about large model
- Autosave reduces data loss

**Future Action**:
- Add RAM detection and warning
- Catch OOM exceptions gracefully

---

### Project File Portability

**Risk Level**: Low-Medium

**Description**: Projects store absolute paths, not portable between machines

**Impact**:
- Cannot share projects easily
- Moving images breaks projects
- Collaboration difficult

**Mitigation**:
- Export functions copy images
- Users can manually update paths in JSON

**Future Action**: Consider relative paths or image embedding option

---

### Large Image Memory Usage

**Risk Level**: Medium

**Description**: Loading very large images or many slices can exhaust memory

**Impact**:
- Application slowdown
- Potential crashes
- Poor performance

**Mitigation**:
- Slice-by-slice loading for multi-dimensional images
- Image downsampling for display (future)
- Lazy loading (future)

**Current Limitation**: All slices loaded into memory

---

## Technical Debt

### No Automated Tests

**Debt Level**: High

**Description**: Zero unit tests, integration tests, or UI tests

**Impact**:
- High risk of regressions
- Refactoring is dangerous
- Manual testing burden
- Slow development velocity

**Effort to Resolve**: High (months)

**Priority**: Medium

**Plan**:
1. Add unit tests for utility functions first (low-hanging fruit)
2. Add integration tests for export/import
3. Consider pytest-qt for critical UI flows

---

### Inconsistent Error Handling

**Debt Level**: Medium

**Description**: Mix of exceptions, return values, and UI warnings

**Examples**:
```python
# Some functions raise exceptions
raise ValueError("Invalid dimension")

# Some show message boxes
QMessageBox.warning(self, "Error", "...")

# Some return None
return None
```

**Impact**:
- Inconsistent user experience
- Hard to predict error behavior
- Difficult to add global error handling

**Effort to Resolve**: Medium (weeks)

**Priority**: Low

**Plan**: Standardize on exception-based approach with top-level handler

---

### Print Statements for Logging

**Debt Level**: Low

**Description**: Uses `print()` instead of proper logging framework

**Impact**:
- Cannot control log levels
- Cannot redirect logs
- Hard to debug production issues
- Console spam

**Effort to Resolve**: Low (days)

**Priority**: Low

**Plan**: Replace with `logging` module

---

### Tight Coupling Between ImageAnnotator and ImageLabel

**Debt Level**: Medium

**Description**: ImageLabel has `main_window` reference and calls methods directly

**Examples**:
```python
# In ImageLabel
self.main_window.add_annotation(polygon)
self.main_window.update_annotation_list()
```

**Impact**:
- Hard to test ImageLabel independently
- Changes ripple between classes
- Circular dependency concerns

**Effort to Resolve**: Medium (refactor to signals/slots)

**Priority**: Low

**Plan**: Refactor to Qt signals for loose coupling

---

### Duplicate Code in Export Functions

**Debt Level**: Low

**Description**: Export formats share similar code (image copying, directory creation)

**Impact**:
- Bug fixes must be applied multiple times
- Inconsistent behavior across formats
- More maintenance

**Effort to Resolve**: Low (extract common functions)

**Priority**: Low

---

### No Type Hints

**Debt Level**: Medium

**Description**: Python code lacks type hints

**Impact**:
- No static type checking
- Harder to understand function contracts
- More runtime errors

**Effort to Resolve**: High (add gradually)

**Priority**: Low

**Plan**: Add type hints to new code, gradually backfill

---

### Hardcoded UI Strings

**Debt Level**: Low

**Description**: No internationalization (i18n) support

**Impact**:
- Cannot translate to other languages
- Limits international user base

**Effort to Resolve**: Medium (Qt has i18n support)

**Priority**: Very Low (no current demand)

---

## Known Issues

### YOLO Training Not Supported for Multi-dimensional Images

**Status**: Known Limitation

**Description**: YOLO training only works with single images, not TIFF/CZI slices

**Workaround**: Export slices as individual images first

**Priority**: Low (niche use case)

---

### Annotation Merge Only Works for Connected Regions

**Status**: Known Limitation

**Description**: Merge tool requires annotations to overlap or touch

**Workaround**: Use paint brush to connect regions first

**Priority**: Low

---

### SAM Point Mode Requires Manual Confirmation

**Status**: By Design

**Description**: User must press Enter to accept SAM prediction

**Rationale**: Allows user to add more points or reject prediction

**Priority**: N/A (intentional)

---

### Autosave Doesn't Ask for File Location

**Status**: Known Behavior

**Description**: Autosave only works after first manual save

**Impact**: New projects lose autosave protection until first save

**Priority**: Low

---

## Upstream Fork Divergence

**Risk Level**: Medium

**Description**: This is a fork of https://github.com/bnsreenu/digitalsreeni-image-annotator

**Impact**:
- May miss upstream features
- May miss upstream bug fixes
- Merge conflicts on updates

**Mitigation**:
- Document fork-specific changes
- Periodically review upstream
- Consider contributing changes back

**Current Fork-Specific Changes**:
- (Document any fork-specific features here)

---

## Security Considerations

### No Input Validation on JSON Loading

**Risk Level**: Low

**Description**: Project JSON files loaded without strict schema validation

**Impact**:
- Malformed files can crash application
- Potential for malicious project files

**Mitigation**:
- Projects are local files (user-controlled)
- Try-catch around JSON loading

**Priority**: Low (desktop app, local files)

---

### Arbitrary File Paths in Projects

**Risk Level**: Low

**Description**: Project files can reference any file path

**Impact**:
- Could load unintended files
- Path traversal (theoretical)

**Mitigation**:
- Desktop app (user has filesystem access anyway)
- File existence checks before loading

**Priority**: Very Low
