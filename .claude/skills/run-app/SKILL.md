---
name: run-app
description: Launch the DigitalSreeni Image Annotator headlessly and run its smoke tests. Use when asked to run/launch the app, verify it starts after a change, or check for import/startup errors in a display-less (web/CI) session.
---

# Run the app headlessly + smoke tests

This is a PyQt6 GUI app; sessions without a display must use Qt's offscreen
platform plugin. GUI interaction is impossible offscreen — a clean launch
proves imports, window construction, and signal wiring, nothing more.

## Launch check (no display)

    QT_QPA_PLATFORM=offscreen timeout 10 python -m src.digitalsreeni_image_annotator.main

- Exit code 124 (timeout killed it) = SUCCESS: the app started and stayed up.
- Any traceback before the timeout = startup failure; fix before proceeding.
- Installed entry points work the same way: `timeout 10 digitalsreeni-image-annotator` / `sreeni`.

## Smoke tests (always run after code changes)

    pytest tests/integration/test_smoke.py -v

Includes the AST-based inline-import gate (ADR-016) — a "clean-looking"
launch alone is NOT sufficient after moving modules. Full suite: `pytest tests/ -v`.

## Linux runtime deps

If Qt fails to load its platform plugin, install the Qt6 runtime set
(Debian/Ubuntu): libxcb-cursor0 libegl1 libgl1 libxcb-xinerama0
libxkbcommon-x11-0 (see .github/workflows/tests.yml for the full CI list).

## Limits

Do not claim GUI behaviour (rendering, dark mode, mouse interaction) was
"verified" from an offscreen launch — say so explicitly and fall back to
the test suite or ask the user to check visually.
