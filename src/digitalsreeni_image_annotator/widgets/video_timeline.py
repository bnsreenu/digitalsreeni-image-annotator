"""Video timeline scrub bar with annotated-frame markers (issue #48).

A dumb, self-contained widget with **no main-window reference**: it renders a
horizontal scrub slider, a thin marker strip that ticks every annotated frame
plus the current one, and a ``F i/N  •  MM:SS / MM:SS`` position label. It is a
pure VIEW — it never changes any frame itself. User interaction emits
``frameSelected(idx)`` and the orchestrator routes that through the normal
``ImageController.switch_slice`` path; programmatic sync comes back via
``set_current_frame`` (which must NOT re-emit ``frameSelected``, or it would
re-enter ``switch_slice``).

Colours come from the widget PALETTE (``highlight`` / ``mid`` / ``text``), never
colour literals (No Hardcoded Colors Rule, CLAUDE.md). Note the app themes via a
QSS stylesheet and never calls ``setPalette``, so ``highlight``/``mid`` resolve
to the *static default* palette (a saturated accent + mid-grey that both read on
the light and soft-dark backgrounds), while the ``text`` role — used for the
high-contrast current-frame tick — does follow the stylesheet's ``color`` rule.
The marks are legible in both themes; the current tick additionally tracks it.

Issue #51 extends this with per-frame STATES (``annotated`` / ``tracked`` /
``needs_review``) painted as contiguous coloured segments; the plain
``set_annotated_frames`` set-based API still works, delegating to the states
model with every frame marked ``"annotated"``.
"""

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

# fps guard: containers sometimes report 0/NaN fps; treat as 30 for MM:SS.
_DEFAULT_FPS = 30.0

# The per-frame states the marker strip understands (issue #51). Any other
# value is ignored by the painter (treated as unmarked).
FRAME_STATES = ("annotated", "tracked", "needs_review")


def compute_state_runs(states, total):
    """Collapse a ``{frame_idx: state}`` map into maximal contiguous runs.

    Returns a list of ``(start_idx, end_idx, state)`` in ascending order, one
    per run of consecutive frame indices that share the same state. A gap in
    the indices or a change of state starts a new run. Frames outside
    ``[0, total)`` are dropped. Pure logic (no Qt) so the run computation is
    unit-testable without painting. (issue #51)
    """
    runs = []
    for idx in sorted(states):
        if not (0 <= idx < total):
            continue
        state = states[idx]
        if runs and runs[-1][1] + 1 == idx and runs[-1][2] == state:
            runs[-1][1] = idx
        else:
            runs.append([idx, idx, state])
    return [(start, end, state) for start, end, state in runs]


class _MarkerStrip(QWidget):
    """Thin painted strip that colours per-frame state segments + the current
    frame.

    Holds only its own render state (total / current / states map); the parent
    :class:`VideoTimeline` pushes updates via :meth:`configure`. Same-state
    consecutive frames are painted as one contiguous segment (issue #51).
    """

    _STRIP_HEIGHT = 6

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(self._STRIP_HEIGHT)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._total = 0
        self._current = 0
        self._states = {}

    def configure(self, total, current, states):
        self._total = total
        self._current = current
        self._states = dict(states)
        self.update()

    def _x_for(self, idx, width):
        # Guard total <= 1: a single (or zero) frame maps to the left edge
        # rather than dividing by zero.
        if self._total <= 1:
            return 0
        return int(idx / (self._total - 1) * (width - 1))

    @staticmethod
    def _state_color(pal, state):
        """Palette-derived colour per state — no hex literals (dark-mode rule).

        - ``annotated``   → ``highlight`` (the saturated accent, unchanged).
        - ``tracked``     → a **desaturated** highlight (same hue, muted) so a
          machine-propagated frame reads distinct from a hand-annotated one
          without introducing a hardcoded colour.
        - ``needs_review``→ the accent hue rotated to a contrasting warm tone
          (derived from ``highlight``, not a literal) so it reads distinct from
          BOTH the annotated segments and the ``text``-coloured current-frame
          tick (``brightText`` collides with ``text`` on dark palettes).
        """
        if state == "tracked":
            base = pal.highlight().color()
            h, s, v, a = base.getHsv()
            return QColor.fromHsv(h, int(s * 0.4), v, a)
        if state == "needs_review":
            base = pal.highlight().color()
            h, s, v, a = base.getHsv()
            # Rotate the accent hue ~150° (e.g. blue accent → warm amber) for a
            # warning-ish tone that contrasts both other states + the tick.
            return QColor.fromHsv((h + 150) % 360, s, v, a)
        return pal.highlight().color()

    def paintEvent(self, event):
        painter = QPainter(self)
        pal = self.palette()
        width = self.width()
        height = self.height()

        # Track baseline (mid) — theme-aware grey.
        painter.setPen(QPen(pal.mid().color(), 1))
        mid_y = height // 2
        painter.drawLine(0, mid_y, width, mid_y)

        if self._total <= 0:
            return

        # Contiguous same-state runs painted as filled segments.
        for start, end, state in compute_state_runs(self._states, self._total):
            color = self._state_color(pal, state)
            x0 = self._x_for(start, width)
            x1 = self._x_for(end, width)
            painter.fillRect(x0, 0, max(1, x1 - x0 + 1), height, color)

        # Current frame: a distinct, high-contrast 2px tick. `text()` always
        # contrasts the widget background in both themes, so it reads clearly
        # over the accent-coloured state segments.
        if 0 <= self._current < self._total:
            painter.setPen(QPen(pal.text().color(), 2))
            x = self._x_for(self._current, width)
            painter.drawLine(x, 0, x, height)


class VideoTimeline(QWidget):
    """Scrub slider + annotated-frame markers + position label (issue #48).

    Public API (relied on by the wiring in :mod:`annotator_window` and the
    #51 tracking controller):

    - ``set_video(total_frames: int, fps: float)``
    - ``set_current_frame(idx: int)`` — programmatic, never emits ``frameSelected``
    - ``set_frame_states(states: dict[int, str])`` — per-frame states (#51)
    - ``set_annotated_frames(indices: set[int])`` — back-compat, delegates to
      ``set_frame_states`` with every index marked ``"annotated"``
    - ``frame_state_runs()`` — the computed ``(start, end, state)`` segments
    - ``clear()``
    - signal ``frameSelected = pyqtSignal(int)`` — user interaction only
    - attribute ``frame_states`` — the stored ``{idx: state}`` map (#51)
    - attribute ``annotated_frames`` — the stored marked-index set (all states)
    """

    frameSelected = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._total_frames = 0
        self._fps = _DEFAULT_FPS
        self._current_frame = 0
        # Re-entrancy guard: set True around programmatic slider.setValue so the
        # resulting valueChanged never re-emits frameSelected (feedback loop).
        self._updating = False
        # Per-frame states (issue #51): {idx: "annotated"|"tracked"|"needs_review"}.
        self.frame_states = {}
        # Back-compat alias — the set of ALL marked frame indices (any state).
        self.annotated_frames = set()

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        # Strip stacked directly on top of the slider so ticks line up with the
        # groove.
        bar = QVBoxLayout()
        bar.setContentsMargins(0, 0, 0, 0)
        bar.setSpacing(0)

        self.strip = _MarkerStrip()
        bar.addWidget(self.strip)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        # Never swallow the arrow keys used for slice navigation.
        self.slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        bar.addWidget(self.slider)

        root.addLayout(bar, 1)

        self.label = QLabel()
        root.addWidget(self.label)

        # valueChanged covers keyboard/track-click/programmatic-user moves;
        # gated on `not _updating` (no feedback loop) and `not isSliderDown`
        # (throttle live scrub — each switch triggers a lazy frame decode, so we
        # emit the drag's final position via sliderReleased instead of every
        # intermediate frame).
        self.slider.valueChanged.connect(self._on_value_changed)
        self.slider.sliderReleased.connect(self._on_slider_released)

        self._update_label()

    # ── configuration ───────────────────────────────────────────────────────

    def set_video(self, total_frames, fps):
        """Configure for a video of ``total_frames`` frames at ``fps``.

        Resets the marker set and the slider range (0..max(0, total-1)); the fps
        guard keeps MM:SS finite when a container reports 0/NaN fps.
        """
        self._total_frames = max(0, int(total_frames))
        self._fps = fps if fps and fps > 0 else _DEFAULT_FPS
        self._current_frame = 0
        self.frame_states = {}
        self.annotated_frames = set()

        self._updating = True
        self.slider.setMaximum(max(0, self._total_frames - 1))
        self.slider.setValue(0)
        self._updating = False

        self._refresh_strip()
        self._update_label()

    def set_current_frame(self, idx):
        """Programmatically sync the slider/label/strip to frame ``idx``.

        MUST NOT emit ``frameSelected`` — this is called from ``switch_slice``
        and emitting would re-enter it (feedback loop). The ``_updating`` guard
        suppresses the valueChanged emission.
        """
        self._current_frame = int(idx)
        self._updating = True
        self.slider.setValue(self._current_frame)
        self._updating = False
        self._refresh_strip()
        self._update_label()

    def set_frame_states(self, states):
        """Store the per-frame ``{idx: state}`` map and repaint the strip (#51).

        ``state`` is one of ``"annotated"`` / ``"tracked"`` / ``"needs_review"``
        (see :data:`FRAME_STATES`); unknown values are ignored by the painter.
        Keeps :attr:`annotated_frames` in sync as the set of all marked indices
        so the older set-based consumers/tests still read the marked set.
        """
        self.frame_states = dict(states)
        self.annotated_frames = set(self.frame_states.keys())
        self._refresh_strip()

    def set_annotated_frames(self, indices):
        """Back-compat: mark ``indices`` as plain ``"annotated"`` frames.

        Delegates to :meth:`set_frame_states` so the single set-based call site
        (and its tests) keep working after the #51 states migration.
        """
        self.set_frame_states({int(i): "annotated" for i in indices})

    def frame_state_runs(self):
        """The computed ``(start, end, state)`` contiguous segments (#51).

        Exposed for tests + any consumer that needs the run decomposition
        without reaching into the painter.
        """
        return compute_state_runs(self.frame_states, self._total_frames)

    def clear(self):
        """Reset to the empty state (no video)."""
        self._total_frames = 0
        self._fps = _DEFAULT_FPS
        self._current_frame = 0
        self.frame_states = {}
        self.annotated_frames = set()
        self._updating = True
        self.slider.setMaximum(0)
        self.slider.setValue(0)
        self._updating = False
        self._refresh_strip()
        self._update_label()

    # ── user-interaction handlers ───────────────────────────────────────────

    def _on_value_changed(self, value):
        # Suppress programmatic syncs (feedback loop) and live-drag intermediate
        # steps (throttle — sliderReleased emits the final drag position).
        if self._updating or self.slider.isSliderDown():
            return
        self._emit(value)

    def _on_slider_released(self):
        self._emit(self.slider.value())

    def _emit(self, value):
        self._current_frame = int(value)
        self._refresh_strip()
        self._update_label()
        self.frameSelected.emit(int(value))

    # ── rendering helpers ───────────────────────────────────────────────────

    def _refresh_strip(self):
        self.strip.configure(
            self._total_frames, self._current_frame, self.frame_states
        )

    def _fmt_time(self, idx):
        fps = self._fps if self._fps and self._fps > 0 else _DEFAULT_FPS
        minutes, seconds = divmod(int(idx / fps), 60)
        return f"{minutes:02d}:{seconds:02d}"

    def _update_label(self):
        if self._total_frames <= 0:
            self.label.setText("")
            return
        last = self._total_frames - 1
        self.label.setText(
            f"F {self._current_frame + 1}/{self._total_frames}  •  "
            f"{self._fmt_time(self._current_frame)} / {self._fmt_time(last)}"
        )
