"""Always-on MLflow experiment tracking for model training (issue #74).

Wraps the ``mlflow`` dependency so the rest of the app can log training runs
without sprinkling mlflow calls everywhere. MLflow is a **core dependency** and
tracking is **mandatory** — there is no enable/disable. The only no-op is
:class:`_NullTracker`, used when a trainer is invoked without a tracker
(direct/programmatic calls, tests).

``mlflow`` is imported lazily inside the methods that need it — never at module
top — mirroring the lazy-import idiom used for the other heavy libraries
(``inference/sam_utils.py``, ``inference/dino_utils.py``) so app startup stays
fast and a *broken* mlflow install can't stop the GUI from launching. Every live
mlflow call is wrapped so a tracking error degrades that run to untracked but
never aborts training.

Storage defaults to a local file store under ``<project>/mlruns`` (see
:func:`resolve_tracking_uri`), matching the issue's default.
"""

import os
import subprocess
import sys
import webbrowser
from pathlib import Path
from urllib.parse import urlparse

_DEFAULT_EXPERIMENT = "image-annotator-training"
_MLRUNS_DIRNAME = "mlruns"

# Port the bundled `mlflow ui` server listens on; the deep-link URLs built for
# the progress dialog must use the same port.
MLFLOW_UI_PORT = 5000


def run_ui_url(experiment_id, run_id, port=MLFLOW_UI_PORT) -> str:
    """Deep link to a specific run in the MLflow web UI."""
    return f"http://localhost:{port}/#/experiments/{experiment_id}/runs/{run_id}"


def to_mlflow_uri(path_or_uri) -> str:
    """Return a value MLflow accepts as a tracking URI.

    MLflow validates the URI *scheme*, so a bare Windows path such as
    ``C:\\Users\\me\\mlruns`` is read as scheme ``c`` and **rejected**
    ("unsupported URI ... for model registry data storage") — i.e. local-file
    tracking silently degrades to untracked on Windows. Local filesystem paths
    must therefore be expressed as ``file://`` URIs. Genuine URIs (``file``,
    ``http(s)``, ``sqlite``, ``databricks`` …) are returned unchanged.
    """
    text = str(path_or_uri)
    scheme = urlparse(text).scheme
    # Empty scheme = POSIX/relative path; a single-letter scheme is a Windows
    # drive (``C:``), not a real URI scheme — both are local paths.
    if len(scheme) > 1:
        return text
    return Path(text).resolve().as_uri()

# Cache the import probe — mlflow's import is non-trivial and the answer
# never changes within a process.
_AVAILABLE = None


def mlflow_available() -> bool:
    """Return True if the ``mlflow`` package can be imported.

    MLflow is a core dependency, so this is expected to be True; it exists only
    to give :func:`launch_mlflow_ui` a clean message for a broken install.
    """
    global _AVAILABLE
    if _AVAILABLE is None:
        try:
            import mlflow  # noqa: F401
            _AVAILABLE = True
        except Exception:
            _AVAILABLE = False
    return _AVAILABLE


def resolve_tracking_uri(main_window=None) -> str:
    """Resolve the MLflow tracking store path.

    Precedence:
    1. A non-empty QSettings override (``tracking/mlflow_uri``).
    2. ``<current_project_dir>/mlruns`` when a project is open.
    3. ``<cwd>/mlruns`` as a last resort.

    Returns a filesystem path (MLflow accepts a local path as its tracking
    URI for the default file store).
    """
    from ..app_settings import load_mlflow_prefs

    uri, _ = load_mlflow_prefs()
    if uri:
        return uri
    project_dir = getattr(main_window, "current_project_dir", None)
    base = project_dir if project_dir else os.getcwd()
    return os.path.join(base, _MLRUNS_DIRNAME)


class MLflowTracker:
    """An always-on MLflow run wrapper.

    Construct it (cheaply) on any thread, then call :meth:`start`,
    :meth:`log_metrics`, :meth:`log_artifact` and :meth:`end` **on the
    thread that does the training** — MLflow runs are thread-bound, so for
    SAM fine-tuning the run is started and ended inside the worker thread's
    ``train()`` call.

    Tracking is mandatory (MLflow is a core dependency), so there is no
    "disabled" mode. The only robustness concession is that any error raised
    by MLflow — including an unexpectedly broken install — is caught and
    reported via ``log`` but never propagated: a tracking failure must never
    abort a training run. When no tracker is supplied to a trainer at all
    (direct/programmatic calls, tests) a :class:`_NullTracker` no-op stands
    in — that is an internal default, not a user-facing off switch.
    """

    def __init__(
        self,
        tracking_uri,
        experiment_name=_DEFAULT_EXPERIMENT,
        run_name=None,
        log=None,
    ):
        self._uri = tracking_uri
        self._experiment = experiment_name or _DEFAULT_EXPERIMENT
        self._run_name = run_name
        self._log = log
        self._active = False  # True only between a successful start() and end()
        self._on_run_url = None  # called with the run's UI deep link on start()
        self.run_id = None
        self.experiment_id = None

    # -- internal helpers ---------------------------------------------------

    def set_log(self, log):
        """Set the progress callback used for status lines.

        Callers pass a thread-safe sink (e.g. a Qt signal's ``emit``) so the
        tracker can report from a worker thread without touching the GUI
        directly.
        """
        self._log = log

    def set_run_url_callback(self, callback):
        """Set a callback invoked with the run's MLflow-UI deep link once the
        run opens in :meth:`start`. Like ``set_log``, callers pass a thread-safe
        sink (a Qt signal's ``emit``) so the GUI work — showing the clickable
        link, launching the UI server, opening the browser — happens on the GUI
        thread, not the worker thread that runs training.
        """
        self._on_run_url = callback

    def _emit(self, msg):
        if self._log is not None:
            try:
                self._log(msg)
            except Exception:
                pass

    @property
    def active(self) -> bool:
        return self._active

    # -- public API ---------------------------------------------------------

    def start(self, params=None):
        """Open the run and log ``params``. Returns True if tracking is live.

        Tracking is always attempted. The broad ``except`` is pure
        crash-safety — a broken MLflow install or a transient backend error
        degrades this one run to untracked rather than killing the training
        job; it is not an opt-out path.
        """
        try:
            import mlflow

            # mlflow 3.x put the local file store ("./mlruns") into maintenance
            # mode and *raises* on it unless this opt-out is set — which would
            # silently degrade our documented file-store default to untracked.
            # setdefault so an explicit user choice is never overridden.
            os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
            mlflow.set_tracking_uri(to_mlflow_uri(self._uri))
            mlflow.set_experiment(self._experiment)
            # Self-heal: a prior run stranded active (e.g. killed before end())
            # would make start_run raise "Run already active" and silently
            # degrade this run to untracked. Close it first.
            if mlflow.active_run() is not None:
                mlflow.end_run()
            run = mlflow.start_run(run_name=self._run_name)
            if params:
                mlflow.log_params({k: v for k, v in params.items() if v is not None})
            self._active = True
            self.run_id = run.info.run_id
            self.experiment_id = run.info.experiment_id
            self._emit(f"MLflow tracking → {self._uri} (experiment "
                       f"'{self._experiment}').")
            # Hand the run's UI deep link to the GUI (clickable link + auto-open).
            if self._on_run_url is not None:
                try:
                    self._on_run_url(run_ui_url(self.experiment_id, self.run_id))
                except Exception:
                    pass
        except Exception as exc:  # never let tracking abort training
            self._active = False
            self._emit(f"MLflow tracking unavailable ({exc}); continuing untracked.")
        return self._active

    def log_metrics(self, metrics, step=None):
        if not self._active:
            return
        try:
            import mlflow

            for name, value in metrics.items():
                mlflow.log_metric(name, float(value), step=step)
        except Exception as exc:
            self._emit(f"MLflow metric logging failed ({exc}).")

    def log_artifact(self, path):
        if not self._active or not path:
            return
        try:
            import mlflow

            if os.path.exists(path):
                mlflow.log_artifact(path)
        except Exception as exc:
            self._emit(f"MLflow artifact logging failed ({exc}).")

    def end(self):
        if not self._active:
            return
        try:
            import mlflow

            mlflow.end_run()
        except Exception as exc:
            self._emit(f"MLflow run finalization failed ({exc}).")
        finally:
            self._active = False


class _NullTracker:
    """No-op stand-in used when a trainer is invoked without a tracker
    (direct/programmatic calls, tests). Matches :class:`MLflowTracker`'s
    surface so trainers need no ``None`` checks. This is an internal default,
    not a user-facing way to disable tracking — the GUI always supplies a
    real :class:`MLflowTracker`.
    """

    active = False
    run_id = None
    experiment_id = None

    def set_log(self, log):
        pass

    def set_run_url_callback(self, callback):
        pass

    def start(self, params=None):
        return False

    def log_metrics(self, metrics, step=None):
        pass

    def log_artifact(self, path):
        pass

    def end(self):
        pass


def start_mlflow_ui_server(tracking_uri, port=MLFLOW_UI_PORT, log=None):
    """Start the local ``mlflow ui`` server (no browser). Returns ``(ok, message)``.

    Idempotent enough for repeated calls: if the port is already serving (e.g. a
    server from an earlier run), the new process simply fails to bind and exits —
    the existing server keeps serving — so callers may guard with a flag to avoid
    the noise but a double-call is harmless.
    """
    if not mlflow_available():
        # MLflow is a core dependency, so this should never happen — but if the
        # install is broken, fail with a clear message instead of crashing.
        return (
            False,
            "MLflow could not be imported (the install may be broken). "
            "Reinstall it with:  pip install mlflow",
        )
    try:
        # Invoke via `<this-interpreter> -m mlflow` rather than the bare
        # `mlflow` console script. The import probe above proves the package
        # is importable in *this* interpreter, but the `mlflow` CLI may not be
        # on PATH (venv/conda launch quirks, frozen bundles) — so a bare
        # `mlflow` could raise FileNotFoundError even though the probe passed.
        # Going through sys.executable keeps the two in lockstep.
        subprocess.Popen(
            [sys.executable, "-m", "mlflow", "ui",
             "--backend-store-uri", to_mlflow_uri(tracking_uri),
             "--port", str(port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        msg = (f"MLflow UI server at http://localhost:{port}\n"
               f"Tracking store: {tracking_uri}\n\n"
               f"If the page doesn't load, port {port} may already be in use "
               f"by another MLflow server — check that tab.")
        if log is not None:
            log(msg)
        return True, msg
    except Exception as exc:
        return (
            False,
            f"Could not start MLflow UI: {exc}\n\n"
            "MLflow is installed but its server failed to launch — the "
            "'mlflow' command may be missing from PATH for this Python "
            "environment.",
        )


def launch_mlflow_ui(tracking_uri, port=MLFLOW_UI_PORT, open_url=None, log=None):
    """Start the ``mlflow ui`` server and open it in a browser.

    ``open_url`` overrides the page opened (defaults to the UI home) so callers
    can deep-link straight to a specific run. Returns ``(ok, message)``.
    """
    ok, msg = start_mlflow_ui_server(tracking_uri, port=port, log=log)
    if ok:
        webbrowser.open(open_url or f"http://localhost:{port}")
    return ok, msg
