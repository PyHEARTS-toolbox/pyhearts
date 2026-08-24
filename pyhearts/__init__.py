"""
PyHEARTS: Python Heart Evaluation and Analysis for Rhythm and Temporal Shape.

Beat-by-beat ECG morphology analysis for research use. The main entry point is
:class:`~pyhearts.core.analyzer.PyHEARTS`.

Typical workflow
----------------
1. ``analyzer = PyHEARTS(sampling_rate=..., species="human")``
2. Optionally preprocess with ``analyzer.preprocess_signal(...)``
3. ``features, cycles = analyzer.analyze_ecg(ecg)``
4. Optionally ``analyzer.save_output(file_id, results_dir)``

See the package README and ``docs/PRESETS.md`` for install and preset details.
"""

# Submodules
from . import feature, io, processing

# Core classes
from .core.analyzer import PyHEARTS

# I/O helpers
from .io import (
    load_monitor_csv,
    load_wfdb_signal,
    pick_lead_index,
    pick_manual_annotation_ext,
)
from .version import __version__

_HAS_SIM: bool | None = None

__all__ = [
    # Version
    "__version__",
    # Core
    "PyHEARTS",
    # I/O
    "load_monitor_csv",
    "load_wfdb_signal",
    "pick_lead_index",
    "pick_manual_annotation_ext",
    # Utilities (optional)
    "generate_ecg_signal",
    # Submodules
    "feature",
    "io",
    "plots",
    "processing",
]


def __getattr__(name: str):
    """Lazy-load optional heavy submodules (plots / sim) until accessed."""
    global _HAS_SIM

    if name == "plots":
        import importlib

        return importlib.import_module(".plots", __name__)

    if name == "generate_ecg_signal":
        try:
            from .sim import generate_ecg_signal as _generate_ecg_signal
        except (ImportError, TypeError):
            _HAS_SIM = False
            globals()["generate_ecg_signal"] = None
            return None
        _HAS_SIM = True
        globals()["generate_ecg_signal"] = _generate_ecg_signal
        return _generate_ecg_signal

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

