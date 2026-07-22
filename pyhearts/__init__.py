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
from . import feature, io, plots, processing

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

# Signal generation (optional - requires neurokit2 which needs Python 3.10+)
try:
    from .sim import generate_ecg_signal

    _HAS_SIM = True
except (ImportError, TypeError):
    generate_ecg_signal = None  # type: ignore[assignment, misc]
    _HAS_SIM = False

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
