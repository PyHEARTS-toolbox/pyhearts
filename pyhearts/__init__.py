"""
PyHEARTS: Python Heart Evaluation and Analysis for Rhythm and Temporal Shape.

Beat-by-beat ECG waveform morphology mapping for interpretable machine learning and AI.
"""

# Submodules
from . import feature, fitmetrics, io, plots, processing
from ._legacy2025 import ProcessCycleConfig as CoreProcessCycleConfig

# Core classes
from .config import ProcessCycleConfig
from .core.hybrid import PyHEARTS

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
    "ProcessCycleConfig",
    "CoreProcessCycleConfig",
    # I/O
    "load_monitor_csv",
    "load_wfdb_signal",
    "pick_lead_index",
    "pick_manual_annotation_ext",
    # Utilities (optional)
    "generate_ecg_signal",
    # Submodules
    "feature",
    "fitmetrics",
    "io",
    "plots",
    "processing",
]
