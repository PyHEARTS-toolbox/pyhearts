"""
PyHEARTS: Python Heart Evaluation and Analysis for Rhythm and Temporal Shape.

Beat-by-beat ECG waveform morphology mapping for interpretable machine learning and AI.
"""

from .version import __version__

# Core classes
from .config import ProcessCycleConfig
from .core.fit import PyHEARTS

# Submodules
from . import feature
from . import fitmetrics
from . import plots
from . import processing

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
    # Utilities (optional)
    "generate_ecg_signal",
    # Submodules
    "feature",
    "fitmetrics",
    "plots",
    "processing",
]
