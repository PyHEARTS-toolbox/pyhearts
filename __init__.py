"""
PyHEARTS: Python Heart Evaluation and Analysis for Rhythm and Temporal Shape.

Beat-by-beat ECG waveform morphology mapping for interpretable machine learning and AI.
"""

from .version import __version__

# Core classes
from .config import ProcessCycleConfig
from .core.fit import PyHEARTS

# Signal generation
from .sim import generate_ecg_signal

# Submodules
from . import feature
from . import fitmetrics
from . import plots
from . import processing

__all__ = [
    # Version
    "__version__",
    # Core
    "PyHEARTS",
    "ProcessCycleConfig",
    # Utilities
    "generate_ecg_signal",
    # Submodules
    "feature",
    "fitmetrics",
    "plots",
    "processing",
]
