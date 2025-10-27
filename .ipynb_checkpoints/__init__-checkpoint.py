# pyhearts/__init__.py
from .version import __version__

# Expose core classes/functions
from .objs.fit import PyHEARTS                          # main class
from .sim import generate_ecg_signal                    # signal generation

# NEW: expose the typed config at top level
from .config import ProcessCycleConfig                  # <- add this

# Expose submodules
from . import processing
from . import plts
from . import feature
from . import fitmetrics

# NEW: make the public API explicit
__all__ = [
    "__version__",
    "PyHEARTS",
    "ProcessCycleConfig",       # <- ensure discoverability
    "generate_ecg_signal",
    "processing",
    "plts",
    "feature",
    "fitmetrics",
]

