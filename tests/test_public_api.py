"""
Sanity checks for the public package API (`pyhearts.__init__` exports).
"""

from importlib.metadata import version

import pyhearts
from pyhearts.core.hybrid import PyHEARTS as HybridPyHEARTS


def test_public_version_string():
    assert pyhearts.__version__ == "2.0.0"
    assert version("pyhearts") == pyhearts.__version__


def test_public_exports_exist():
    # Core exports
    assert hasattr(pyhearts, "PyHEARTS")
    assert hasattr(pyhearts, "ProcessCycleConfig")
    assert hasattr(pyhearts, "CoreProcessCycleConfig")
    assert hasattr(pyhearts, "load_monitor_csv")

    # Submodules are part of the public API
    assert hasattr(pyhearts, "feature")
    assert hasattr(pyhearts, "processing")
    assert hasattr(pyhearts, "plots")
    assert hasattr(pyhearts, "fitmetrics")
    assert hasattr(pyhearts, "io")
    assert pyhearts.PyHEARTS is HybridPyHEARTS
