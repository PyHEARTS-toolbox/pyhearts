"""
Sanity checks for the public package API (`pyhearts.__init__` exports).
"""

from importlib.metadata import version

import pyhearts
from pyhearts.core.analyzer import PyHEARTS as CorePyHEARTS


def test_public_version_string():
    assert pyhearts.__version__ == "1.0.0"
    assert version("pyhearts") == pyhearts.__version__


def test_public_exports_exist():
    assert hasattr(pyhearts, "PyHEARTS")
    assert hasattr(pyhearts, "load_monitor_csv")
    assert hasattr(pyhearts, "load_wfdb_signal")
    # Advanced config is available under pyhearts.config, not the package root.
    assert "ProcessCycleConfig" not in pyhearts.__all__
    assert not hasattr(pyhearts, "CoreProcessCycleConfig")

    assert hasattr(pyhearts, "feature")
    assert hasattr(pyhearts, "processing")
    assert hasattr(pyhearts, "plots")
    assert not hasattr(pyhearts, "fitmetrics")
    assert hasattr(pyhearts, "io")
    assert pyhearts.PyHEARTS is CorePyHEARTS
