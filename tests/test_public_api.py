"""
Sanity checks for the public package API (`pyhearts.__init__` exports).
"""

import pyhearts


def test_public_version_string():
    assert isinstance(pyhearts.__version__, str)
    assert pyhearts.__version__.count(".") >= 1


def test_public_exports_exist():
    # Core exports
    assert hasattr(pyhearts, "PyHEARTS")
    assert hasattr(pyhearts, "ProcessCycleConfig")
    assert hasattr(pyhearts, "load_monitor_csv")

    # Submodules are part of the public API
    assert hasattr(pyhearts, "feature")
    assert hasattr(pyhearts, "processing")
    assert hasattr(pyhearts, "plots")
    assert hasattr(pyhearts, "fitmetrics")
    assert hasattr(pyhearts, "io")




