from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyhearts.io import load_monitor_csv


def test_load_monitor_csv_centers_adc_midpoint_and_builds_uniform_time(tmp_path: Path) -> None:
    # Two-column, headerless CSV: time, value
    p = tmp_path / "monitor.csv"
    p.write_text(
        "\n".join(
            [
                "10:00.0,8192",
                "10:00.0,8193",
                "10:00.1,8194",
                "10:00.1,8191",
            ]
        )
        + "\n"
    )

    out = load_monitor_csv(p, sampling_rate_hz=500.0, adc_midpoint=8192.0, mv_per_count=None)

    assert out.ecg.shape == (4,)
    assert np.allclose(out.ecg, np.array([0.0, 1.0, 2.0, -1.0], dtype=np.float32))
    assert out.sampling_rate_hz == 500.0

    # Time should be uniformly sampled starting at 10 min = 600s
    assert np.allclose(out.time_s, np.array([600.0, 600.002, 600.004, 600.006]))


def test_load_monitor_csv_can_infer_fs_from_endpoints(tmp_path: Path) -> None:
    p = tmp_path / "monitor.csv"
    p.write_text(
        "\n".join(
            [
                "10:00.0,100",
                "10:00.0,101",
                "10:00.1,102",
                "10:00.1,103",
            ]
        )
        + "\n"
    )

    out = load_monitor_csv(p, sampling_rate_hz=None, adc_midpoint=None, assume_uniform_sampling=True)
    assert out.meta["fs_est_from_endpoints_hz"] == pytest.approx(30.0)
    assert out.sampling_rate_hz == pytest.approx(30.0)


def test_load_monitor_csv_raises_if_using_nonuniform_time_with_repeats(tmp_path: Path) -> None:
    p = tmp_path / "monitor.csv"
    p.write_text(
        "\n".join(
            [
                "10:00.0,1",
                "10:00.0,2",
                "10:00.1,3",
            ]
        )
        + "\n"
    )

    with pytest.raises(ValueError, match="not strictly increasing"):
        load_monitor_csv(p, sampling_rate_hz=500.0, assume_uniform_sampling=False)


