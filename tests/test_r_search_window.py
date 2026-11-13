"""R is searched near the detected QRS fiducial (human preset), not across the whole cycle."""

from dataclasses import replace

import numpy as np
import pandas as pd

from pyhearts._morphology.config import ProcessCycleConfig
from pyhearts._morphology.processing.processcycle import process_cycle
from tests.test_reseed_missing_components import _empty_output


def _cycle_tall_t(fs=500.0, n=400, r=200.0):
    """R at sample 200 (signal_x = 0), T wave taller than R."""
    xs = np.arange(n, dtype=float)
    y = 0.15 * np.exp(-((xs - (r - 80)) ** 2) / (2 * 10.0**2))
    y += 1.00 * np.exp(-((xs - r) ** 2) / (2 * 5.0**2))
    y -= 0.25 * np.exp(-((xs - (r + 18)) ** 2) / (2 * 6.0**2))
    y += 1.40 * np.exp(-((xs - (r + 120)) ** 2) / (2 * 20.0**2))
    return pd.DataFrame({"signal_x": (xs - r) / fs, "signal_y": y, "index": (xs + 1000).astype(int), "cycle": 0})


def _r_center(cfg):
    out, *_ = process_cycle(_cycle_tall_t(), _empty_output(), 500.0, 0, None, None,
                            previous_gauss_features=None, expected_max_energy=1.0, cfg=cfg)
    return out["R_center_idx"][0]


def test_human_preset_keeps_r_in_qrs_when_t_is_taller():
    assert abs(_r_center(ProcessCycleConfig.for_human()) - 200) <= 5


def test_legacy_whole_cycle_search_labels_tall_t_as_r():
    legacy = replace(ProcessCycleConfig.for_human(), r_search_half_window_ms=None)
    assert abs(_r_center(legacy) - 320) <= 10
