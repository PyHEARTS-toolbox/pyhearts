"""Tests for per-beat P/T candidate visibility diagnostics."""

from __future__ import annotations

import numpy as np

from pyhearts.processing.candidate_visibility import diagnose_peak_candidates
from pyhearts.processing.t_candidate_scoring import TCandidate


def _cand(sample_idx: int, source: str = "positive_peak") -> TCandidate:
    return TCandidate(
        sample_idx=sample_idx,
        rt_ms=100.0,
        source=source,
        signed_amp=1.0,
        prominence=1.0,
        width_ms=30.0,
        curvature=0.1,
        is_terminal=False,
        before_first_pos=False,
        sign_matches_template=True,
    )


def test_manual_inside_window_and_like_candidate_selected():
    fs = 250.0
    manual = 1000.0
    selected = 1002.0
    cands = [_cand(1001), _cand(1100, "negative_peak")]
    diag = diagnose_peak_candidates(
        wave="T",
        manual_sample=manual,
        selected_sample=selected,
        window_lo=900,
        window_hi=1200,
        candidates=cands,
        fs=fs,
        manual_like_tolerance_ms=15.0,
    )
    assert diag.manual_inside_window
    assert diag.manual_like_candidate
    assert not diag.manual_like_not_selected
    assert abs(diag.selected_delta_ms - 8.0) < 0.1


def test_manual_like_not_selected():
    fs = 250.0
    manual = 1000.0
    selected = 1080.0
    cands = [_cand(1001), _cand(1080)]
    diag = diagnose_peak_candidates(
        wave="T",
        manual_sample=manual,
        selected_sample=selected,
        window_lo=900,
        window_hi=1200,
        candidates=cands,
        fs=fs,
        manual_like_tolerance_ms=15.0,
    )
    assert diag.manual_like_candidate
    assert diag.manual_like_not_selected
    assert np.isfinite(diag.selected_delta_ms)


def test_manual_outside_window():
    fs = 250.0
    diag = diagnose_peak_candidates(
        wave="P",
        manual_sample=800.0,
        selected_sample=950.0,
        window_lo=900,
        window_hi=1200,
        candidates=[_cand(950)],
        fs=fs,
    )
    assert not diag.manual_inside_window
