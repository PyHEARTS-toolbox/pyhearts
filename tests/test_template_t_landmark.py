"""STPQ template Tⱼ landmark: early dominant peak vs isoelectric fallback."""

import numpy as np
from dataclasses import replace

from pyhearts.config import ProcessCycleConfig
from pyhearts.processing.delineation_signal import savgol_search_segment
from pyhearts.processing.record_delineation import (
    _apply_morphology_peak_t_landmark_override,
    _compute_template_thresholds,
    _fixed_window_morphology_peak_frac,
    _landmarks_closest_to_baseline,
    _landmarks_t_isoelectric_fallback,
    _MORPH_LANDMARK_OVERRIDE_MIN_DELTA_FRAC,
    _t_region_amplitude_reference,
    build_stpq_beat_template,
    delineate_stpq_template,
)


def _early_t_then_flat_st(n: int = 160) -> np.ndarray:
    """Early T apex ~30% S→Q, isoelectric ST ~50% (sel114-like)."""
    x = np.linspace(0, 1, n)
    tmpl = 0.02 * np.sin(2 * np.pi * x)
    t_idx = int(round(0.30 * (n - 1)))
    tmpl[t_idx] += 0.35
    st_idx = int(round(0.50 * (n - 1)))
    tmpl[st_idx] = float(np.median(tmpl))
    return tmpl.astype(float)


def _late_dominant_t(n: int = 160) -> np.ndarray:
    """Small early bump ~30%, main peak ~46% (sel104-like)."""
    x = np.linspace(0, 1, n)
    tmpl = 0.01 * x
    early = int(round(0.30 * (n - 1)))
    tmpl[early] += 0.12
    main = int(round(0.46 * (n - 1)))
    tmpl[main] += 0.35
    return tmpl.astype(float)


def _s_dominated_early_s(n: int = 160) -> np.ndarray:
    """Large negative S at ~8%, small T bump at ~35% (sel16420-like normalization bug)."""
    tmpl = np.zeros(n, dtype=float)
    s_idx = int(round(0.08 * (n - 1)))
    tmpl[s_idx] = -0.80
    t_idx = int(round(0.35 * (n - 1)))
    tmpl[t_idx] += 0.18
    return tmpl


def _flat_top_t_low_prom(n: int = 160) -> np.ndarray:
    """Flat-topped T: amp passes, prom fails, right limb descends within +20 ms."""
    tmpl = np.full(n, 0.02, dtype=float)
    peak = int(round(0.30 * (n - 1)))
    for i in range(n):
        f = i / (n - 1)
        if f < 0.28:
            tmpl[i] = 0.02 + 0.12 * f / 0.28
        elif f <= 0.30:
            tmpl[i] = 0.14
        else:
            tmpl[i] = 0.14 - 0.10 * min(1.0, (f - 0.30) / 0.08)
    tmpl[peak] = 0.14
    return tmpl


def _true_flat_plateau(n: int = 112) -> np.ndarray:
    """Genuinely flat after peak — sele0203-like (prom=0, no descent)."""
    tmpl = np.full(n, 0.02, dtype=float)
    s_idx = int(round(0.08 * (n - 1)))
    tmpl[s_idx] = -0.25
    for i in range(n):
        f = i / (n - 1)
        if f < 0.15:
            tmpl[i] = max(tmpl[i], 0.02)
        elif f < 0.20:
            tmpl[i] = 0.02 + 0.08 * (f - 0.15) / 0.05
        elif f <= 0.40:
            tmpl[i] = 0.10
        else:
            tmpl[i] = 0.10
    return tmpl


def test_t_amplitude_norm_uses_t_region_not_global_s():
    cfg = replace(
        ProcessCycleConfig(),
        p_t_search_savgol=False,
        record_template_t_landmark_min_peak_frac=0.20,
        record_template_t_amplitude_norm_sq_frac=(0.15, 0.80),
    )
    fs = 250.0
    tmpl = _s_dominated_early_s()
    baseline = float(np.median(tmpl))
    global_peak = float(np.max(np.abs(tmpl - baseline)))
    t_ref = _t_region_amplitude_reference(tmpl, baseline, cfg)
    assert t_ref < global_peak * 0.5
    t_j, _, src = _landmarks_closest_to_baseline(tmpl, cfg, fs)
    frac = t_j / (tmpl.size - 1)
    assert 0.28 <= frac <= 0.42


def test_early_peak_landmark_sel114_morphology():
    cfg = replace(ProcessCycleConfig(), p_t_search_savgol=False)
    fs = 250.0
    tmpl = _early_t_then_flat_st()
    t_j, _, src = _landmarks_closest_to_baseline(tmpl, cfg, fs)
    frac = t_j / (tmpl.size - 1)
    assert 0.22 <= frac <= 0.38


def test_late_peak_landmark_prefers_main_lobe_in_window():
    cfg = replace(ProcessCycleConfig(), p_t_search_savgol=False)
    fs = 250.0
    tmpl = _late_dominant_t()
    t_j, _, src = _landmarks_closest_to_baseline(tmpl, cfg, fs)
    frac = t_j / (tmpl.size - 1)
    # Main peak at 46% is outside 15–40%; early bump at ~30% wins
    assert 0.28 <= frac <= 0.35


def test_flat_morphology_falls_back_to_isoelectric():
    cfg = ProcessCycleConfig(record_template_t_landmark_min_peak_frac=0.20)
    fs = 250.0
    n = 120
    tmpl = np.zeros(n, dtype=float)
    tmpl[int(round(0.50 * (n - 1)))] = 0.0
    t_j, _, src = _landmarks_closest_to_baseline(tmpl, cfg, fs)
    assert t_j == _landmarks_t_isoelectric_fallback(tmpl)


def _flat_st_plateau_sel102_like(n: int = 154) -> np.ndarray:
    """ST upslope then flat plateau ~35–40%; manual T near isoelectric ~32% (sel102-like)."""
    tmpl = np.zeros(n, dtype=float)
    for i in range(n):
        f = i / (n - 1)
        if f < 0.15:
            tmpl[i] = -0.05 * f / 0.15
        elif f < 0.35:
            tmpl[i] = -0.05 + 0.14 * (f - 0.15) / 0.20
        elif f <= 0.40:
            tmpl[i] = 0.09
        else:
            tmpl[i] = 0.09
    idx_gold = int(round(0.32 * (n - 1)))
    tmpl[idx_gold] = 0.003
    return tmpl


def _inverted_t_sel100_like(n: int = 190) -> np.ndarray:
    """Inverted T trough ~32%; small positive ST bump ~40% (sel100-like)."""
    tmpl = np.zeros(n, dtype=float)
    t_idx = int(round(0.32 * (n - 1)))
    tmpl[t_idx] = -0.11
    st_idx = int(round(0.40 * (n - 1)))
    tmpl[st_idx] = 0.02
    tmpl += 0.02 * np.linspace(0, 1, n)
    return tmpl


def test_inverted_t_landmark_prefers_negative_trough():
    cfg = replace(
        ProcessCycleConfig(),
        p_t_search_savgol=False,
        record_template_t_landmark_min_peak_frac=0.20,
        record_template_t_landmark_min_prominence_frac=0.10,
        record_template_t_amplitude_norm_sq_frac=(0.15, 0.80),
    )
    fs = 250.0
    tmpl = _inverted_t_sel100_like()
    t_j, _, src = _landmarks_closest_to_baseline(tmpl, cfg, fs)
    frac = t_j / (tmpl.size - 1)
    assert 0.28 <= frac <= 0.36


def test_flat_st_plateau_uses_rising_edge_or_plateau():
    """Flat ST hump fails prominence → rising-edge onset or plateau apex, not late isoelectric."""
    cfg = replace(
        ProcessCycleConfig(),
        p_t_search_savgol=False,
        record_template_t_landmark_min_peak_frac=0.20,
        record_template_t_landmark_min_prominence_frac=0.10,
    )
    fs = 250.0
    tmpl = _flat_st_plateau_sel102_like()
    t_j, _, src = _landmarks_closest_to_baseline(tmpl, cfg, fs)
    frac = t_j / (tmpl.size - 1)
    assert src in ("rising_edge", "plateau_apex", "early_peak")
    assert 0.075 <= frac <= 0.38


def test_plateau_exception_accepts_low_prom_with_descending_limb():
    cfg = replace(
        ProcessCycleConfig(),
        p_t_search_savgol=False,
        record_template_t_landmark_min_peak_frac=0.20,
        record_template_t_landmark_min_prominence_frac=0.10,
    )
    fs = 250.0
    tmpl = _flat_top_t_low_prom()
    t_j, _, src = _landmarks_closest_to_baseline(tmpl, cfg, fs)
    frac = t_j / (tmpl.size - 1)
    assert src in ("plateau_apex", "early_peak")
    assert 0.26 <= frac <= 0.34


def test_rising_edge_when_flat_plateau_no_descent():
    cfg = replace(
        ProcessCycleConfig(),
        p_t_search_savgol=False,
        record_template_t_landmark_min_peak_frac=0.20,
        record_template_t_landmark_min_prominence_frac=0.10,
    )
    fs = 250.0
    tmpl = _true_flat_plateau()
    t_j, _, src = _landmarks_closest_to_baseline(tmpl, cfg, fs)
    assert src == "rising_edge"
    frac = t_j / (tmpl.size - 1)
    assert 0.08 <= frac <= 0.12


def test_rising_edge_uses_8pct_window_not_shared_15pct_lo():
    """Rising-edge onset scan starts at 8% while peak gate window stays at 15%."""
    cfg = replace(
        ProcessCycleConfig(),
        p_t_search_savgol=False,
        record_template_t_landmark_sq_frac=(0.15, 0.40),
        record_template_t_rising_edge_lo_frac=0.08,
        record_template_t_landmark_min_peak_frac=0.20,
        record_template_t_landmark_min_prominence_frac=0.10,
    )
    fs = 250.0
    tmpl = _true_flat_plateau()
    t_j, _, src = _landmarks_closest_to_baseline(tmpl, cfg, fs)
    assert src == "rising_edge"
    frac = t_j / (tmpl.size - 1)
    assert 0.08 <= frac <= 0.12


def test_config_window_is_respected():
    cfg = replace(
        ProcessCycleConfig(),
        record_template_t_landmark_sq_frac=(0.20, 0.35),
    )
    fs = 250.0
    tmpl = _early_t_then_flat_st(n=200)
    t_j, _, src = _landmarks_closest_to_baseline(tmpl, cfg, fs)
    frac = t_j / (tmpl.size - 1)
    assert 0.18 <= frac <= 0.37


def test_morphology_fixed_window_sel104_like_becomes_normal():
    cfg = replace(
        ProcessCycleConfig(),
        record_template_t_morphology_sq_frac=(0.20, 0.60),
    )
    tmpl = _late_dominant_t(n=160)
    t_j, p_j, _ = _landmarks_closest_to_baseline(tmpl, cfg, 250.0)
    _, _, _, _, morph = _compute_template_thresholds(tmpl, t_j, p_j, 1.0, cfg)
    assert morph == "normal"


def test_morphology_fixed_window_sel100_like_stays_inverted():
    cfg = replace(
        ProcessCycleConfig(),
        record_template_t_morphology_sq_frac=(0.20, 0.60),
    )
    tmpl = _inverted_t_sel100_like(n=190)
    t_j, p_j, _ = _landmarks_closest_to_baseline(tmpl, cfg, 250.0)
    _, _, _, _, morph = _compute_template_thresholds(tmpl, t_j, p_j, 1.0, cfg)
    assert morph == "inverted_t"


def _rising_edge_then_late_positive_t(n: int = 160) -> np.ndarray:
    """Rising-edge onset ~8%, dominant positive T ~46% (sel104/sel221-like)."""
    tmpl = np.zeros(n, dtype=float)
    rise = int(round(0.08 * (n - 1)))
    tmpl[rise] = -0.12
    for i in range(rise + 1, n):
        f = i / (n - 1)
        if f < 0.46:
            tmpl[i] = -0.12 + 0.55 * (f - 0.08) / 0.38
        else:
            tmpl[i] = 0.43 - 0.08 * (f - 0.46) / 0.54
    main = int(round(0.46 * (n - 1)))
    tmpl[main] = 0.43
    return tmpl


def test_fixed_window_morphology_peak_frac_in_20_60_window():
    cfg = replace(
        ProcessCycleConfig(),
        record_template_t_morphology_sq_frac=(0.20, 0.60),
    )
    tmpl = _rising_edge_then_late_positive_t()
    frac = _fixed_window_morphology_peak_frac(tmpl, cfg)
    assert frac is not None
    assert 0.38 <= frac <= 0.50


def test_morphology_peak_override_reroutes_rising_edge_to_late_t():
    cfg = replace(
        ProcessCycleConfig(),
        record_template_t_morphology_sq_frac=(0.20, 0.60),
    )
    tmpl = _rising_edge_then_late_positive_t()
    t_j = int(round(0.08 * (tmpl.size - 1)))
    t_new, t_off = _apply_morphology_peak_t_landmark_override(tmpl, t_j, cfg)
    assert t_off is not None
    assert t_new == int(t_off)
    assert t_new / (tmpl.size - 1) >= 0.38
    assert abs(t_new / (tmpl.size - 1) - _fixed_window_morphology_peak_frac(tmpl, cfg)) < 0.02


def test_morphology_peak_override_skips_when_delta_below_threshold():
    cfg = replace(
        ProcessCycleConfig(),
        record_template_t_morphology_sq_frac=(0.20, 0.60),
    )
    tmpl = _inverted_t_sel100_like(n=190)
    t_j, _, _ = _landmarks_closest_to_baseline(tmpl, cfg, 250.0)
    fixed_frac = _fixed_window_morphology_peak_frac(tmpl, cfg)
    assert fixed_frac is not None
    assert abs(fixed_frac - t_j / (tmpl.size - 1)) < _MORPH_LANDMARK_OVERRIDE_MIN_DELTA_FRAC
    t_new, t_off = _apply_morphology_peak_t_landmark_override(tmpl, t_j, cfg)
    assert t_new == t_j
    assert t_off is None


def test_delineate_stpq_template_negative_early_peak_normal_sets_t_offset():
    """sel301-class: beat-level early_peak narrowing must not break template T offset."""
    from pyhearts.processing.record_delineation import MedianBeatTemplate

    cfg = replace(ProcessCycleConfig.for_human_unified(), p_t_search_savgol=False)
    fs = 250.0
    n = 190
    sig = _inverted_t_sel100_like(n)
    t_j = int(round(0.32 * (n - 1)))
    p_j = int(round(0.75 * (n - 1)))
    th_t_up, th_t_down, th_p_up, th_p_down, morph = _compute_template_thresholds(
        sig, t_j, p_j, 1.0, cfg
    )
    raw = MedianBeatTemplate(
        template=sig,
        pre_r_samples=20,
        r_center_idx=0,
        p_offset_samples=None,
        t_offset_samples=None,
        p_polarity="positive",
        t_polarity="negative",
        median_rr_samples=200.0,
        n_beats=10,
        valid=True,
        template_anchor="s_to_q",
        t_landmark_idx=t_j,
        p_landmark_idx=p_j,
        th_t_up=th_t_up,
        th_t_down=th_t_down,
        th_p_up=th_p_up,
        th_p_down=th_p_down,
        t_morphology="normal",
        t_landmark_source="early_peak",
    )
    out = delineate_stpq_template(raw, fs, cfg)
    assert out.t_offset_samples is not None
    assert out.t_morphology == "normal"
    assert out.t_landmark_source == "early_peak"


def test_delineate_stpq_template_q2c_preserves_negative_polarity():
    from pyhearts.processing.record_delineation import MedianBeatTemplate

    cfg = replace(ProcessCycleConfig.for_human_unified(), p_t_search_savgol=False)
    fs = 250.0
    n = 190
    sig = _inverted_t_sel100_like(n)
    t_j = int(round(0.32 * (n - 1)))
    p_j = int(round(0.75 * (n - 1)))
    th_t_up, th_t_down, th_p_up, th_p_down, morph = _compute_template_thresholds(
        sig, t_j, p_j, 1.0, cfg
    )
    raw = MedianBeatTemplate(
        template=sig,
        pre_r_samples=20,
        r_center_idx=0,
        p_offset_samples=None,
        t_offset_samples=None,
        p_polarity="positive",
        t_polarity="negative",
        median_rr_samples=200.0,
        n_beats=10,
        valid=True,
        template_anchor="s_to_q",
        t_landmark_idx=t_j,
        p_landmark_idx=p_j,
        th_t_up=th_t_up,
        th_t_down=th_t_down,
        th_p_up=th_p_up,
        th_p_down=th_p_down,
        t_morphology="normal",
        t_landmark_source="early_peak",
    )
    out = delineate_stpq_template(raw, fs, cfg, manual_ann_ext="q2c")
    assert out.t_polarity == "negative"
