"""P/T detection mode helpers (derivative per-cycle vs record-only record-T path)."""

from __future__ import annotations

from pyhearts.config import ProcessCycleConfig


def p_t_detection_is_record_only(cfg: ProcessCycleConfig) -> bool:
    """True when per-cycle P/T derivative detection is disabled; record record-T pass only."""
    return cfg.p_t_detection_method == "record_only"


def p_t_detection_uses_derivative_per_cycle(cfg: ProcessCycleConfig) -> bool:
    return not p_t_detection_is_record_only(cfg)
