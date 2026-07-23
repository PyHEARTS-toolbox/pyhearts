#!/usr/bin/env python3
"""Freeze analyzer configs for held-out validation (run before any LUDB scoring).

Writes immutable JSON lockfiles under ``validation/``. Re-running creates a new
``validation_id`` only if you change the output path; do not overwrite a locked
file after seeing held-out results.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pyhearts import PyHEARTS
from pyhearts._morphology.config import ProcessCycleConfig as MorphCfg
from pyhearts.core.analyzer import PIPELINE_VERSION
from pyhearts.version import __version__


def jsonable(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def git_meta() -> tuple[str, bool]:
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip() != ""
    return sha, dirty


def freeze_public_default() -> dict[str, Any]:
    analyzer = PyHEARTS(sampling_rate=500.0, species="human", verbose=False)
    sha, dirty = git_meta()
    return {
        "validation_id": "config_frozen_v1",
        "arm": "public_default",
        "description": (
            "Immutable public-default config under test for LUDB held-out validation. "
            "Constructed as PyHEARTS(sampling_rate=500, species='human'): morphology "
            "for_human() core + ProcessCycleConfig.for_human_unified() record-T."
        ),
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "package_version": __version__,
        "git_sha": sha,
        "git_dirty": dirty,
        "pipeline_version": PIPELINE_VERSION,
        "analyzer": {
            "species": "human",
            "sampling_rate_hz_reference": 500.0,
            "apply_record_t": True,
            "construction": "PyHEARTS(sampling_rate=..., species='human')",
        },
        "core_config": jsonable(analyzer._core_cfg),
        "t_config": jsonable(analyzer._t_cfg),
        "knobs_for_sensitivity_sweep": {
            "rpeak_prominence_multiplier": analyzer._core_cfg.rpeak_prominence_multiplier,
            "epoch_corr_thresh": analyzer._core_cfg.epoch_corr_thresh,
            "snr_mad_multiplier": dict(analyzer._core_cfg.snr_mad_multiplier),
            "bound_factor": analyzer._core_cfg.bound_factor,
        },
    }


def freeze_paper_era() -> dict[str, Any]:
    core = MorphCfg.for_human()
    sha, dirty = git_meta()
    return {
        "validation_id": "config_frozen_paper_era_v1",
        "arm": "paper_era_morphology",
        "description": (
            "Paper-era morphology human preset "
            "(_morphology.config.ProcessCycleConfig.for_human), record-T disabled. "
            "Secondary comparison arm only; primary held-out claim uses "
            "config_frozen_v1 (public_default)."
        ),
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "package_version": __version__,
        "git_sha": sha,
        "git_dirty": dirty,
        "pipeline_version": "morphology-only",
        "analyzer": {
            "species": "human",
            "sampling_rate_hz_reference": 500.0,
            "apply_record_t": False,
            "construction": "morphology ProcessCycleConfig.for_human(); apply_record_t=False",
        },
        "core_config": jsonable(core),
        "t_config": None,
        "knobs_for_sensitivity_sweep": {
            "rpeak_prominence_multiplier": core.rpeak_prominence_multiplier,
            "epoch_corr_thresh": core.epoch_corr_thresh,
            "snr_mad_multiplier": dict(core.snr_mad_multiplier),
            "bound_factor": core.bound_factor,
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("validation"),
        help="Directory for lockfiles (default: validation/)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing lockfiles (forbidden after held-out results).",
    )
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        ("config_frozen_v1.json", freeze_public_default),
        ("config_frozen_paper_era_v1.json", freeze_paper_era),
    ]
    for name, builder in specs:
        path = args.out_dir / name
        if path.exists() and not args.force:
            raise SystemExit(
                f"{path} already exists; refusing to overwrite. "
                "Pass --force only before any held-out run, or write a new validation_id."
            )
        payload = builder()
        path.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {path} arm={payload['arm']} knobs={payload['knobs_for_sensitivity_sweep']}")


if __name__ == "__main__":
    main()
