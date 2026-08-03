# Held-out validation protocol (minimum viable)

This directory locks configs **before** scoring and runs the only fully
non-circular morphology evaluation: **LUDB**.

## Roles of datasets

| Dataset | Role |
|---------|------|
| AA, PTB-XL | Tuning (paper Methods) |
| QTDB | Development / benchmark (incl. Dec 2024 re-tune) — **not** held-out |
| SPH, MGH, mouse | Paper-era held-outs for paper-era parameters only (no recorded tuning objective or record list) |
| **LUDB** | **Held-out** for frozen public-default config |

Parameter selection predates a formal protocol; exact tuning records are
unavailable. The frozen-config LUDB run is therefore the only fully
non-circular evaluation.

## Floor (four pieces)

1. **Freeze** `config_frozen_v1.json` (public default) and commit it before any
   LUDB scoring. Optional second arm: `config_frozen_paper_era_v1.json`.
2. **Primary run** on LUDB only, metrics locked a priori:
   unconditional Se @ ±40 / ±150 ms, conditional bias + scatter, Se(tolerance)
   curve.
3. **Sensitivity sweep** on LUDB *after* primary numbers are locked
   (prominence, epoch_corr, SNR MAD, bound_factor × {0.5, 0.75, 1.0, 1.25, 1.5}).
4. **Honest relabel** of QTDB / AA / PTB-XL / SPH in README and Methods.

## Detector-level R-peak Se / PPV (window-bounded)

LUDB R marks are mid-strip only (~7.2 s of each 10 s; ~2.07 unmarked edge
beats/record). Full-strip PPV is invalid. Before any detector scoring, the
window rule is locked in:

- `validation/rpeak_detection_window_spec_v1.md` (human)
- `validation/rpeak_detection_window_spec_v1.json` (machine)

Summary: \(W=[t_\mathrm{first\_ann}, t_\mathrm{last\_ann}]\); match all
detections to all annotations within \(T\); then FP = unmatched detections
inside \(W_{+T}\); unmatched detections outside \(W_{+T}\) are
**indeterminate** (excluded from PPV). Expansion safety @ ±150 ms confirmed
on 200/200 records (worst clearance 210 ms, record 38).

Primary run also emits detector tables (same frozen config, no morphology
metric changes):

- `ludb_rpeak_detection.csv` — one row per record × tolerance (±40 / ±150)
- `ludb_rpeak_detection_summary.csv` — pooled totals; `n_in_window` /
  `n_indeterminate` / PPV denominators are tolerance-dependent

Soft indeterminate scale ~414 (somewhat below is benign; upper tail flags
margin over-detection).

## Commands

```bash
# Regenerate lockfiles only if no held-out results exist yet:
PYTHONPATH=. python validation/freeze_config.py

# Primary held-out (after freeze is committed):
PYTHONPATH=. python validation/run_ludb_heldout.py \
  --frozen validation/config_frozen_v1.json \
  --data-dir data/ludb \
  --out-dir validation/output/ludb_heldout_v1

# Sensitivity (only after primary output exists):
PYTHONPATH=. python validation/run_ludb_heldout.py \
  --frozen validation/config_frozen_v1.json \
  --data-dir data/ludb \
  --out-dir validation/output/ludb_sensitivity_v1 \
  --sensitivity
```

Download LUDB (PhysioNet) if needed:

```bash
python -c "import wfdb; wfdb.dl_database('ludb', dl_dir='data/ludb')"
```

## Archived QTDB analysis runs

QTDB lead-II fiducial runs from the July 2026 validation pass are parked under
`validation/parked/fiducial_lead_ii_qtdb_2026-07-22/` (outputs + scripts).
They are **not** part of the active held-out protocol; proceed with LUDB only
until QTDB is intentionally re-enabled.
