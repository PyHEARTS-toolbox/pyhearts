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
