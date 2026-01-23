# PyHEARTS Flow Chart (Beginner-Friendly)

PyHEARTS turns **raw ECG** into an **interpretable, beat-by-beat feature table** (plus optional recording-level summaries like HRV). The core idea is: detect beats → model each beat’s P/Q/R/S/T waves with physiologically constrained Gaussian fits → compute features.

## High-level pipeline

```mermaid
flowchart TD
  A[Raw ECG signal<br/>(samples in mV)] --> B[Preprocess<br/>bandpass / notch / detrend]
  B --> C{Signal quality check}
  C -->|warn/flag| C1[Low quality noted<br/>SNR / baseline wander / amplitude]
  C -->|continue| D[R-peak detection<br/>(find each heartbeat)]
  D --> E[Epoch into cycles<br/>(window around each R)]

  E --> F{{Per-cycle processing<br/>(repeat for each beat)}}
  F --> G[Detect fiducials<br/>P, Q, R, S, T + onsets/offsets]
  G --> H[Fit waves with constraints<br/>Gaussian / skewed Gaussian]
  H --> I[Extract per-beat features<br/>morphology + intervals + ST + fit quality]
  I --> J[Aggregate into feature table<br/>rows = beats, columns = features]

  J --> K[Beat-to-beat variability summary<br/>(recording-level, automatic)]
  J --> L[HRV metrics<br/>(recording-level, optional)]
  J --> M[Outputs<br/>CSVs + metadata JSON + optional plots]
```

## What you get (outputs)

```mermaid
flowchart LR
  A[ECG recording] --> B[Per-beat feature table<br/>(one row per beat)]
  B --> C[Modeling / statistics<br/>interpretable ML-ready features]
  A --> D[Recording-level summaries]
  D --> E[Beat-to-beat variability<br/>(e.g., QT variability)]
  D --> F[HRV metrics<br/>(e.g., SDNN, RMSSD)]
```

## “Mental model” for beginners

- **PyHEARTS is not a black box classifier**: it produces *measured* ECG descriptors you can inspect (P/QRS/T timing, amplitudes, widths, slopes, QT/PR/QRS intervals, ST measures, fit quality).
- **It’s beat-centric**: most outputs are computed per cardiac cycle, then optionally summarized across the recording.


