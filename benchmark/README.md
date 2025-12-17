# PyHEARTS Benchmark Suite

Tools for generating test ECG data and benchmarking package performance.

## Quick Start

### 1. Generate Test Data

```bash
# Generate full test suite (60 signals)
# 4 sampling rates × 5 noise levels × 3 heart rates
python -m benchmark.generate_test_data

# Or generate quick test set (6 signals) for rapid iteration
python -m benchmark.generate_test_data --quick
```

### 2. Run Benchmark

```bash
# Run benchmark on test data
python -m benchmark.run_benchmark

# Compare with a baseline
python -m benchmark.run_benchmark --compare benchmark/results/baseline.csv
```

## Test Data Specifications

### Sampling Rates
- 250 Hz (low-end clinical devices)
- 500 Hz (standard)
- 750 Hz (high quality)
- 1000 Hz (research grade)

### Noise Levels

| Level    | Gaussian Noise | Drift | Line Noise | Use Case |
|----------|----------------|-------|------------|----------|
| clean    | 0%             | 0     | 0          | Ideal conditions |
| mild     | 2%             | 0.1   | 0.02       | Good quality |
| moderate | 5%             | 0.3   | 0.05       | Typical clinical |
| severe   | 10%            | 0.5   | 0.08       | Challenging |
| extreme  | 20%            | 1.0   | 0.15       | Stress test |

### Heart Rates
- 60 BPM (resting)
- 75 BPM (normal)
- 90 BPM (elevated)

## Metrics

The benchmark evaluates:

- **Sensitivity (Recall)**: TP / (TP + FN) - How many real R-peaks were detected
- **Precision**: TP / (TP + FP) - How many detected peaks were real
- **F1 Score**: Harmonic mean of sensitivity and precision
- **Timing Error**: Difference between detected and true R-peak locations (ms)
- **Processing Time**: Wall-clock time to analyze each signal

## Directory Structure

```
benchmark/
├── test_data/
│   ├── signals/              # .npy files with ECG signals
│   ├── rpeaks_ground_truth/  # .npy files with true R-peak locations
│   └── metadata.json         # Signal parameters
├── results/
│   ├── benchmark_latest.csv  # Most recent benchmark
│   └── baseline.csv          # Baseline for comparison
├── generate_test_data.py
├── run_benchmark.py
└── README.md
```

## Workflow for Package Improvements

1. **Establish baseline:**
   ```bash
   python -m benchmark.generate_test_data
   python -m benchmark.run_benchmark -o benchmark/results/baseline.csv
   ```

2. **Make changes to PyHEARTS**

3. **Run comparison benchmark:**
   ```bash
   python -m benchmark.run_benchmark --compare benchmark/results/baseline.csv
   ```

4. **If improvements confirmed, save new baseline:**
   ```bash
   cp benchmark/results/benchmark_latest.csv benchmark/results/baseline.csv
   ```

## Python API

```python
from benchmark import (
    generate_quick_test_set,
    run_benchmark,
    compare_benchmarks,
)
from pathlib import Path

# Generate data
generate_quick_test_set(Path("benchmark/test_data"))

# Run benchmark
results_df = run_benchmark(
    data_dir=Path("benchmark/test_data"),
    output_file=Path("benchmark/results/current.csv"),
)

# Analyze specific conditions
severe_results = results_df[results_df['noise_level'] == 'severe']
print(f"F1 on severe noise: {severe_results['f1_score'].mean():.3f}")
```

