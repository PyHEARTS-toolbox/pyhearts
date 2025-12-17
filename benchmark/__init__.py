"""Benchmark utilities for PyHEARTS."""
from .generate_test_data import (
    generate_ecg_with_ground_truth,
    generate_full_test_suite,
    generate_quick_test_set,
    NOISE_CONFIGS,
    SAMPLING_RATES,
    HEART_RATES,
)
from .run_benchmark import (
    run_benchmark,
    benchmark_signal,
    compare_benchmarks,
    BenchmarkResult,
)

__all__ = [
    "generate_ecg_with_ground_truth",
    "generate_full_test_suite", 
    "generate_quick_test_set",
    "run_benchmark",
    "benchmark_signal",
    "compare_benchmarks",
    "BenchmarkResult",
    "NOISE_CONFIGS",
    "SAMPLING_RATES",
    "HEART_RATES",
]

