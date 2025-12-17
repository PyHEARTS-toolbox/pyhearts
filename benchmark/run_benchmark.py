#!/usr/bin/env python3
"""
Run PyHEARTS benchmark on test data.

Evaluates package performance across different noise levels and sampling rates.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional
import traceback

import numpy as np
import pandas as pd


@dataclass
class BenchmarkResult:
    """Result of benchmarking a single signal."""
    signal_id: str
    sampling_rate: int
    heart_rate: int
    noise_level: str
    
    # R-peak detection metrics
    n_true_peaks: int = 0
    n_detected_peaks: int = 0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    
    # Derived metrics
    sensitivity: float = 0.0  # TP / (TP + FN)
    precision: float = 0.0   # TP / (TP + FP) 
    f1_score: float = 0.0
    
    # Timing accuracy (for matched peaks)
    mean_timing_error_ms: float = 0.0
    std_timing_error_ms: float = 0.0
    max_timing_error_ms: float = 0.0
    
    # Processing time
    processing_time_sec: float = 0.0
    
    # Success/failure
    success: bool = True
    error_message: str = ""


def match_peaks(
    detected: np.ndarray,
    ground_truth: np.ndarray,
    tolerance_samples: int,
) -> tuple[int, int, int, np.ndarray]:
    """
    Match detected peaks to ground truth peaks.
    
    Args:
        detected: Detected peak indices
        ground_truth: Ground truth peak indices
        tolerance_samples: Maximum distance for a match
        
    Returns:
        true_positives, false_positives, false_negatives, timing_errors
    """
    if len(detected) == 0:
        return 0, 0, len(ground_truth), np.array([])
    
    if len(ground_truth) == 0:
        return 0, len(detected), 0, np.array([])
    
    # For each ground truth, find the closest detected peak
    matched_detected = set()
    matched_gt = set()
    timing_errors = []
    
    for i, gt_peak in enumerate(ground_truth):
        distances = np.abs(detected - gt_peak)
        closest_idx = np.argmin(distances)
        closest_dist = distances[closest_idx]
        
        if closest_dist <= tolerance_samples and closest_idx not in matched_detected:
            matched_detected.add(closest_idx)
            matched_gt.add(i)
            timing_errors.append(detected[closest_idx] - gt_peak)
    
    true_positives = len(matched_gt)
    false_negatives = len(ground_truth) - true_positives
    false_positives = len(detected) - len(matched_detected)
    
    return true_positives, false_positives, false_negatives, np.array(timing_errors)


def benchmark_signal(
    signal: np.ndarray,
    rpeaks_true: np.ndarray,
    sampling_rate: int,
    metadata: dict,
    tolerance_ms: float = 50.0,
) -> BenchmarkResult:
    """
    Benchmark PyHEARTS on a single signal.
    
    Args:
        signal: ECG signal array
        rpeaks_true: Ground truth R-peak indices
        sampling_rate: Sampling rate in Hz
        metadata: Signal metadata dict
        tolerance_ms: Tolerance for R-peak matching in ms
    """
    result = BenchmarkResult(
        signal_id=metadata['signal_id'],
        sampling_rate=sampling_rate,
        heart_rate=metadata['heart_rate'],
        noise_level=metadata['noise_level'],
        n_true_peaks=len(rpeaks_true),
    )
    
    tolerance_samples = int(tolerance_ms * sampling_rate / 1000)
    
    try:
        from pyhearts import PyHEARTS
        from pyhearts.config import ProcessCycleConfig
        
        # Time the processing
        start_time = time.perf_counter()
        
        # Run PyHEARTS
        cfg = ProcessCycleConfig.for_human()
        analyzer = PyHEARTS(sampling_rate, cfg=cfg, plot=False)
        analyzer.analyze_ecg(signal)
        
        end_time = time.perf_counter()
        result.processing_time_sec = end_time - start_time
        
        # Get detected R-peaks
        if hasattr(analyzer, 'r_peak_indices') and analyzer.r_peak_indices is not None:
            detected_peaks = np.array(analyzer.r_peak_indices)
        else:
            detected_peaks = np.array([])
        
        result.n_detected_peaks = len(detected_peaks)
        
        # Match peaks
        tp, fp, fn, timing_errors = match_peaks(
            detected_peaks, rpeaks_true, tolerance_samples
        )
        
        result.true_positives = tp
        result.false_positives = fp
        result.false_negatives = fn
        
        # Calculate metrics
        if tp + fn > 0:
            result.sensitivity = tp / (tp + fn)
        if tp + fp > 0:
            result.precision = tp / (tp + fp)
        if result.sensitivity + result.precision > 0:
            result.f1_score = 2 * (result.sensitivity * result.precision) / (result.sensitivity + result.precision)
        
        # Timing errors in ms
        if len(timing_errors) > 0:
            timing_errors_ms = timing_errors * 1000 / sampling_rate
            result.mean_timing_error_ms = float(np.mean(timing_errors_ms))
            result.std_timing_error_ms = float(np.std(timing_errors_ms))
            result.max_timing_error_ms = float(np.max(np.abs(timing_errors_ms)))
        
    except Exception as e:
        result.success = False
        result.error_message = f"{type(e).__name__}: {str(e)}"
        traceback.print_exc()
    
    return result


def run_benchmark(
    data_dir: Path,
    output_file: Optional[Path] = None,
    tolerance_ms: float = 50.0,
) -> pd.DataFrame:
    """
    Run benchmark on all signals in the test data directory.
    
    Args:
        data_dir: Directory containing test data
        output_file: Optional path to save results CSV
        tolerance_ms: Tolerance for R-peak matching
        
    Returns:
        DataFrame with benchmark results
    """
    data_dir = Path(data_dir)
    
    # Load metadata
    with open(data_dir / "metadata.json") as f:
        metadata_list = json.load(f)
    
    print(f"Running benchmark on {len(metadata_list)} signals...")
    print(f"R-peak matching tolerance: {tolerance_ms}ms")
    print()
    
    results = []
    
    for i, meta in enumerate(metadata_list, 1):
        signal_id = meta['signal_id']
        sampling_rate = meta['sampling_rate']
        
        print(f"[{i}/{len(metadata_list)}] {signal_id}...", end=" ", flush=True)
        
        # Load signal and ground truth
        signal = np.load(data_dir / "signals" / f"{signal_id}.npy")
        rpeaks_true = np.load(data_dir / "rpeaks_ground_truth" / f"{signal_id}_rpeaks.npy")
        
        # Run benchmark
        result = benchmark_signal(
            signal=signal,
            rpeaks_true=rpeaks_true,
            sampling_rate=sampling_rate,
            metadata=meta,
            tolerance_ms=tolerance_ms,
        )
        
        results.append(result)
        
        if result.success:
            print(f"F1={result.f1_score:.3f} ({result.processing_time_sec:.2f}s)")
        else:
            print(f"FAILED: {result.error_message[:50]}")
    
    # Convert to DataFrame
    df = pd.DataFrame([asdict(r) for r in results])
    
    # Print summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    
    # Summary by noise level
    print("\nBy Noise Level:")
    noise_summary = df.groupby('noise_level').agg({
        'f1_score': ['mean', 'std'],
        'sensitivity': 'mean',
        'precision': 'mean',
        'processing_time_sec': 'mean',
        'success': 'sum',
    }).round(3)
    print(noise_summary.to_string())
    
    # Summary by sampling rate
    print("\nBy Sampling Rate:")
    sr_summary = df.groupby('sampling_rate').agg({
        'f1_score': ['mean', 'std'],
        'sensitivity': 'mean',
        'precision': 'mean',
        'processing_time_sec': 'mean',
    }).round(3)
    print(sr_summary.to_string())
    
    # Overall stats
    print(f"\nOverall F1 Score: {df['f1_score'].mean():.3f} ± {df['f1_score'].std():.3f}")
    print(f"Success Rate: {df['success'].sum()}/{len(df)}")
    print(f"Total Processing Time: {df['processing_time_sec'].sum():.1f}s")
    
    # Save results
    if output_file:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=False)
        print(f"\n✅ Results saved to {output_file}")
    
    return df


def compare_benchmarks(
    baseline_file: Path,
    current_file: Path,
) -> pd.DataFrame:
    """
    Compare two benchmark results to see improvements/regressions.
    """
    baseline = pd.read_csv(baseline_file)
    current = pd.read_csv(current_file)
    
    # Merge on signal_id
    merged = baseline.merge(
        current,
        on='signal_id',
        suffixes=('_baseline', '_current')
    )
    
    # Calculate deltas
    merged['f1_delta'] = merged['f1_score_current'] - merged['f1_score_baseline']
    merged['sensitivity_delta'] = merged['sensitivity_current'] - merged['sensitivity_baseline']
    merged['precision_delta'] = merged['precision_current'] - merged['precision_baseline']
    merged['time_delta'] = merged['processing_time_sec_current'] - merged['processing_time_sec_baseline']
    
    print("\n" + "="*60)
    print("BENCHMARK COMPARISON")
    print("="*60)
    
    print(f"\nF1 Score Change: {merged['f1_delta'].mean():+.3f}")
    print(f"Sensitivity Change: {merged['sensitivity_delta'].mean():+.3f}")
    print(f"Precision Change: {merged['precision_delta'].mean():+.3f}")
    print(f"Processing Time Change: {merged['time_delta'].mean():+.3f}s")
    
    # Show biggest improvements and regressions
    print("\nBiggest Improvements (F1):")
    top_improvements = merged.nlargest(5, 'f1_delta')[['signal_id', 'f1_score_baseline', 'f1_score_current', 'f1_delta']]
    print(top_improvements.to_string(index=False))
    
    print("\nBiggest Regressions (F1):")
    top_regressions = merged.nsmallest(5, 'f1_delta')[['signal_id', 'f1_score_baseline', 'f1_score_current', 'f1_delta']]
    print(top_regressions.to_string(index=False))
    
    return merged


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run PyHEARTS benchmark")
    parser.add_argument(
        "--data-dir", "-d",
        type=str,
        default="benchmark/test_data",
        help="Directory containing test data"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="benchmark/results/benchmark_latest.csv",
        help="Output file for results"
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=50.0,
        help="R-peak matching tolerance in ms"
    )
    parser.add_argument(
        "--compare",
        type=str,
        default=None,
        help="Compare with a baseline results file"
    )
    
    args = parser.parse_args()
    
    results = run_benchmark(
        data_dir=Path(args.data_dir),
        output_file=Path(args.output),
        tolerance_ms=args.tolerance,
    )
    
    if args.compare:
        compare_benchmarks(
            baseline_file=Path(args.compare),
            current_file=Path(args.output),
        )

