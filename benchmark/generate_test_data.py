#!/usr/bin/env python3
"""
Generate benchmark ECG test data with varying noise levels and sampling rates.

This creates a comprehensive test dataset for validating and improving the PyHEARTS package.
"""
from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
import neurokit2 as nk


@dataclass
class SignalMetadata:
    """Metadata for a generated ECG signal."""
    signal_id: str
    sampling_rate: int
    duration_sec: float
    heart_rate: int
    noise_level: str  # 'clean', 'mild', 'moderate', 'severe', 'extreme'
    noise_std: float
    drift_amplitude: float
    line_noise_amplitude: float
    n_samples: int
    n_expected_beats: int
    random_seed: int


# Noise level configurations
NOISE_CONFIGS = {
    'clean': {
        'noise_std': 0.0,
        'drift_amplitude': 0.0,
        'line_noise_amplitude': 0.0,
    },
    'mild': {
        'noise_std': 0.02,
        'drift_amplitude': 0.1,
        'line_noise_amplitude': 0.02,
    },
    'moderate': {
        'noise_std': 0.05,
        'drift_amplitude': 0.3,
        'line_noise_amplitude': 0.05,
    },
    'severe': {
        'noise_std': 0.1,
        'drift_amplitude': 0.5,
        'line_noise_amplitude': 0.08,
    },
    'extreme': {
        'noise_std': 0.2,
        'drift_amplitude': 1.0,
        'line_noise_amplitude': 0.15,
    },
}

# Sampling rates to test
SAMPLING_RATES = [250, 500, 750, 1000]

# Heart rates to test (for variety)
HEART_RATES = [60, 75, 90]

# Duration in seconds
DURATION = 60


def generate_ecg_with_ground_truth(
    sampling_rate: int,
    heart_rate: int,
    noise_level: str,
    duration: float = 60.0,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, SignalMetadata]:
    """
    Generate an ECG signal with known noise parameters and ground truth R-peaks.
    
    Returns:
        signal: The noisy ECG signal
        rpeaks_true: Ground truth R-peak indices
        metadata: Signal metadata
    """
    np.random.seed(random_seed)
    
    noise_cfg = NOISE_CONFIGS[noise_level]
    
    # Generate clean ECG with neurokit2
    ecg_clean = nk.ecg_simulate(
        duration=duration,
        sampling_rate=sampling_rate,
        heart_rate=heart_rate,
        random_state=random_seed,
        noise=0,  # We'll add our own noise
    )
    
    # Detect R-peaks on the clean signal (ground truth)
    _, rpeaks_info = nk.ecg_peaks(ecg_clean, sampling_rate=sampling_rate)
    rpeaks_true = rpeaks_info['ECG_R_Peaks']
    
    n_samples = len(ecg_clean)
    time = np.arange(n_samples) / sampling_rate
    
    # Add controlled noise components
    signal = ecg_clean.copy()
    
    if noise_cfg['noise_std'] > 0:
        # Gaussian noise (scaled to signal std)
        noise = np.random.normal(0, noise_cfg['noise_std'] * np.std(ecg_clean), n_samples)
        signal = signal + noise
    
    if noise_cfg['drift_amplitude'] > 0:
        # Baseline wander (low frequency drift)
        drift = noise_cfg['drift_amplitude'] * np.sin(2 * np.pi * 0.1 * time)  # 0.1 Hz drift
        drift += noise_cfg['drift_amplitude'] * 0.5 * np.sin(2 * np.pi * 0.25 * time)
        signal = signal + drift
    
    if noise_cfg['line_noise_amplitude'] > 0:
        # Power line interference (50/60 Hz)
        line_noise = noise_cfg['line_noise_amplitude'] * np.sin(2 * np.pi * 60 * time)
        signal = signal + line_noise
    
    # Calculate expected beats
    n_expected_beats = int(heart_rate * duration / 60)
    
    metadata = SignalMetadata(
        signal_id=f"ecg_{sampling_rate}hz_{heart_rate}bpm_{noise_level}_seed{random_seed}",
        sampling_rate=sampling_rate,
        duration_sec=duration,
        heart_rate=heart_rate,
        noise_level=noise_level,
        noise_std=noise_cfg['noise_std'],
        drift_amplitude=noise_cfg['drift_amplitude'],
        line_noise_amplitude=noise_cfg['line_noise_amplitude'],
        n_samples=n_samples,
        n_expected_beats=n_expected_beats,
        random_seed=random_seed,
    )
    
    return signal.astype(np.float32), np.array(rpeaks_true), metadata


def generate_full_test_suite(output_dir: Path, seed_base: int = 42) -> List[SignalMetadata]:
    """
    Generate a complete suite of test signals.
    
    Creates signals for all combinations of:
    - Sampling rates: 250, 500, 750, 1000 Hz
    - Noise levels: clean, mild, moderate, severe, extreme
    - Heart rates: 60, 75, 90 BPM
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    signals_dir = output_dir / "signals"
    rpeaks_dir = output_dir / "rpeaks_ground_truth"
    signals_dir.mkdir(exist_ok=True)
    rpeaks_dir.mkdir(exist_ok=True)
    
    all_metadata = []
    total_signals = len(SAMPLING_RATES) * len(NOISE_CONFIGS) * len(HEART_RATES)
    
    print(f"Generating {total_signals} test signals...")
    print(f"  Sampling rates: {SAMPLING_RATES}")
    print(f"  Noise levels: {list(NOISE_CONFIGS.keys())}")
    print(f"  Heart rates: {HEART_RATES}")
    print()
    
    signal_idx = 0
    for sr in SAMPLING_RATES:
        for noise_level in NOISE_CONFIGS.keys():
            for hr in HEART_RATES:
                signal_idx += 1
                seed = seed_base + signal_idx
                
                print(f"  [{signal_idx}/{total_signals}] {sr}Hz, {hr}bpm, {noise_level}...", end=" ")
                
                signal, rpeaks, metadata = generate_ecg_with_ground_truth(
                    sampling_rate=sr,
                    heart_rate=hr,
                    noise_level=noise_level,
                    duration=DURATION,
                    random_seed=seed,
                )
                
                # Save signal
                signal_path = signals_dir / f"{metadata.signal_id}.npy"
                np.save(signal_path, signal)
                
                # Save ground truth R-peaks
                rpeaks_path = rpeaks_dir / f"{metadata.signal_id}_rpeaks.npy"
                np.save(rpeaks_path, rpeaks)
                
                all_metadata.append(metadata)
                print(f"OK ({len(rpeaks)} beats)")
    
    # Save metadata index
    metadata_list = [asdict(m) for m in all_metadata]
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata_list, f, indent=2)
    
    print(f"\n✅ Generated {len(all_metadata)} signals in {output_dir}")
    print(f"   Signals: {signals_dir}")
    print(f"   Ground truth R-peaks: {rpeaks_dir}")
    print(f"   Metadata: {output_dir / 'metadata.json'}")
    
    return all_metadata


def generate_quick_test_set(output_dir: Path, seed_base: int = 42) -> List[SignalMetadata]:
    """
    Generate a smaller test set for quick iteration.
    
    Uses only:
    - Sampling rates: 250, 1000 Hz (min and max)
    - Noise levels: clean, moderate, extreme
    - Heart rate: 75 BPM only
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    signals_dir = output_dir / "signals"
    rpeaks_dir = output_dir / "rpeaks_ground_truth"
    signals_dir.mkdir(exist_ok=True)
    rpeaks_dir.mkdir(exist_ok=True)
    
    quick_sampling_rates = [250, 1000]
    quick_noise_levels = ['clean', 'moderate', 'extreme']
    quick_heart_rates = [75]
    
    all_metadata = []
    total_signals = len(quick_sampling_rates) * len(quick_noise_levels) * len(quick_heart_rates)
    
    print(f"Generating {total_signals} quick test signals...")
    
    signal_idx = 0
    for sr in quick_sampling_rates:
        for noise_level in quick_noise_levels:
            for hr in quick_heart_rates:
                signal_idx += 1
                seed = seed_base + signal_idx
                
                print(f"  [{signal_idx}/{total_signals}] {sr}Hz, {hr}bpm, {noise_level}...", end=" ")
                
                signal, rpeaks, metadata = generate_ecg_with_ground_truth(
                    sampling_rate=sr,
                    heart_rate=hr,
                    noise_level=noise_level,
                    duration=DURATION,
                    random_seed=seed,
                )
                
                signal_path = signals_dir / f"{metadata.signal_id}.npy"
                np.save(signal_path, signal)
                
                rpeaks_path = rpeaks_dir / f"{metadata.signal_id}_rpeaks.npy"
                np.save(rpeaks_path, rpeaks)
                
                all_metadata.append(metadata)
                print(f"OK ({len(rpeaks)} beats)")
    
    metadata_list = [asdict(m) for m in all_metadata]
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata_list, f, indent=2)
    
    print(f"\n✅ Generated {len(all_metadata)} quick test signals in {output_dir}")
    
    return all_metadata


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate benchmark ECG test data")
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="benchmark/test_data",
        help="Output directory for test data"
    )
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Generate quick test set (fewer combinations)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    output_path = Path(args.output)
    
    if args.quick:
        generate_quick_test_set(output_path, args.seed)
    else:
        generate_full_test_suite(output_path, args.seed)

