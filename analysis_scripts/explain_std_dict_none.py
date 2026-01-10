#!/usr/bin/env python3
"""
Explain why std_dict.get("R") can return None even when R is detected.
"""

import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.absolute()))
from pyhearts.processing.gaussian import compute_gauss_std

print("="*80)
print("Why std_dict.get('R') can return None even when R is detected")
print("="*80)

print("\nThe key insight:")
print("- R peak DETECTION (finding center_idx and height) is separate from")
print("  R std ESTIMATION (computing Gaussian width from half-height)")
print("- Detection can succeed even when std estimation fails")

print("\n" + "="*80)
print("Scenario where std estimation fails:")
print("="*80)

# Real-world scenario: R peak detected, but signal shape prevents half-height search
signal = np.zeros(100)

# R peak is detected at index 50 with height 1.0
# But the actual signal has interference/noise that prevents clean half-height detection
signal[45:55] = [0.3, 0.4, 0.5, 0.6, 0.7, 1.0, 0.7, 0.6, 0.5, 0.4]  # R peak
# But there's interference that keeps signal above threshold
signal[40:50] = 0.6  # Interference keeps signal above 0.5 threshold
signal[50:60] = 0.6  # Interference keeps signal above 0.5 threshold

guess_idxs = {"R": (50, 1.0)}
std_dict = compute_gauss_std(signal, guess_idxs)

print(f"\nSignal around R peak:")
print(f"  signal[45:55] = {signal[45:55]}")
print(f"  R detected at index 50, height = 1.0")
print(f"  Threshold for half-height = 0.5 * 1.0 = 0.5")
print(f"\nHalf-height search:")
print(f"  Left search: starts at 50, looks for signal < 0.5")
print(f"    signal[50] = {signal[50]:.1f} > 0.5, continue left")
print(f"    signal[49] = {signal[49]:.1f} > 0.5, continue left")
print(f"    ... signal[40] = {signal[40]:.1f} > 0.5, hits boundary at 0")
print(f"  Right search: starts at 50, looks for signal < 0.5")
print(f"    signal[50] = {signal[50]:.1f} > 0.5, continue right")
print(f"    ... signal[99] = {signal[99]:.1f} > 0.5, hits boundary at 99")
print(f"\nResult: width = 99 - 0 = 99, std = 99 / 2.3548 = {99/2.3548:.2f}")
print(f"  std_dict = {std_dict}")

# But what if the search fails differently?
print("\n" + "="*80)
print("Another scenario: Signal shape issue")
print("="*80)

signal2 = np.zeros(100)
# R peak detected, but signal has unusual shape
signal2[50] = 1.0  # Single point peak
# But surrounding signal is exactly at threshold
signal2[49] = 0.5
signal2[51] = 0.5
# Rest is below threshold
signal2[48] = 0.4
signal2[52] = 0.4

guess_idxs2 = {"R": (50, 1.0)}
std_dict2 = compute_gauss_std(signal2, guess_idxs2)

print(f"\nSignal:")
print(f"  signal2[48:53] = {signal2[48:53]}")
print(f"  R detected at index 50, height = 1.0")
print(f"  Threshold = 0.5")
print(f"\nHalf-height search:")
print(f"  Left: signal2[50] = {signal2[50]:.1f} > 0.5, move left")
print(f"        signal2[49] = {signal2[49]:.1f} == 0.5, STOP (not > threshold)")
print(f"        left_idx = 49")
print(f"  Right: signal2[50] = {signal2[50]:.1f} > 0.5, move right")
print(f"         signal2[51] = {signal2[51]:.1f} == 0.5, STOP (not > threshold)")
print(f"         right_idx = 51")
print(f"  width = 51 - 49 = 2")
print(f"  std = 2 / 2.3548 = {2/2.3548:.2f}")
print(f"  std_dict = {std_dict2}")

print("\n" + "="*80)
print("The Real Issue:")
print("="*80)
print("""
When compute_gauss_std is called with ALL components together (line 1394):
  std_dict = compute_gauss_std(sig_detrended, guess_idxs)

The signal might have:
1. Interference from other components (Q, S) that affect the half-height search
2. Signal shape that doesn't allow clean half-height detection
3. Detrending artifacts that affect the signal around R
4. The detected R height might not match the actual signal value at that location

Even though R is DETECTED (we know center_idx and height), the std ESTIMATION
can fail if the half-height search doesn't work properly.

The early computation (line 277) works because:
- It's called with ONLY R: r_guess = {"R": (r_center_idx, r_height)}
- The signal is cleaner at that point
- There's no interference from other components yet

The later computation (line 1394) can fail because:
- It's called with ALL components together
- The signal might have been modified or have interference
- The half-height search might be affected by overlapping components
""")

print("\n" + "="*80)
print("Solution:")
print("="*80)
print("""
Use the pre-computed r_std as a fallback when std_dict.get("R") is None.
This ensures R is always included in the Gaussian fit when detected, allowing
shape features to be computed.
""")



