#!/usr/bin/env python3
"""
Test half-height detection limitations for R peaks.
"""

import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.absolute()))
from pyhearts.processing.gaussian import compute_gauss_std

print("="*80)
print("Testing Half-Height Detection Limitations for R Peaks")
print("="*80)

# Test 1: Inverted R peak (negative height)
print("\n1. INVERTED R PEAK (negative height):")
signal1 = np.zeros(100)
signal1[45:55] = -np.exp(-((np.arange(10) - 5) ** 2) / (2 * 2 ** 2)) * 1.0  # Inverted R
guess_idxs1 = {"R": (50, -1.0)}  # Negative height
std_dict1 = compute_gauss_std(signal1, guess_idxs1)
print(f"   R height: -1.0 (inverted)")
print(f"   Threshold: 0.5 * (-1.0) = -0.5")
print(f"   Signal[50] = {signal1[50]:.3f}")
print(f"   Search looks for signal < -0.5 (since height < 0)")
print(f"   Result: {std_dict1}")
print(f"   R std: {std_dict1.get('R', 'NOT FOUND (None)')}")

# Test 2: R peak where signal never crosses threshold (flat top)
print("\n2. R PEAK WITH FLAT TOP (signal never drops to half-height):")
signal2 = np.zeros(100)
signal2[40:60] = 0.8  # Flat top at 0.8, but R detected height is 1.0
signal2[50] = 1.0  # Peak point
guess_idxs2 = {"R": (50, 1.0)}
std_dict2 = compute_gauss_std(signal2, guess_idxs2)
print(f"   R detected height: 1.0")
print(f"   Threshold: 0.5")
print(f"   Signal[40:60] = {signal2[40:60][:5]}... (all 0.8)")
print(f"   Problem: Signal never drops below 0.5, so search continues to boundaries")
print(f"   Result: {std_dict2}")
print(f"   R std: {std_dict2.get('R', 'NOT FOUND (None)')}")

# Test 3: R peak where detected height doesn't match actual signal
print("\n3. R PEAK WHERE DETECTED HEIGHT > ACTUAL SIGNAL:")
signal3 = np.zeros(100)
signal3[45:55] = np.exp(-((np.arange(10) - 5) ** 2) / (2 * 2 ** 2)) * 0.6  # Actual peak is 0.6
guess_idxs3 = {"R": (50, 1.0)}  # But detected as 1.0
std_dict3 = compute_gauss_std(signal3, guess_idxs3)
print(f"   R detected height: 1.0")
print(f"   Actual signal[50]: {signal3[50]:.3f}")
print(f"   Threshold: 0.5")
print(f"   Problem: Signal at center (0.6) is above threshold, but shape is wrong")
print(f"   Result: {std_dict3}")
print(f"   R std: {std_dict3.get('R', 'NOT FOUND (None)')}")

# Test 4: R peak with Q/S interference
print("\n4. R PEAK WITH Q/S INTERFERENCE (overlapping components):")
signal4 = np.zeros(100)
# Q wave (negative) overlaps with R
signal4[40:50] = -np.exp(-((np.arange(10) - 5) ** 2) / (2 * 2 ** 2)) * 0.4
# R wave (positive) overlaps with Q
signal4[45:55] += np.exp(-((np.arange(10) - 5) ** 2) / (2 * 2 ** 2)) * 1.0
# S wave (negative) overlaps with R
signal4[50:60] += -np.exp(-((np.arange(10) - 5) ** 2) / (2 * 2 ** 2)) * 0.4
guess_idxs4 = {"R": (50, 1.0)}
std_dict4 = compute_gauss_std(signal4, guess_idxs4)
print(f"   R detected height: 1.0")
print(f"   Threshold: 0.5")
print(f"   Signal[45:55] = {signal4[45:55]}")
print(f"   Problem: Q/S interference affects half-height search")
print(f"   Result: {std_dict4}")
print(f"   R std: {std_dict4.get('R', 'NOT FOUND (None)')}")

# Test 5: Very narrow R peak (single sample or very sharp)
print("\n5. VERY NARROW R PEAK (single sample or very sharp):")
signal5 = np.zeros(100)
signal5[50] = 1.0  # Single point
signal5[49] = 0.3
signal5[51] = 0.3
guess_idxs5 = {"R": (50, 1.0)}
std_dict5 = compute_gauss_std(signal5, guess_idxs5)
print(f"   R detected height: 1.0")
print(f"   Threshold: 0.5")
print(f"   Signal[49:52] = {signal5[49:52]}")
print(f"   Left search: signal[49] = 0.3 < 0.5, stop at 49")
print(f"   Right search: signal[51] = 0.3 < 0.5, stop at 51")
print(f"   Width: 51 - 49 = 2")
print(f"   Result: {std_dict5}")
print(f"   R std: {std_dict5.get('R', 'NOT FOUND (None)')}")

# Test 6: R peak where width calculation gives 0 or negative
print("\n6. R PEAK WHERE WIDTH <= 0 (edge case):")
signal6 = np.zeros(100)
signal6[50] = 1.0
# Signal exactly at threshold on both sides
signal6[49] = 0.5
signal6[51] = 0.5
guess_idxs6 = {"R": (50, 1.0)}
std_dict6 = compute_gauss_std(signal6, guess_idxs6)
print(f"   R detected height: 1.0")
print(f"   Threshold: 0.5")
print(f"   Signal[49:52] = {signal6[49:52]}")
print(f"   Left search: signal[49] = 0.5 (not > 0.5), stop at 49")
print(f"   Right search: signal[51] = 0.5 (not > 0.5), stop at 51")
print(f"   Width: 51 - 49 = 2 (should work)")
print(f"   Result: {std_dict6}")
print(f"   R std: {std_dict6.get('R', 'NOT FOUND (None)')}")

# Test 7: R peak at signal boundary
print("\n7. R PEAK AT SIGNAL BOUNDARY:")
signal7 = np.zeros(50)
signal7[0:10] = np.exp(-((np.arange(10) - 5) ** 2) / (2 * 2 ** 2)) * 1.0
signal7[0] = 0.0  # Boundary
guess_idxs7 = {"R": (5, 1.0)}
std_dict7 = compute_gauss_std(signal7, guess_idxs7)
print(f"   R detected height: 1.0")
print(f"   Threshold: 0.5")
print(f"   Signal[0:10] = {signal7[0:10]}")
print(f"   Left search hits boundary at 0")
print(f"   Result: {std_dict7}")
print(f"   R std: {std_dict7.get('R', 'NOT FOUND (None)')}")

# Test 8: R peak in detrended signal with baseline offset
print("\n8. R PEAK IN DETRENDED SIGNAL (baseline offset):")
signal8 = np.ones(100) * 0.3  # Baseline offset
signal8[45:55] += np.exp(-((np.arange(10) - 5) ** 2) / (2 * 2 ** 2)) * 1.0
guess_idxs8 = {"R": (50, 1.0)}  # Height is relative to baseline
std_dict8 = compute_gauss_std(signal8, guess_idxs8)
print(f"   R detected height: 1.0 (relative)")
print(f"   Actual signal[50]: {signal8[50]:.3f}")
print(f"   Threshold: 0.5")
print(f"   Problem: Threshold might not account for baseline")
print(f"   Result: {std_dict8}")
print(f"   R std: {std_dict8.get('R', 'NOT FOUND (None)')}")

print("\n" + "="*80)
print("KEY FINDINGS:")
print("="*80)
print("""
1. INVERTED PEAKS: The code DOES handle inverted peaks (height < 0)
   - Uses signal[left_idx] < threshold (instead of >)
   - Should work for inverted R peaks

2. LIMITATIONS THAT CAUSE FAILURE:
   a) Width <= 0: If left_idx >= right_idx, returns None
   b) Signal never crosses threshold: Search continues to boundaries
      (but this still returns a value, just very large)
   c) Detected height mismatch: If actual signal doesn't match detected height,
      threshold calculation is wrong

3. REAL-WORLD ISSUES:
   - Q/S interference can affect half-height search
   - Detrending artifacts can affect signal shape
   - Baseline offsets can affect threshold calculation
   - Very narrow peaks might have width issues

4. THE ACTUAL PROBLEM:
   The code rarely returns None in practice (width > 0 is usually true).
   The real issue might be that when called with ALL components together,
   the signal has interference that makes the std estimate unreliable,
   OR the early computation (line 277) uses a cleaner signal state.
""")



