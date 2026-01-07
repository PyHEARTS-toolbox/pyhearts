#!/usr/bin/env python3
"""
Test if single-component (R only) Gaussian fitting works.
"""

import numpy as np
from scipy.optimize import curve_fit
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.absolute()))
from pyhearts.processing.gaussian import gaussian_function

# Test single-component fit
print("Testing single-component (R only) Gaussian fitting:")
print("="*80)

# Create a simple R peak signal
signal = np.zeros(100)
signal[45:55] = np.exp(-((np.arange(10) - 5) ** 2) / (2 * 2 ** 2)) * 1.0
xs = np.arange(len(signal))

# Initial guess for single R peak
p0 = [50, 1.0, 2.0]  # center, height, std

# Bounds
lower_bounds = [40, 0.5, 1.0]
upper_bounds = [60, 2.0, 5.0]
bounds = (lower_bounds, upper_bounds)

try:
    fit, _ = curve_fit(
        gaussian_function,
        xs,
        signal,
        p0=p0,
        bounds=bounds,
        method="trf",
        maxfev=1000
    )
    print(f"✓ Single-component fit SUCCEEDED")
    print(f"  Fitted parameters: center={fit[0]:.2f}, height={fit[1]:.2f}, std={fit[2]:.2f}")
except Exception as e:
    print(f"✗ Single-component fit FAILED: {e}")

# Test multi-component fit (for comparison)
print("\nTesting multi-component (P, Q, R, S, T) Gaussian fitting:")
signal2 = np.zeros(200)
signal2[20:30] = np.exp(-((np.arange(10) - 5) ** 2) / (2 * 3 ** 2)) * 0.1  # P
signal2[40:45] = -np.exp(-((np.arange(5) - 2) ** 2) / (2 * 1 ** 2)) * 0.2  # Q
signal2[50:60] = np.exp(-((np.arange(10) - 5) ** 2) / (2 * 2 ** 2)) * 1.0  # R
signal2[65:70] = -np.exp(-((np.arange(5) - 2) ** 2) / (2 * 1 ** 2)) * 0.2  # S
signal2[100:130] = np.exp(-((np.arange(30) - 15) ** 2) / (2 * 8 ** 2)) * 0.3  # T
xs2 = np.arange(len(signal2))

p0_multi = [25, 0.1, 3.0,  # P
            42, -0.2, 1.0,  # Q
            55, 1.0, 2.0,   # R
            67, -0.2, 1.0,  # S
            115, 0.3, 8.0]  # T

lower_multi = [15, 0.05, 1.0, 40, -0.3, 0.5, 45, 0.5, 1.0, 60, -0.3, 0.5, 100, 0.1, 5.0]
upper_multi = [35, 0.2, 5.0, 50, -0.1, 2.0, 65, 2.0, 5.0, 75, -0.1, 2.0, 130, 0.5, 15.0]
bounds_multi = (lower_multi, upper_multi)

try:
    fit_multi, _ = curve_fit(
        gaussian_function,
        xs2,
        signal2,
        p0=p0_multi,
        bounds=bounds_multi,
        method="trf",
        maxfev=1000
    )
    print(f"✓ Multi-component fit SUCCEEDED")
    print(f"  Fitted {len(fit_multi)//3} components")
except Exception as e:
    print(f"✗ Multi-component fit FAILED: {e}")

print("\n" + "="*80)
print("Conclusion:")
print("="*80)
print("Single-component Gaussian fitting should work fine.")
print("The issue is likely that R is being excluded from guess_dict")
print("before the fitting step, not that the fitting itself fails.")


