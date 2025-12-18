import numpy as np
from scipy.stats import norm
from typing import Dict, Tuple, Optional, Union


def compute_gauss_std(
    signal: np.ndarray,
    guess_idxs: Dict[str, Tuple[int, float]],
) -> Dict[str, float]:
    """
    Estimate Gaussian std for each labeled peak based on full width at half max.

    Parameters
    ----------
    signal : 1D numpy array
        Detrended ECG signal.
    guess_idxs : dict
        {label: (center_idx, height)}

    Returns
    -------
    std_dict : dict
        {label: estimated_std} in samples
    """
    HALF_FRAC = 0.5  # fixed threshold fraction
    FWHM_TO_STD = 2.3548

    def from_center(center_idx: int, height: float) -> float:
        if center_idx is None or np.isnan(height):
            return None

        threshold = HALF_FRAC * height
        left_idx, right_idx = center_idx, center_idx

        if height >= 0:
            while left_idx > 0 and signal[left_idx] > threshold:
                left_idx -= 1
            while right_idx < len(signal) - 1 and signal[right_idx] > threshold:
                right_idx += 1
        else:
            while left_idx > 0 and signal[left_idx] < threshold:
                left_idx -= 1
            while right_idx < len(signal) - 1 and signal[right_idx] < threshold:
                right_idx += 1

        width = right_idx - left_idx
        return width / FWHM_TO_STD if width > 0 else None

    std_dict = {}
    for label, (center, height) in guess_idxs.items():
        std = from_center(center, height)
        if std is not None:
            std_dict[label] = max(std, 1.0)  # min for stability

    return std_dict


from typing import Union
import numpy as np

def gaussian_function(
    xs: Union[np.ndarray, list[float]],
    *features: float
) -> np.ndarray:
    """
    Compute the sum of one or more Gaussian functions.

    Parameters
    ----------
    xs : array-like
        1D array of x-axis values.
    *features : float
        Sequence of Gaussian parameters in groups of three:
        (center, height, width).
        Multiple Gaussians can be specified by providing
        multiple triples in sequence.

    Returns
    -------
    np.ndarray
        Output array where each element is the sum of all
        Gaussian components evaluated at the corresponding `xs` point.

    Notes
    -----
    - Width values <= 0 are clamped to a small positive constant
      (`1e-10`) to avoid division-by-zero errors.
    - The function supports an arbitrary number of Gaussian components.
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.zeros_like(xs, dtype=float)
    min_width = 1e-10  # avoid division by zero

    for i in range(0, len(features), 3):
        center, height, width = features[i : i + 3]
        width = max(width, min_width)
        ys += height * np.exp(-((xs - center) ** 2) / (2 * width**2))

    return ys


def skewed_gaussian_function(xs, *features):
    """
    Compute the sum of one or more skewed Gaussian functions.
    
    Uses a skew-normal distribution parameterization where:
    - alpha = 0 gives a symmetric Gaussian
    - alpha > 0 gives right-skewed (longer tail on right)
    - alpha < 0 gives left-skewed (longer tail on left)
    
    Parameters
    ----------
    xs : 1d array
        Input x-axis values.
    *features : float
        Sequence of skewed Gaussian parameters in groups of four:
        (center, height, width, alpha).
        Multiple peaks can be specified by providing multiple quadruples.
        
    Returns
    -------
    ys : 1d array
        Output values for sum of skewed Gaussian functions.
        
    Notes
    -----
    The skew-normal PDF is: f(x) = 2 * phi(x) * Phi(alpha * x)
    where phi is the standard normal PDF and Phi is the standard normal CDF.
    We scale by height and shift/scale by center and width.
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.zeros_like(xs, dtype=float)
    min_width = 1e-10  # avoid division by zero
    
    for ii in range(0, len(features), 4):
        ctr, hgt, wid, alpha = features[ii : ii + 4]
        wid = max(wid, min_width)
        
        # Standardized variable
        z = (xs - ctr) / wid
        
        # Standard Gaussian component (unnormalized)
        gaussian_part = np.exp(-0.5 * z**2)
        
        # Skew component: CDF of standard normal at alpha * z
        skew_part = norm.cdf(alpha * z)
        
        # Combined skew-normal (factor of 2 for proper normalization)
        skew_gaussian = 2.0 * gaussian_part * skew_part
        
        # Scale to desired height (normalize peak to 1, then multiply by height)
        if np.max(skew_gaussian) > 0:
            skew_gaussian = (skew_gaussian / np.max(skew_gaussian)) * hgt
        
        ys += skew_gaussian
    
    return ys


def calc_r_squared(
    signal: Union[np.ndarray, list[float]],
    fit: Union[np.ndarray, list[float]]
) -> float:
    """
    Calculate the coefficient of determination (R²) between an input signal and its fitted model.

    Parameters
    ----------
    signal : array-like
        Original data (observed values).
    fit : array-like
        Model-fitted data (predicted values).

    Returns
    -------
    float
        R² goodness-of-fit value in the range [0, 1].
        Returns np.nan if correlation cannot be computed.
    """
    signal = np.asarray(signal, dtype=float)
    fit = np.asarray(fit, dtype=float)

    # Handle NaNs or mismatched lengths early
    if signal.shape != fit.shape or signal.size == 0:
        return np.nan

    # Compute correlation coefficient
    r_matrix = np.corrcoef(signal, fit)
    r_value = r_matrix[0, 1]

    # Return R², guarding against NaN correlation
    return float(r_value ** 2) if np.isfinite(r_value) else np.nan