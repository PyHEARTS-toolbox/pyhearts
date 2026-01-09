"""
Beat-to-beat variability metrics for morphological features.

Computes variability statistics (std, CV, IQR, MAD) across cycles for key features.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


def compute_variability_metrics(
    feature_series: np.ndarray,
    feature_name: str = "feature"
) -> Dict[str, float]:
    """
    Compute variability metrics for a feature series across cycles.
    
    Parameters
    ----------
    feature_series : np.ndarray
        Array of feature values across cycles. May contain NaN values.
    feature_name : str
        Name of the feature (for error messages).
    
    Returns
    -------
    Dict[str, float]
        Dictionary with keys:
        - '{feature_name}_std': Standard deviation
        - '{feature_name}_cv': Coefficient of variation (std/mean)
        - '{feature_name}_iqr': Interquartile range (75th - 25th percentile)
        - '{feature_name}_mad': Median absolute deviation
        - '{feature_name}_range': Range (max - min)
        Values are np.nan if insufficient valid data.
    """
    # Remove NaN values
    clean_values = feature_series[~np.isnan(feature_series)]
    
    if len(clean_values) < 2:
        # Need at least 2 values for variability metrics
        return {
            f"{feature_name}_std": np.nan,
            f"{feature_name}_cv": np.nan,
            f"{feature_name}_iqr": np.nan,
            f"{feature_name}_mad": np.nan,
            f"{feature_name}_range": np.nan,
        }
    
    # Standard deviation
    std_val = float(np.std(clean_values, ddof=1))
    
    # Coefficient of variation (CV = std / mean)
    mean_val = float(np.mean(clean_values))
    if abs(mean_val) > 1e-6:  # Avoid division by zero
        cv_val = float(std_val / abs(mean_val))
    else:
        cv_val = np.nan
    
    # Interquartile range (IQR = 75th percentile - 25th percentile)
    q25 = float(np.percentile(clean_values, 25))
    q75 = float(np.percentile(clean_values, 75))
    iqr_val = float(q75 - q25)
    
    # Median absolute deviation (MAD)
    median_val = float(np.median(clean_values))
    mad_val = float(1.4826 * np.median(np.abs(clean_values - median_val)))
    
    # Range (max - min)
    range_val = float(np.max(clean_values) - np.min(clean_values))
    
    return {
        f"{feature_name}_std": std_val,
        f"{feature_name}_cv": cv_val,
        f"{feature_name}_iqr": iqr_val,
        f"{feature_name}_mad": mad_val,
        f"{feature_name}_range": range_val,
    }


def compute_beat_to_beat_variability(
    output_dict: Dict[str, List],
    priority_features: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Compute beat-to-beat variability metrics for priority features.
    
    Parameters
    ----------
    output_dict : Dict[str, List]
        Dictionary of feature series (lists) from PyHEARTS output.
    priority_features : List[str], optional
        List of feature names to compute variability for.
        If None, uses default priority features.
    
    Returns
    -------
    Dict[str, float]
        Dictionary mapping variability metric names to values.
        Format: '{feature_name}_{metric}' where metric is one of:
        std, cv, iqr, mad, range
    """
    if priority_features is None:
        # Default priority features for age/disease prediction
        priority_features = [
            # Interval features
            "QT_interval_ms",
            "QRS_interval_ms",
            "PR_interval_ms",
            "RR_interval_ms",
            # QTc features
            "QTc_Bazett_ms",
            "QTc_Fridericia_ms",
            # Wave amplitudes (heights)
            "R_gauss_height",
            "P_gauss_height",
            "T_gauss_height",
            # Wave durations
            "R_duration_ms",
            "P_duration_ms",
            "T_duration_ms",
            # Additional key features
            "QRS_interval_ms",
            "ST_segment_ms",
        ]
    
    variability_metrics = {}
    
    for feature_name in priority_features:
        if feature_name not in output_dict:
            # Feature not in output, skip
            continue
        
        feature_series = np.array(output_dict[feature_name])
        
        # Compute variability metrics
        metrics = compute_variability_metrics(feature_series, feature_name)
        variability_metrics.update(metrics)
    
    return variability_metrics


