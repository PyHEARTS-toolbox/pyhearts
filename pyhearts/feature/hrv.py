import numpy as np


def calc_hrv_metrics(rr_intervals: np.ndarray):
    """
    Calculate heart rate variability (HRV) metrics from R-R intervals.

    Parameters
    ----------
    rr_intervals : np.ndarray
        Array of R-R intervals in milliseconds (ms). May contain NaN values.

    Returns
    -------
    average_heart_rate : int
        Mean heart rate in beats per minute (bpm), rounded to nearest int.
    sdnn : int
        Standard deviation of NN intervals (SDNN), rounded to nearest int.
    rmssd : int
        Root mean square of successive differences (RMSSD), rounded to nearest int.
    nn50 : int
        Number of successive RR interval differences greater than 50 ms (NN50).
    pnn50 : float
        Percentage of successive RR differences > 50ms, rounded to 2 decimal places.
    sd1 : float
        Short-term HRV from Poincaré plot (perpendicular to line of identity), rounded to 2 decimal places.
    sd2 : float
        Long-term HRV from Poincaré plot (along line of identity), rounded to 2 decimal places.
    """
    clean_rr_intervals = rr_intervals[~np.isnan(rr_intervals)]

    # Calculate instantaneous heart rate in bpm
    heart_rate = 60 / (clean_rr_intervals / 1000)
    average_heart_rate = np.nanmean(heart_rate) if len(heart_rate) > 0 else np.nan

    if len(clean_rr_intervals) > 1:
        sdnn = np.std(clean_rr_intervals, ddof=1)
        rmssd = np.sqrt(np.mean(np.diff(clean_rr_intervals) ** 2))
        nn50 = int(np.sum(np.abs(np.diff(clean_rr_intervals)) > 50))
        
        # Calculate pNN50: percentage of successive differences > 50ms
        n_pairs = len(clean_rr_intervals) - 1
        if n_pairs > 0:
            pnn50 = (nn50 / n_pairs) * 100.0
        else:
            pnn50 = np.nan
        
        # Calculate Poincaré plot metrics (SD1 and SD2)
        # SD1: Short-term variability (perpendicular to line of identity)
        # SD1 = RMSSD / sqrt(2)
        sd1 = rmssd / np.sqrt(2.0) if np.isfinite(rmssd) else np.nan
        
        # SD2: Long-term variability (along line of identity)
        # SD2 = sqrt(2 * SDNN^2 - SD1^2)
        if np.isfinite(sdnn) and np.isfinite(sd1):
            sd2_squared = 2.0 * (sdnn ** 2) - (sd1 ** 2)
            sd2 = np.sqrt(sd2_squared) if sd2_squared >= 0 else np.nan
        else:
            sd2 = np.nan
    else:
        sdnn, rmssd, nn50, pnn50, sd1, sd2 = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    # Round and convert to appropriate types, handling NaNs safely
    def safe_int(val):
        return int(round(val)) if not np.isnan(val) else None
    
    def safe_float(val, decimals=2):
        return round(float(val), decimals) if not np.isnan(val) else None

    return (
        safe_int(average_heart_rate),
        safe_int(sdnn),
        safe_int(rmssd),
        nn50 if not np.isnan(nn50) else None,
        safe_float(pnn50, 2),
        safe_float(sd1, 2),
        safe_float(sd2, 2),
    )
