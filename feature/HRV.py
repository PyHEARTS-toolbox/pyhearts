import numpy as np


def calc_hrv_metrics(rr_intervals: np.ndarray):
    """
    Calculate basic heart rate variability (HRV) metrics from R-R intervals.

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
    """
    clean_rr_intervals = rr_intervals[~np.isnan(rr_intervals)]

    # Calculate instantaneous heart rate in bpm
    heart_rate = 60 / (clean_rr_intervals / 1000)
    average_heart_rate = np.nanmean(heart_rate) if len(heart_rate) > 0 else np.nan

    if len(clean_rr_intervals) > 1:
        sdnn = np.std(clean_rr_intervals, ddof=1)
        rmssd = np.sqrt(np.mean(np.diff(clean_rr_intervals) ** 2))
        nn50 = int(np.sum(np.abs(np.diff(clean_rr_intervals)) > 50))
    else:
        sdnn, rmssd, nn50 = np.nan, np.nan, np.nan

    # Round and convert to int, handling NaNs safely
    def safe_int(val):
        return int(round(val)) if not np.isnan(val) else None

    return (
        safe_int(average_heart_rate),
        safe_int(sdnn),
        safe_int(rmssd),
        nn50 if not np.isnan(nn50) else None
    )
