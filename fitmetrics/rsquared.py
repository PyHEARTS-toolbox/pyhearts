import numpy as np

def calc_r_squared(sig, fit):
    """
    Calculate the coefficient of determination (R-squared) between two arrays.

    Parameters
    ----------
    sig : array_like
        The original data or observed signal.
    fit : array_like
        The model-predicted or fitted data.

    Returns
    -------
    float
        The R-squared value, representing how well the fit approximates the original data.
        Ranges from 0 to 1, where 1 indicates a perfect fit.
    """
    # Compute the Pearson correlation coefficient matrix
    r_val = np.corrcoef(sig, fit)

    # Square the correlation coefficient to get R-squared
    r_squared = r_val[0, 1] ** 2

    return r_squared
