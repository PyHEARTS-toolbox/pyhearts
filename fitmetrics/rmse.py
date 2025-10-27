import numpy as np

def calc_rmse(observed, predicted):
    """
    Calculate the Root Mean Squared Error (RMSE) between two arrays.

    Parameters
    ----------
    observed : array_like
        The ground truth or actual observed values.
    predicted : array_like
        The model-predicted values.

    Returns
    -------
    float
        The RMSE value, representing the standard deviation of prediction errors.
        Lower values indicate better model performance.
    """
    # Compute the squared differences between observed and predicted values
    squared_errors = (observed - predicted) ** 2

    # Compute the mean of the squared errors
    mean_squared_error = np.mean(squared_errors)

    # Take the square root to obtain RMSE
    rmse = np.sqrt(mean_squared_error)

    return rmse
