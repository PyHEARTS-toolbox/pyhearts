from typing import List, Tuple, Union


def calc_bounds(
    center: int,
    height: float,
    std: int,
    bound_factor: float,
    flip_height: bool = False
) -> Tuple[List[Union[int, float]], List[Union[int, float]]]:
    """
    Calculate lower and upper bounds for a Gaussian component's features: 
    center (mu), height (amplitude), and standard deviation (sigma).

    Ensures lower < upper for all bounds. Height bounds respect flip_height logic
    and allow for negative peaks.

    Parameters
    ----------
    center : int
        Center position of the Gaussian.
    height : float
        Amplitude of the Gaussian peak.
    std : int
        Standard deviation of the Gaussian.
    bound_factor : float, optional
        Proportion to scale bounds around each feature (default is 0.2).
    flip_height : bool, optional
        If True, reverses direction of height bounds (for negative peaks).

    Returns
    -------
    Tuple[List[Union[int, float]], List[Union[int, float]]]
        Lower bounds and upper bounds, each as a list of [center, height, std].
    """
    delta = int(bound_factor * std)

    # --- Center bounds ---
    center_lower = center - delta
    center_upper = center + delta
    if center_upper - center_lower < 2:
        offset = ((2 - (center_upper - center_lower)) // 2) + 1
        center_lower -= offset
        center_upper += offset

    # --- Height bounds ---
    factor_low = 1 - bound_factor
    factor_high = 1 + bound_factor

    if flip_height ^ (height < 0):  # XOR for flipping
        lower_height = height * factor_high
        upper_height = height * factor_low
    else:
        lower_height = height * factor_low
        upper_height = height * factor_high

    # Ensure lower_height < upper_height
    if lower_height == upper_height:
        epsilon = 1e-6 if height >= 0 else -1e-6
        upper_height += epsilon

    elif lower_height > upper_height:
        lower_height, upper_height = upper_height, lower_height

    # --- Std bounds ---
    lower_std = max(int(std * factor_low), 1)
    upper_std = int(std * factor_high)

    # Enforce wider bounds for very small std values
    if std < 2:
        lower_std = 1
        upper_std = 5  # ensures upper - lower = 4

    # General case: Ensure lower_std < upper_std and a minimum width
    elif lower_std >= upper_std or (upper_std - lower_std < 1):
        if lower_std == 1:
            upper_std = max(upper_std, 2)
        else:
            lower_std = max(lower_std - 1, 1)
            upper_std = lower_std + 1

    return [center_lower, lower_height, lower_std], [center_upper, upper_height, upper_std]


def calc_bounds_skewed(
    center: int,
    height: float,
    std: int,
    alpha: float,
    bound_factor: float,
    skew_bounds: Tuple[float, float] = (-3.0, 3.0),
    flip_height: bool = False
) -> Tuple[List[Union[int, float]], List[Union[int, float]]]:
    """
    Calculate bounds for a skewed Gaussian component: center, height, std, and alpha (skew).
    
    Parameters
    ----------
    center : int
        Center position of the Gaussian.
    height : float
        Amplitude of the Gaussian peak.
    std : int
        Standard deviation of the Gaussian.
    alpha : float
        Skew parameter (0 = symmetric, >0 = right-skewed, <0 = left-skewed).
    bound_factor : float
        Proportion to scale bounds around each feature.
    skew_bounds : Tuple[float, float]
        Min and max allowed values for alpha (default (-3.0, 3.0)).
    flip_height : bool
        If True, reverses direction of height bounds.
        
    Returns
    -------
    Tuple[List, List]
        Lower and upper bounds as [center, height, std, alpha].
    """
    # Get standard Gaussian bounds
    lower_3, upper_3 = calc_bounds(center, height, std, bound_factor, flip_height)
    
    # Add alpha bounds
    alpha_lo, alpha_hi = skew_bounds
    
    # Allow alpha to vary within bounds, centered on initial guess
    alpha_range = (alpha_hi - alpha_lo) * bound_factor
    lower_alpha = max(alpha - alpha_range, alpha_lo)
    upper_alpha = min(alpha + alpha_range, alpha_hi)
    
    # Ensure lower < upper
    if lower_alpha >= upper_alpha:
        lower_alpha = alpha_lo
        upper_alpha = alpha_hi
    
    return lower_3 + [lower_alpha], upper_3 + [upper_alpha]
