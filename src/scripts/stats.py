r"""
Generic statistical routines for this project.
"""

import numpy as np
from _globals import RANDOM_SEED


def median_standard_error(x, B=1000, seed=RANDOM_SEED):
    """
    Use bootstrapping to calculate the standard error of the median.
    
    Parameters
    ----------
    x : array-like
        Data array.
    B : int, optional
        Number of bootstrap samples. The default is 1000.
    
    Returns
    -------
    float
        Standard error of the median.
    """
    rng = np.random.default_rng(seed)
    # Randomly sample input array *with* replacement, all at once
    samples = rng.choice(x, size=len(x) * B, replace=True).reshape((B, len(x)))
    medians = np.median(samples, axis=1)
    # The standard error is the standard deviation of the medians
    return np.std(medians)


def weighted_quantile(df, val, weight, quantile=0.5):
    """
    Calculate the quantile of a pandas column weighted by another column.
    
    Parameters
    ----------
    df : pandas.DataFrame
    val : str
        Name of values column.
    weight : str
        Name of weights column.
    quantile : float, optional
        The quantile to calculate. Must be in [0,1]. The default is 0.5.
    
    Returns
    -------
    wq : float
        The weighted quantile of the dataframe column.
    """
    if quantile >= 0 and quantile <= 1:
        if df.shape[0] == 0:
            return np.nan
        else:
            df_sorted = df.sort_values(val)
            cumsum = df_sorted[weight].cumsum()
            cutoff = df_sorted[weight].sum() * quantile
            wq = df_sorted[cumsum >= cutoff][val].iloc[0]
            return wq
    else:
        raise ValueError("Quantile must be in range [0,1].")
