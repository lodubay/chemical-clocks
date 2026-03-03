r"""
Generic statistical routines for this project.
"""

import numpy as np
from multizone._globals import RANDOM_SEED


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
    if len(x)>0:
        rng = np.random.default_rng(seed)
        # Randomly sample input array *with* replacement, all at once
        samples = rng.choice(x, size=len(x) * B, replace=True).reshape((B, len(x)))
        medians = np.median(samples, axis=1)
        # The standard error is the standard deviation of the medians
        return np.std(medians)
    else:
        return np.nan


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


def kde2D(x, y, bandwidth, xbins=100j, ybins=100j, **kwargs):
    """Build 2D kernel density estimate (KDE).

    Parameters
    ----------
    x : array-like
    y : array-like
    bandwidth : float
    xbins : complex, optional [default: 100j]
    ybins : complex, optional [default: 100j]

    Other keyword arguments are passed to sklearn.neighbors.KernelDensity

    Returns
    -------
    xx : MxN numpy array
        Density grid x-coordinates (M=xbins, N=ybins)
    yy : MxN numpy array
        Density grid y-coordinates
    logz : MxN numpy array
        Grid of log-likelihood density estimates
    """
    from sklearn.neighbors import KernelDensity
    # Error handling for xbins and ybins
    if type(xbins) == np.ndarray and type(ybins) == np.ndarray:
        if xbins.shape == ybins.shape:
            if len(xbins.shape) == 2 and len(ybins.shape) == 2:
                xx = xbins
                yy = ybins
            else:
                raise ValueError('Input xbins and ybins must have dimension 2.')
        else:
            raise ValueError('Got xbins and ybins of different shape.')
    elif type(xbins) == complex and type(ybins) == complex:
        # create grid of sample locations (default: 100x100)
        xx, yy = np.mgrid[x.min():x.max():xbins,
                          y.min():y.max():ybins]
    else:
        raise TypeError('Input xbins and ybins must have type complex ' + \
                        '(e.g. 100j) or numpy.ndarray.')

    xy_sample = np.vstack([yy.ravel(), xx.ravel()]).T
    xy_train  = np.vstack([y, x]).T

    kde_skl = KernelDensity(kernel='gaussian', bandwidth=bandwidth, **kwargs)
    kde_skl.fit(xy_train)

    # score_samples() returns the log-likelihood of the samples
    logz = kde_skl.score_samples(xy_sample)
    return xx, yy, np.reshape(logz, xx.shape)
