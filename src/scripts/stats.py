r"""
Generic statistical routines for this project.
"""

import numpy as np
from multizone._globals import RANDOM_SEED

def deming_regression(x_obs, y_obs, x_err, y_err):
    """
    Calculate the Deming regression, a variant of least-squares regression
    that accounts for errors in both x and y.
    
    Parameters
    ----------
    x_obs : numpy.ndarray
        Array of length N with x-values of observed data.
    y_obs : numpy.ndarray
        Array of length N with y-values of observed data.
    x_err : numpy.ndarray
        Array of length N with uncertainties on the x-values. Note that this is
        assumed to be the standard deviation of the errors, which are normally
        distributed and centered on 0.
    y_err : numpy.ndarray
        Array of length N with uncertainties on the y-values.
    
    Returns
    -------
    beta : np.ndarray of length 2
        Parameters of best-fit line, with beta[0] the slope and beta[1] the
        intercept.

    Notes
    -----
    The Deming regression assumes that the ratio of the error variances is
    constant.
    """
    # Center the data
    mean_x = np.mean(x_obs)
    x_obs = x_obs - mean_x
    mean_y = np.mean(y_obs)
    y_obs = y_obs - mean_y

    # Ratio of variances of the errors
    # sigma_x = np.std(x_err)
    # sigma_y = np.std(y_err)
    sigma_x = np.mean(x_err)
    sigma_y = np.mean(y_err)
    lam = (sigma_y**2) / (sigma_x**2)

    # For centered data, compute the sums of squares:
    Sxx = np.sum(x_obs**2)
    Syy = np.sum(y_obs**2)
    Sxy = np.sum(x_obs * y_obs)

    # Least-squares estimate of model parameters
    beta_deming = (
        Syy - lam * Sxx + np.sqrt((Syy - lam * Sxx)**2 + 4 * lam * Sxy**2)
    ) / (2 * Sxy)
    emp_m_dem = beta_deming
    emp_b_dem = mean_y - beta_deming * mean_x
    
    return np.array([emp_m_dem, emp_b_dem])


def bootstrap_standard_error(func, *args, B=1000, seed=RANDOM_SEED, **kwargs):
    """
    Use bootstrapping to calculate the standard error of the given function.
    
    Parameters
    ----------
    func : <function>
        Function accepting one or several arrays of the same length.
    args : array-like
        Data array(s) to pass to func. If multiple, must all have same length.
    B : int [default: 1000]
        Number of bootstrap samples.
    seed : int [default: RANDOM_SEED]
        Seed for random number generator.
    **kwargs passed to func().

    Returns
    -------
    float or array-like
        Standard error of the given function return values.
    """
    assert len(args) > 0, 'Missing at least one positional argument for data.'
    nobs = len(args[0])
    assert all([nobs == len(x) for x in args]), \
        'Positional arguments must have equal length.'
    rng = np.random.default_rng(seed)
    vals = [] # return values from function
    for i in range(B):
        # Randomly sample input array(s) *with* replacement
        sample_indices = rng.integers(0, nobs, size=nobs)
        data_samples = [x[sample_indices] for x in args]
        vals.append(func(*data_samples, **kwargs))
    # The standard error is the standard deviation of the values
    return np.std(np.array(vals), axis=0)


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
