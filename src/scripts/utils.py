"""
Utility functions and classes for many scripts.
"""

from numbers import Number

import numpy as np
from numpy.random import default_rng
import pandas as pd
from astropy.table import Table
import vice

from multizone._globals import RANDOM_SEED, MAX_SF_RADIUS, ZONE_WIDTH
from stats import median_standard_error

# =============================================================================
# SCIENCE FUNCTIONS
# =============================================================================

def alpha_cut(feh):
    """
    Dividing line between low- and high-alpha populations at a given [Fe/H].

    Parameters
    ----------
    feh : numpy.ndarray
        Array of [Fe/H] values.
    
    Returns
    -------
    numpy.ndarray
        Values of [Mg/Fe] that divide low- and high-alpha populations.
    """
    return np.where(
        feh >= 0.0,
        0.09,
        0.09 - 0.13*feh
    )


def vac2air(wl):
    """
    Convert vacuum wavelengths to wavelengths in air at STP.

    Parameters
    ----------
    wl : array-like
        Vacuum wavelengths in Angstroms.

    Returns
    -------
    array-like
        Air wavelengths in Angstroms.
    """
    # Wave number
    s = 10**4 / wl
    # Refraction index
    n = 1 + 0.0000834254 + 0.02406147 / (130 - s**2) + 0.00015998 / (38.9 - s**2)
    return wl / n


def air2vac(wl):
    """
    Convert wavelengths in air at STP to vacuum wavelengths.
    
    Parameters
    ----------
    wl : array-like
        Wavelengths in air in Angstroms.
    
    Returns
    -------
    array-like
        Vacuum wavelengths in Angstroms.
    """
    # Wave number
    s = 10**4 / wl
    # Index of refraction
    n = 1 + 0.00008336624212083 + 0.02408926869968 / (130.1065924522 - s**2) + 0.0001599740894897 / (38.92568793293 - s**2)
    return wl * n


# =============================================================================
# DATA UTILITY FUNCTIONS
# =============================================================================

def vice_to_mwm_col(col):
    """
    Convert VICE output abundance labels to MWM labels.
    
    Parameters
    ----------
    col : str
        Name of column in VICE multizone stars output.
    
    Returns
    -------
    str
        Column label in MWM DataFrame.
        
    """
    return col[1:-1].replace('/', '_')


def mwm_to_vice_col(col):
    """
    Convert MWM abundance labels to VICE output labels.
    
    Parameters
    ----------
    col : str
        Name of column in MWM DataFrame.
    
    Returns
    -------
    str
        Column label in VICE output.
        
    """
    return '[' + '/'.join(col.split('_')) + ']'


def get_bin_centers(bin_edges):
    """
    Calculate the centers of bins defined by the given bin edges.
    
    Parameters
    ----------
    bin_edges : array-like of length N
        Edges of bins, including the left-most and right-most bounds.
     
    Returns
    -------
    bin_centers : numpy.ndarray of length N-1
        Centers of bins
    """
    bin_edges = np.array(bin_edges, dtype=float)
    if len(bin_edges) > 1:
        return 0.5 * (bin_edges[:-1] + bin_edges[1:])
    else:
        raise ValueError('The length of bin_edges must be at least 2.')


def fits_to_pandas(path, **kwargs):
    """
    Import a table in the form of a FITS file and convert it to a pandas
    DataFrame.

    Parameters
    ----------
    path : Path or str
        Path to fits file
    Other keyword arguments are passed to astropy.table.Table

    Returns
    -------
    df : pandas DataFrame
    """
    # Read FITS file into astropy table
    table = Table.read(path, format='fits', **kwargs)
    # Filter out multidimensional columns
    cols = [name for name in table.colnames if len(table[name].shape) <= 1]
    # Convert byte-strings to ordinary strings and convert to pandas
    df = decode(table[cols].to_pandas())
    return df


def decode(df):
    """
    Decode DataFrame with byte strings into ordinary strings.

    Parameters
    ----------
    df : pandas DataFrame
    """
    str_df = df.select_dtypes([object])
    str_df = str_df.stack().str.decode('utf-8').unstack()
    for col in str_df:
        df[col] = str_df[col]
    return df


def box_smooth(hist, bins, width):
    """
    Box-car smoothing function for a pre-generated histogram.

    Parameters
    ----------
    bins : array-like
        Bins dividing the histogram, including the end. Length must be 1 more
        than the length of hist, and bins must be evenly spaced.
    hist : array-like
        Histogram of data
    width : float
        Width of the box-car smoothing function in data units
    """
    bin_width = bins[1] - bins[0]
    box_width = int(width / bin_width)
    box = np.ones(box_width) / box_width
    hist_smooth = np.convolve(hist, box, mode='same')
    return hist_smooth


def sample_rows(df, n, weights=None, reset=False, seed=RANDOM_SEED):
    """
    Randomly sample n unique rows from a pandas DataFrame.

    Parameters
    ----------
    df : pandas DataFrame
    n : int
        Number of random samples to draw
    weights : array, optional
        Probability weights of the given DataFrame
    reset : bool, optional
        If True, reset sample DataFrame index

    Returns
    -------
    pandas DataFrame
        Re-indexed DataFrame of n sampled rows
    """
    if isinstance(df, pd.DataFrame):
        # Number of samples can't exceed length of DataFrame
        n = min(n, df.shape[0])
        # Initialize default numpy random number generator
        rng = default_rng(seed)
        # Randomly sample without replacement
        rand_indices = rng.choice(df.index, size=n, replace=False, p=weights)
        sample = df.loc[rand_indices]
        if reset:
            sample.reset_index(inplace=True, drop=True)
        return sample
    else:
        raise TypeError('Expected pandas DataFrame.')
    
    
def binned_quantiles(
        data, col, bin_col, q=0.5, bins=50, bin_edges=[], min_count=0, 
        est_errors=False, seed=RANDOM_SEED, nsamples=1000,
    ):
    """
    Calculate percentile trends in bins of a second parameter.
    
    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame with at least two columns.
    col : str
        Data column corresponding to the first parameter, for which the
        intervals will be calculated in each bin.
    bin_col : str
        Data column corresponding to the second (binning) parameter.
    q : float, optional
        The quantile to calculate, 0 <= q <= 1.
    bins : int, optional
        The number of equal-size bins to divide the data along bin_col.
        The default is 50.
    bin_edges : array-like, optional
        Edges of bins for calculating the quantile. Will override the value
        of bins if provided.
    min_count : int, optional [default: 0]
        Minimum data count required to calculate a quantile. If there are fewer
        points in that bin, the quantile will be NaN.
    est_errors : bool, optional [default: False]
        If True, return an array of the standard error on the given quantile,
        estimated via bootstrapping. Note: this makes the calculation take
        a lot longer!
    nsamples : int, optional [default: 1000]
        Number of bootstrap samples to generate if est_errors == True.
    seed : int, optional [default: RANDOM_SEED]
        Seed for random number generator used for bootstrapping errors.
    
    Returns
    -------
    bin_centers : numpy.ndarray
        Center of each bin in bin_col.
    quantiles : numpy.ndarray
        Quantile values of col in each bin.
    errors : numpy.ndarray (if est_errors==True)
        Bootstrap-estimated standard error.
    """
    data = data.dropna(subset=col)
    if len(bin_edges) == 0:
        bin_edges = np.linspace(data[bin_col].min(), data[bin_col].max(), bins+1)
    bin_centers = get_bin_centers(bin_edges)
    grouped = data.groupby(pd.cut(data[bin_col], bin_edges), observed=False)[col]
    counts = grouped.count().values
    quantile = grouped.quantile(q).values
    nans = np.nan * np.ones(counts.shape)
    ret = (bin_centers, np.where(counts > min_count, quantile, nans))
    if est_errors:
        errors = grouped.apply(median_standard_error, B=nsamples, seed=seed).values
        ret = (
            bin_centers, 
            np.where(counts > min_count, quantile, nans), 
            np.where(counts > min_count, errors, nans)
        )
    return ret
    
    
def binned_medians(data, col, bin_col, bins=50, bin_edges=[], min_count=0):
    """
    Calculate median trends in bins of a second parameter.
    
    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame with at least two columns.
    col : str
        Data column corresponding to the first parameter, for which the
        intervals will be calculated in each bin.
    bin_col : str
        Data column corresponding to the second (binning) parameter.
    bins : int, optional
        The number of equal-size bins to divide the data along bin_col.
        The default is 50.
    bin_edges : array-like, optional
        Edges of bins for calculating the quantile. Will override the value
        of bins if provided.
    min_count : int, optional [default: 0]
        Minimum data count required to calculate a quantile. If there are fewer
        points in that bin, the quantile will be NaN.
    
    Returns
    -------
    bin_centers : numpy.ndarray
        Center of each bin in bin_col.
    medians : numpy.ndarray
        Median values of col in each bin.
    """
    return binned_quantiles(
        data, col, bin_col, 
        q=0.5, bins=bins, bin_edges=bin_edges, min_count=min_count
    )


def radial_gradient(multioutput, parameter, index=-1, 
                    Rmax=MAX_SF_RADIUS, zone_width=ZONE_WIDTH):
    """
    Return the value of the given model parameter at all zones.
    
    Parameters
    ----------
    multioutput : vice.multioutput
        VICE multi-zone output instance for the desired model.
    parameter : str
        Name of parameter in vice.history dataframe.
    index : int, optional
        Index to select for each zone. The default is -1, which corresponds
        to the last simulation timestep or the present day.
    Rmax : float, optional
        Maximum radius in kpc. The default is 15.5.
    zone_width : float, optional
        Annular zone width in kpc. The default is 0.1.
        
    Returns
    -------
    list
        Parameter values at each zone at the given time index.
    """
    return [multioutput.zones['zone%i' % z].history[index][parameter] 
            for z in range(int(Rmax/zone_width))]


def plot_gas_abundance(ax, mzs, xcol, ycol, label='', **kwargs):
    """
    Plot the ISM abundance tracks for the mean zone.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes on which to plot the gas abundance.
    mzs : MultizoneStars object
        Object containing model stars data. Gas abundance will be plotted from
        the mean radius of the data.
    xcol : str
        Column of data to plot on the x-axis.
    ycol : str
        Column of data to plot on the y-axis.
    label : str, optional
        Line label. The default is ''.
    **kwargs passed to matplotlib.pyplot.plot()

    Returns
    -------
    lines : list of Line2D
        Output of Axes.plot().
    """
    zone = int(0.5 * (mzs.galr_lim[0] + mzs.galr_lim[1]) / mzs.zone_width)
    zone_path = str(mzs.fullpath / ('zone%d' % zone))
    hist = vice.history(zone_path)
    lines = ax.plot(hist[xcol], hist[ycol], label=label, **kwargs)
    return lines
