"""
Utility functions and classes for many scripts.
"""

import numpy as np
from numpy.random import default_rng
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from astropy.table import Table
import vice

from _globals import RANDOM_SEED

# =============================================================================
# SCIENCE FUNCTIONS
# =============================================================================

def good_ages(df):
    """
    Perform quality cuts for good StarFlow ages.
    """
    return df[
        (df['training_density'] > 3e9) & # Stone-Martinez et al. (2025) recommendation
        (df['age'] > 0)
    ].copy()


def apply_alpha_cut(df, buffer=0.02):
    """
    Divide sample into high- and low-alpha populations.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with MWM sample.
    buffer : float [default: 0.02]
        Buffer in [Mg/Fe] between the dividing line and the start of the 
        high- or low-alpha populations.

    Returns
    -------
    pandas.DataFrame
        Same dataframe with two new boolean columns, 'high_alpha' and 'low_alpha'.
    """
    df['low_alpha'] = df['mg_fe'] < alpha_cut(df['fe_h']) - buffer
    df['high_alpha'] = df['mg_fe'] > alpha_cut(df['fe_h']) + buffer
    return df


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


class amplified_agb(vice.yields.agb.interpolator): 
    """
    Amplify the AGB yields by a multiplicative factor.

    Inherits from vice.yields.agb.interpolator.
    """
    def __init__(self, element, study = 'cristallo11', prefactor=3):
        self.prefactor = prefactor
        super().__init__(element, study=study)
    
    def __call__(self, mass, metallicity): 
        return self.prefactor * super().__call__(mass, metallicity) 


# =============================================================================
# DATA UTILITY FUNCTIONS
# =============================================================================

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
    
    
def binned_quantiles(data, col, bin_col, q=0.5, bins=50, bin_edges=[], min_count=0):
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
    
    Returns
    -------
    bin_centers : numpy.ndarray
        Center of each bin in bin_col.
    quantiles : numpy.ndarray
        Quantile values of col in each bin.
    """
    data = data.dropna(subset=col)
    if len(bin_edges) == 0:
        bin_edges = np.linspace(data[bin_col].min(), data[bin_col].max(), bins+1)
    bin_centers = get_bin_centers(bin_edges)
    grouped = data.groupby(pd.cut(data[bin_col], bin_edges), observed=False)[col]
    counts = grouped.count().values
    quantile = grouped.quantile(q).values
    nans = np.nan * np.ones(counts.shape)
    return bin_centers, np.where(counts > min_count, quantile, nans)
    
    
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

# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def get_color_list(cmap, bins):
    """
    Split a discrete colormap into a list of colors based on bin edges.
    
    Parameters
    ----------
    cmap : matplotlib colormap
    bins : array-like
        Bin edges, including left- and right-most edges
    
    Returns
    -------
    list
        List of colors of length len(bins) - 1
    """
    rmin, rmax = bins[0], bins[-2]
    colors = cmap([(r-rmin)/(rmax-rmin) for r in bins[:-1]])
    return colors


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    """
    Truncate an existing colormap.

    Parameters
    ----------
    cmap : matplotlib colormap instance
    minval : float, optional
        Lower truncation bound, between 0 and 1. Default is 0.
    maxval : float, optional
        Upper truncation bound, between 0 and 1. Default is 1.
    n : int, optional
        Number of segments in the new colormap. Default is 100.
    
    Returns
    -------
    new_cmap : matplotlib.colors.LinearSegmentedColormap
        New, truncated colormap.
    """
    new_cmap = LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def latex_float(f):
    """
    Convert exponential float to LaTeX string.
    """
    float_str = '{0:.2g}'.format(f)
    if 'e' in float_str:
        base, exponent = float_str.split('e')
        return r'${0} \times 10^{{{1}}}$'.format(base, int(exponent))
    else:
        return float_str
    

def insert_colorbar_axes(fig, orientation='vertical', width=0.02, pad=0.01):
    """
    Insert a new Axes object for a colorbar in a multi-panel figure.

    Parameters
    ----------
    fig : matplotlib.figure.Figure instance
        Figure to add the colorbar to.
    orientation : str, optional [default: 'vertical']
        Orientation for the colorbar. If 'vertical', space will be taken from
        the right side of the figure. If 'horizontal', space will be taken
        from the bottom.
    width : float, optional [default: 0.02]
        Width of the colorbar as a fraction of the total figure width.
    pad : float, optional [default: 0.01]
        Padding between existing axes and colorbar.

    Returns
    -------
    cax : matplotlib.axes.Axes instance
        New Axes object for colorbar.
    """
    if orientation == 'horizontal':
        # Define colorbar axis
        height = fig.subplotpars.right - fig.subplotpars.left
        cax = plt.axes([fig.subplotpars.left, fig.subplotpars.bottom, 
                        height, width])
        # Adjust subplots
        plt.subplots_adjust(bottom=fig.subplotpars.bottom + (width + pad + 0.03))
    else:
        # Adjust subplots
        plt.subplots_adjust(right=fig.subplotpars.right - (width + pad + 0.03))
        # Define colorbar axis
        height = fig.subplotpars.top - fig.subplotpars.bottom
        cax = plt.axes([fig.subplotpars.right + pad, fig.subplotpars.bottom, 
                        width, height])
    return cax


def colored_text_legend(ax, **kwargs):
    """
    Make a text-only legend with color-coding.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    kwargs passed to plt.legend()

    Returns
    -------
    leg : matplotlib.legend.Legend
    """
    # Remove legend handles
    leg = ax.legend(handlelength=0, handletextpad=0, **kwargs)
    for line in leg.get_lines():
        line.set_visible(False)
    # Color-code legend text
    for line, text in zip(leg.get_lines(), leg.get_texts()):
        text.set_color(line.get_color())
    return leg
