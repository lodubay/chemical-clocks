"""
Functions for plotting.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import PathCollection
from matplotlib.lines import Line2D

# AASTeX plot widths in inches
ONE_COLUMN_WIDTH = 3.25
TWO_COLUMN_WIDTH = 7.

# Default colormaps
DENSITY_COLORMAP = 'gist_heat_r'
AGE_COLORMAP = 'Spectral_r'
RADIUS_COLORMAP = 'managua'
ABUNDANCE_COLORMAP = 'viridis'

# Hayden plot bins
RBINS = [(3, 5), (5, 7), (7, 9), (9, 11), (11, 13)] # left to right
ZBINS = [(1, 2), (0.5, 1), (0, 0.5)] # top to bottom


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
    cmap : str or matplotlib colormap instance
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
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
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


def colored_text_legend(ax, show_handles=False, invert=False, **kwargs):
    """
    Make a text-only legend with color-coding.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    show_handles : bool [default: False]
        If True, show legend handles while still changing text color
    invert : bool [default: False]
        If True, invert the order of the legend entries.
    kwargs passed to plt.legend()

    Returns
    -------
    leg : matplotlib.legend.Legend
    """
    handles, labels = ax.get_legend_handles_labels()
    if invert:
        handles = handles[::-1]
        labels = labels[::-1]
    # Remove legend handles
    if show_handles:
        leg = ax.legend(handles, labels, **kwargs)
    else:
        leg = ax.legend(handles, labels, handlelength=0, handletextpad=0, markerscale=0, **kwargs)
        for line in leg.get_lines():
            line.set_visible(False)
    # Color-code legend text by line and point colors
    for handle, text in zip(handles, leg.get_texts()):
        if isinstance(handle, PathCollection):
            text.set_color(handle.get_facecolor()[0])
        elif isinstance(handle, Line2D):
            text.set_color(handle.get_color())
    return leg


def setup_hayden_plot(
        rbins=RBINS, 
        zbins=ZBINS, 
        width=TWO_COLUMN_WIDTH, 
        row_label=r'z_{\rm max}',
        col_label='R_g',
        labelsize=None,
    ):
    """
    Set up a Hayden-style grid of subplots by radius and midplane distance.

    Parameters
    ----------
    rbins : list of tuples, optional
        List of bounds on the radius in kpc for each column, from left to right.
        The default is [(3, 5), (5, 7), (7, 9), (9, 11), (11, 13)].
    zbins : list of tuples, optional
        List of bounds on the midplane distance in kpc for each row, from top
        to bottom. The default is [(1, 2), (0.5, 1), (0, 0.5)].
    width : float, optional
        Figure width in inches. The default is 7.5.
    row_label : str, optional
        Midplane distance variable (assumes LaTeX). The default is 'z_{\rm max}'.
    col_label : str, optional
        Galactocentric radius variable (assumes LaTeX). The default is 'R_g'.
    labelsize : int, optional
        Font size for row and column labels. If None, the default text size
        is used. Default is None.
    """
    fig, axs = plt.subplots(
        len(zbins), len(rbins),
        figsize=(width, (width / len(rbins)) * len(zbins)),
        sharex=True, sharey=True, 
        gridspec_kw={'hspace': 0, 'wspace': 0}
    )
    # Row and column labels
    if labelsize is None:
        labelsize = plt.rcParams['font.size']
    for i, ax in enumerate(axs[0,:]):
        ax.set_title(
            r'$%s\leq %s<%s$ kpc' % (rbins[i][0], col_label, rbins[i][1]), 
            fontsize=labelsize
        )
    for i, ax in enumerate(axs[:,-1]):
        ax.yaxis.set_label_position('right')
        ax.set_ylabel(
            r'$%s\leq %s<%s$ kpc' % (zbins[i][0], row_label, zbins[i][1]), 
            fontsize=labelsize, labelpad=6
        )
    return fig, axs


def iterate_rz_bins(rbins=RBINS, zbins=ZBINS):
    """
    Iterate through a 2D grid of radial and vertical coordinate bounds.

    Returns
    -------
    i : int
        Row-index (0 is top)
    j : int
        Column-index (0 is left)
    zlim : tuple of floats
        Lower and upper limits on midplane distance in kpc.
    rlim : tuple of floats
        Lower and upper limits on radius in kpc.
    """
    # 2D grid of indices
    ii = np.tile(range(len(zbins)), (len(rbins), 1)).T
    jj = np.tile(range(len(rbins)), (len(zbins), 1))
    zflat = [zbins[i] for i in ii.flatten()]
    rflat = [rbins[j] for j in jj.flatten()]
    return zip(ii.flatten(), jj.flatten(), zflat, rflat)
