"""
Functions for plotting.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

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
