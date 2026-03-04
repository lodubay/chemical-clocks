"""
This script plots a Hayden-style [Ce/Mg]-[Mg/H] plot of a multizone model
compared to MWM data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MultipleLocator
import vice

from multizone_stars import MultizoneStars
from plotting import insert_colorbar_axes, TWO_COLUMN_WIDTH
from utils import plot_gas_abundance
from stats import kde2D
import paths
from multizone._globals import MAX_SF_RADIUS, END_TIME

OUTPUT_NAME = 'insideout-mscale/diskmodel'

ZONE_WIDTH = 0.1 # kpc
GALR_BINS = [3, 5, 7, 9, 11, 13]
ABSZ_BINS = [0, 0.5, 1, 2]


def main(style='paper', cmap='Spectral_r'):
    # Import MWM sample
    sample = pd.read_csv(paths.data / 'MWM' / 'sample.csv')
    # Set up figure
    plt.style.use(paths.styles / f'{style}.mplstyle')
    rows = len(ABSZ_BINS) - 1
    cols = len(GALR_BINS) - 1
    width = TWO_COLUMN_WIDTH
    fig, axs = plt.subplots(
        rows, cols, 
        figsize=(width, (width/cols)*rows),
        sharex=True, sharey=True,
        gridspec_kw={'hspace': 0., 'wspace': 0.}
    )
    cax = insert_colorbar_axes(fig)
    age_bounds = np.arange(0, 12.1, 2)
    cmap = plt.get_cmap(cmap)
    cbar = fig.colorbar(
        ScalarMappable(
            BoundaryNorm(age_bounds, cmap.N, extend='max'), 
            cmap
        ), 
        cax=cax, 
        label='Age [Gyr]'
    )
    # Plot multizone output
    mzs = MultizoneStars.from_output(OUTPUT_NAME)
    for i, row in enumerate(axs):
        zlim = (ABSZ_BINS[-(i+2)], ABSZ_BINS[-(i+1)])
        for j, ax in enumerate(row):
            rlim = (GALR_BINS[j], GALR_BINS[j+1])
            mzs_subset = mzs.region(rlim, zlim)
            mzs_subset.scatter_plot(ax, '[mg/h]', '[ce/mg]', color='age',
                                cmap=cbar.cmap, norm=cbar.norm)
            plot_gas_abundance(ax, mzs_subset, '[mg/h]', '[ce/mg]', c='k', ls='--')
            # Plot MWM data contours
            sample_subset = sample[
                (sample['Rg'] >= rlim[0]) &
                (sample['Rg'] < rlim[1]) &
                (sample['z_max'] >= zlim[0]) &
                (sample['z_max'] < zlim[1])
            ]
            plot_kde2D_contours(ax, sample_subset, 'mg_h', 'ce_mg')
    
    # Format axes
    axs[0,0].set_xlim((-0.8, 0.6))
    axs[0,0].set_ylim((-0.8, 0.8))
    # Label bins in z-height
    row_label_pos = (0.5, 0.95)
    for i, ax in enumerate(axs[:,0]):
        absz_lim = (ABSZ_BINS[-(i+2)], ABSZ_BINS[-(i+1)])
        ax.text(row_label_pos[0], row_label_pos[1], 
                r'$%s\leq |z| < %s$ kpc' % absz_lim,
                transform=ax.transAxes, ha='center', va='top')
    # Label bins in Rgal
    for i, ax in enumerate(axs[0]):
        galr_lim = (GALR_BINS[i], GALR_BINS[i+1])
        ax.set_title(
            r'$%s\leq R_{\rm{gal}} < %s$ kpc' % galr_lim, 
            size=plt.rcParams['font.size']
        )
    for ax in axs[-1]:
        ax.set_xlabel('[Mg/H]')
    for ax in axs[:,0]:
        ax.set_ylabel('[Ce/Mg]')
    # Set x-axis ticks
    axs[0,0].xaxis.set_major_locator(MultipleLocator(0.5))
    axs[0,0].xaxis.set_minor_locator(MultipleLocator(0.1))
    # Set y-axis ticks
    axs[0,0].yaxis.set_major_locator(MultipleLocator(0.5))
    axs[0,0].yaxis.set_minor_locator(MultipleLocator(0.1))
    
    plt.savefig(paths.figures / 'model_cemg_mgh.pdf')
    plt.close()


def plot_kde2D_contours(ax, data, xcol, ycol, enclosed=[0.9, 0.7, 0.5, 0.3, 0.1],
                        c='k', lw=0.5, ls='-',
                        plot_kwargs={}, **kwargs):
    """
    Plot 2D density contours from the kernel density estimate for the
    given columns.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object on which to draw the scatter plot.
    data : pandas.DataFrame
        DataFrame of MWM data.
    xcol : str
        Name of column to plot on the x-axis.
    ycol : str
        Name of column to plot on the y-axis.
    enclosed : list, optional
        List of probabilities enclosed by the contour levels, ordered
        from highest probability (lowest contour level) to lowest (highest).
        The default is [0.8, 0.3].
    c : str or matplotlib color or list of previous, optional
        Color(s) of each contour line. The default is 'r'.
    lw : float or list of floats, optional
        Line widths corresponding to each contour line. The default is 0.5.
    ls : str or list of str, optional
        Line styles of each contour. If a list, length must be equal
        to the length of 'enclosed'. The default is ['--', '-'].
    **kwargs passed to `kde2D()`.
    """
    xx, yy, logz = get_kde2D(data, xcol, ycol, **kwargs)
    # scale the linear density to the max value
    scaled_density = np.exp(logz) / np.max(np.exp(logz))
    # contour levels at 1 and 2 sigma
    levels = contour_levels_2D(scaled_density, enclosed=enclosed)
    ax.contour(xx, yy, scaled_density, levels, colors=c,
                linewidths=lw, linestyles=ls, **plot_kwargs)


def contour_levels_2D(arr2d, enclosed=[0.8, 0.3]):
    """
    Calculate the contour levels which contain the given enclosed probabilities.
    
    Parameters
    ----------
    arr2d : np.ndarray
        2-dimensional array of densities.
    enclosed : list, optional
        List of enclosed probabilities of the contour levels. The default is
        [0.8, 0.3].
    """
    levels = []
    l = 0.
    i = 0
    while l < 1 and i < len(enclosed):
        frac_enclosed = np.sum(arr2d[arr2d > l]) / np.sum(arr2d)
        if frac_enclosed <= enclosed[i] + 0.01:
            levels.append(l)
            i += 1
        l += 0.01
    return levels


def get_kde2D(data, xcol, ycol, bandwidth=0.03, overwrite=False, **kwargs):
    """
    Generate 2-dimensional kernel density estimate (KDE) of APOGEE data, 
    or import previously saved KDE if it already exists.
    
    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame of MWM data.
    xcol : str
        Name of column with x-axis data
    ycol : str
        Name of column with y-axis data
    bandwidth : float
        Kernel density estimate bandwidth. A larger number will produce
        smoother contour lines. The default is 0.03.
    overwrite : bool
        If True, force re-generate the 2D KDE and save the output.
    **kwargs passed to stats.kde2D()
    
    Returns
    -------
    xx, yy, logz: tuple of numpy.array
        Outputs of stats.kde2D()
    """    
    # Path to save 2D KDE for faster plot times
    rlim = (round(data['Rg'].min(), 1), round(data['Rg'].max(), 1))
    zlim = (round(data['z_max'].min(), 1), round(data['z_max'].max(), 1))
    path = kde2D_path(xcol, ycol, rlim, zlim)
    if path.exists() and not overwrite:
        xx, yy, logz = read_kde(path)
    else:
        xx, yy, logz = kde2D(data[xcol], data[ycol], bandwidth, **kwargs)
        save_kde(xx, yy, logz, path)
    return xx, yy, logz


def read_kde(path):
    """
    Read a text file generated by save_kde()
    """
    arr2d = np.genfromtxt(path)
    nrows = int(arr2d.shape[0]/3)
    xx = arr2d[:nrows]
    yy = arr2d[nrows:2*nrows]
    logz = arr2d[2*nrows:]
    return xx, yy, logz


def save_kde(xx, yy, logz, path):
    """
    Generate a text file containing the KDE of the given region along with its
    corresponding x and y coordinates.
    """
    if not path.parents[0].is_dir():
        path.parents[0].mkdir(parents=True)
    with open(path, 'w') as f:
        for arr in [xx, yy, logz]:
            f.write('#\n')
            np.savetxt(f, arr)


def kde2D_path(xcol, ycol, galr_lim, absz_lim):
    """
    Generate file name for the KDE of the given region.
    
    Parameters
    ---------
    xcol : str
        Name of column with x-axis data
    ycol : str
        Name of column with y-axis data
    """
    kde_dir = '_'.join([''.join(xcol.split('_')).lower(),
                        ''.join(ycol.split('_')).lower()])
    filename = 'r%s-%s_z%s-%s.dat' % (galr_lim + absz_lim)
    return paths.data / 'MWM' / 'kde' / kde_dir / filename


if __name__ == '__main__':
    main()
