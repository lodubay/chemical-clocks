"""
This script plots the slope of the age-[Ce/Mg] trend as a function of 
metallicity for each region of the Galaxy.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import ScalarMappable
from scipy import stats

from utils import get_bin_centers
from plotting import ONE_COLUMN_WIDTH, colored_text_legend, RADIUS_COLORMAP, insert_colorbar_axes
import paths

ZLIM = (0, 0.5) # global z_max limits
MET_COL = 'fe_h' # Column with metallicity values
MET_LABEL = '[Fe/H]'
AGE_FIT_RANGE = (1, 8) # Range of ages to fit linear trend
MIN_COUNT = 20 # Minimum number of stars in each bin for trend fitting
AGE_DELTA = 5 # Gyr, linear age shift for regression


def main(style='paper', cmap=RADIUS_COLORMAP, zlim=ZLIM):
    plt.style.use(paths.styles / f'{style}.mplstyle')

    # Metallicity bins
    met_bin_edges = np.arange(-0.85, 0.56, 0.1)
    dr = 2
    radius_bin_edges = np.arange(3, 15+dr, dr)

    # Import MWM sample
    mwm_sample = pd.read_csv(paths.data / 'sample.csv')

    # Set up figure
    fig, axs = plt.subplots(
        2,
        figsize=(ONE_COLUMN_WIDTH, 1.5*ONE_COLUMN_WIDTH), 
        sharex=True,
        gridspec_kw={'hspace': 0}
    )
    cmap = plt.get_cmap(cmap)
    norm = BoundaryNorm(radius_bin_edges, cmap.N)
    # plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95)
    cax = insert_colorbar_axes(fig, orientation='horizontal', pad=0.05)

    for j in range(len(radius_bin_edges)-1):
        rlim = radius_bin_edges[j:j+2]
        mean_radius = np.mean(rlim)
        color = cmap(norm(mean_radius))
        region = mwm_sample[
            (mwm_sample['Rg'] >= rlim[0]) &
            (mwm_sample['Rg'] < rlim[1]) &
            (mwm_sample['z_max'] >= zlim[0]) &
            (mwm_sample['z_max'] < zlim[1]) &
            (mwm_sample['good_age']) # limit to good ages
            (mwm_sample['low_alpha']) # restrict age trends to low-alpha only
        ]
        # Bin by metallicity and fit linear trend to stars
        region_fits, mets = fit_metallicity_bins(region, met_bin_edges)
        # Plot fit slopes
        slopes = np.array([f.slope for f in region_fits])
        errors = np.array([f.stderr for f in region_fits])
        axs[0].plot(
            mets, slopes, '.-',
            color=color,
            label=f'{int(mean_radius)} kpc'
        )
        axs[0].fill_between(
            mets, slopes - errors, slopes + errors,
            color=color, 
            alpha=0.5, 
            edgecolor='none'
        )
        # Plot intercepts
        intercepts = np.array([f.intercept for f in region_fits])
        int_errs = np.array([f.intercept_stderr for f in region_fits])
        axs[1].plot(mets, intercepts, '.-', color=color)
        axs[1].fill_between(
            mets, intercepts - int_errs, intercepts + int_errs,
            color=color,
            alpha=0.5,
            edgecolor='none'
        )
    # Dotted horizontal line at 0
    axs[0].axhline(0, ls=':', c='gray', zorder=0)
    # indicate Solar value
    axs[1].plot(0, 0, 'wo', zorder=9)
    axs[1].text(
        0, 0, r'$\odot$',
        va='center', ha='center', zorder=10, weight='bold', usetex=True
    )

    # Colorbar
    fig.colorbar(
        ScalarMappable(norm, cmap), 
        cax=cax, 
        orientation='horizontal', 
        label='Guiding radius [kpc]'
    )

    # Format axes
    axs[0].set_xlim((-0.7, 0.6))
    axs[0].set_ylim((-0.1, 0.05))
    axs[0].xaxis.set_major_locator(MultipleLocator(0.5))
    axs[0].xaxis.set_minor_locator(MultipleLocator(0.1))
    axs[0].yaxis.set_major_locator(MultipleLocator(0.05))
    axs[0].yaxis.set_minor_locator(MultipleLocator(0.01))
    axs[0].set_ylabel('Slope [dex/Gyr]')
    axs[1].set_ylim((-0.12, 0.38))
    axs[1].yaxis.set_major_locator(MultipleLocator(0.1))
    axs[1].yaxis.set_minor_locator(MultipleLocator(0.02))
    axs[1].set_ylabel(r'[Ce/Mg] at $\tau=5$ Gyr')
    axs[1].set_xlabel(MET_LABEL)
    # colored_text_legend(ax, loc='center right', frameon=True)

    plt.savefig(paths.figures / 'global_metallicity_fits')


def fit_metallicity_bins(
        data, 
        bins, 
        min_count=MIN_COUNT, 
        xcol='age', 
        ycol='ce_mg_corr', 
        met_col=MET_COL,
        age_fit_range=AGE_FIT_RANGE,
        age_delta=AGE_DELTA,
        **kwargs
    ):
    """
    Bin data by metallicity and fit a linear trend in each bin.
    
    Parameters
    ----------
    data : pandas.DataFrame
    bins : array-like
        Metallicity bin edges
    min_count : int, optional [default: 20]
        Minimum number of stars in a bin required to calculate a fit.
    xcol : str, optional [default: 'age']
        Column for the independent fit variable.
    ycol : str, optional [default: 'ce_mg_corr']
        Column for the dependent fit variable.
    met_col : str, optional [default: 'fe_h']
        Column for the binning variable.
    age_fit_range : tuple of floats, optional [default: (1, 8)]
        Range of ages considered valid for fit procedure. Data outside this
        range will not contribute to the fit.
    age_delta : float, optional [default: 5]
        Linear age shift in Gyr to center regression
    **kwargs passed to scipy.stats.linregress
    
    Returns
    -------
    fits : list of LinregressResult instances
        List of linear regression fits for each metallicity bin.
    bin_centers : list
        Mean of each metallicity bin for which a fit was performed.
    """
    fits = []
    bin_centers = []
    for k in range(len(bins)-1):
        met_lim = bins[k:k+2]
        subset = data[
            (data[met_col] >= met_lim[0]) & 
            (data[met_col] < met_lim[1]) &
            (data[xcol] >= age_fit_range[0]) &
            (data[xcol] < age_fit_range[1])
        ]
        if subset.shape[0] > min_count:
            # Fit linear age trend
            regress = stats.linregress(
                subset[xcol] - age_delta, subset[ycol], **kwargs
            )
            fits.append(regress)
            bin_centers.append(np.mean(met_lim))
    return fits, bin_centers


if __name__ == '__main__':
    main()
