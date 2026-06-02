"""
This script plots the slope of the age-[Ce/Mg] trend as a function of 
metallicity for each region of the Galaxy.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import BoundaryNorm
from scipy import stats

from utils import get_bin_centers
from plotting import ONE_COLUMN_WIDTH, colored_text_legend, RADIUS_COLORMAP
import paths

RBINS = [(3, 5), (5, 7), (7, 9), (9, 11), (11, 13), (13, 15)]
ZBINS = [(1, 2), (0.5, 1), (0, 0.5)] # top to bottom
MET_COL = 'fe_h' # Column with metallicity values
MET_LABEL = '[Fe/H]'
AGE_FIT_RANGE = (1, 8) # Range of ages to fit linear trend
MIN_COUNT = 20 # Minimum number of stars in each bin for trend fitting
AGE_DELTA = 5 # Gyr, linear age shift for regression
SOLAR_AGE = 4.6 # Gyr

def main(style='paper', cmap=RADIUS_COLORMAP):
    plt.style.use(paths.styles / f'{style}.mplstyle')

    # Metallicity bins
    met_bin_edges = np.arange(-0.85, 0.56, 0.1)
    dr = RBINS[0][1] - RBINS[0][0]
    radius_bin_edges = np.arange(RBINS[0][0], RBINS[-1][1]+dr, dr)

    # Import MWM sample
    mwm_rgb = pd.read_csv(paths.data / 'MWM' / 'sample.csv')
    mwm_rgb = mwm_rgb[mwm_rgb['good_age']].copy()
    local_sample = mwm_rgb[
        (mwm_rgb['Rg'] >= 7) &
        (mwm_rgb['Rg'] < 9) &
        (mwm_rgb['z_max'] < 0.5) &
        (mwm_rgb['low_alpha']) # restrict age trends to low-alpha only
    ]
    local_fits, local_mets = fit_metallicity_bins(local_sample, met_bin_edges)
    local_slopes = [f.slope for f in local_fits]

    # Set up figure
    fig, axs = plt.subplots(
        len(ZBINS), 1,
        figsize=(ONE_COLUMN_WIDTH, 2*ONE_COLUMN_WIDTH), 
        sharex=True, sharey=True,
        gridspec_kw={'hspace': 0, 'wspace': 0},
    )
    cmap = plt.get_cmap(cmap)
    norm = BoundaryNorm(radius_bin_edges, cmap.N)
    # plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95)

    for i, ax in enumerate(axs):
        zlim = ZBINS[i]
        for j, rlim in enumerate(RBINS):
            mean_radius = np.mean(rlim)
            region = mwm_rgb[
                (mwm_rgb['Rg'] >= rlim[0]) &
                (mwm_rgb['Rg'] < rlim[1]) &
                (mwm_rgb['z_max'] >= zlim[0]) &
                (mwm_rgb['z_max'] < zlim[1]) &
                (mwm_rgb['low_alpha']) # restrict age trends to low-alpha only
            ]
            # Bin by metallicity and fit linear trend to stars
            region_fits, mets = fit_metallicity_bins(region, met_bin_edges)
            slopes = np.array([f.slope for f in region_fits])
            errors = np.array([f.stderr for f in region_fits])
            # Plot fit slopes
            if mean_radius == 8:
                color = 'k'
            else:
                color = cmap(norm(mean_radius))
            ax.plot(
                mets, slopes, '.-',
                color=color,
                label=f'{int(mean_radius)} kpc'
            )
            ax.fill_between(
                mets, slopes - errors, slopes + errors,
                color=color, 
                alpha=0.5, edgecolor='none'
            )
            # ax.errorbar(
            #     mets, slopes, 
            #     yerr=errors, 
            #     marker='o', c='k', linestyle='none', ms=3, capsize=0
            # )
        # Plot Solar neighborhood fits for comparison
        if i<2:
            ax.plot(local_mets, local_slopes, 'k--')
        # Dotted horizontal line
        ax.axhline(0, ls=':', c='gray', zorder=0)

    # Format axes
    axs[0].set_xlim((-0.7, 0.6))
    axs[0].set_ylim((-0.08, 0.08))
    axs[0].xaxis.set_major_locator(MultipleLocator(0.5))
    axs[0].xaxis.set_minor_locator(MultipleLocator(0.1))
    axs[0].yaxis.set_major_locator(MultipleLocator(0.05))
    axs[0].yaxis.set_minor_locator(MultipleLocator(0.01))
    axs[-1].set_xlabel(MET_LABEL)
    for i, ax in enumerate(axs):
        ax.set_title(r'$%s\leq z_{\rm max}<%s$ kpc' % ZBINS[i], y=0.83)
        ax.set_ylabel('Slope [dex/Gyr]')
    colored_text_legend(axs[0], loc='center right', frameon=True)

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
