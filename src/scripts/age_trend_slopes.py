"""
This script plots the slope of the age-[Ce/Mg] trend as a function of 
metallicity for each region of the Galaxy.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy import stats

from utils import get_bin_centers
from _globals import TWO_COLUMN_WIDTH
import paths

RBINS = [(3, 5), (5, 7), (7, 9), (9, 11), (11, 13)] # left to right
ZBINS = [(1, 2), (0.5, 1), (0, 0.5)] # top to bottom
MET_COL = 'm_h_atm' # Column with metallicity values
MET_LABEL = r'[M/H]$_{\rm atm}$'
AGE_FIT_RANGE = (1, 8) # Range of ages to fit linear trend
MIN_COUNT = 20 # Minimum number of stars in each bin for trend fitting

def main(style='paper'):
    plt.style.use(paths.styles / f'{style}.mplstyle')

    # Metallicity bins
    met_bin_edges = np.arange(-0.85, 0.56, 0.1)

    # Import MWM sample
    mwm_rgb = pd.read_csv(paths.data / 'MWM' / 'MWM_RGB.csv')
    local_sample = mwm_rgb[
        (mwm_rgb['Rg'] >= 7) &
        (mwm_rgb['Rg'] < 9) &
        (mwm_rgb['z_max'] < 0.5)
    ]
    local_fits, local_mets = fit_metallicity_bins(local_sample, met_bin_edges)
    local_slopes = [f.slope for f in local_fits]

    # Set up figure
    fig, axs = plt.subplots(
        len(ZBINS), len(RBINS),
        figsize=(TWO_COLUMN_WIDTH, 0.6*TWO_COLUMN_WIDTH), 
        sharex=True, sharey=True,
        gridspec_kw={'hspace': 0, 'wspace': 0},
    )
    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95)
    # scatterplot style arguments
    kwargs = dict(s=1, marker='.', rasterized=True, edgecolor='none')

    for i, row in enumerate(axs):
        zlim = ZBINS[i]
        for j, ax in enumerate(row):
            rlim = RBINS[j]
            region = mwm_rgb[
                (mwm_rgb['Rg'] >= rlim[0]) &
                (mwm_rgb['Rg'] < rlim[1]) &
                (mwm_rgb['z_max'] >= zlim[0]) &
                (mwm_rgb['z_max'] < zlim[1])
            ]
            # Bin by metallicity and fit linear trend to stars
            region_fits, mets = fit_metallicity_bins(region, met_bin_edges)
            slopes = [f.slope for f in region_fits]
            errors = [f.stderr for f in region_fits]
            # Plot fit slopes
            ax.errorbar(
                mets, slopes, 
                yerr=errors, 
                marker='o', c='k', linestyle='none', ms=3, capsize=0
            )
            # Plot Solar neighborhood fits for comparison
            ax.plot(local_mets, local_slopes, 'k-')

    # Format axes
    axs[0,0].set_xlim((-0.9, 0.6))
    axs[0,0].set_ylim((-0.12, 0.12))
    axs[0,0].xaxis.set_major_locator(MultipleLocator(0.5))
    axs[0,0].xaxis.set_minor_locator(MultipleLocator(0.1))
    axs[0,0].yaxis.set_major_locator(MultipleLocator(0.05))
    axs[0,0].yaxis.set_minor_locator(MultipleLocator(0.01))
    for ax in axs[-1,:]:
        ax.set_xlabel(MET_LABEL)
    for i, ax in enumerate(axs[0,:]):
        ax.set_title(r'$%s\leq R_{\rm guide}<%s$ kpc' % RBINS[i], fontsize=8)
    for ax in axs[:,0]:
        ax.set_ylabel('Slope [dex/Gyr]')
    for i, ax in enumerate(axs[:,-1]):
        ax.yaxis.set_label_position('right')
        ax.set_ylabel(
            r'$%s\leq z_{\rm max}<%s$ kpc' % ZBINS[i], 
            fontsize=8, labelpad=6
        )

    plt.savefig(paths.figures / 'age_trend_slopes')


def fit_metallicity_bins(
        data, 
        bins, 
        min_count=MIN_COUNT, 
        xcol='age', 
        ycol='ce_mg', 
        met_col=MET_COL,
        age_fit_range=AGE_FIT_RANGE,
        **kwargs
    ):
    """
    Bin data by metallicity and fit a linear trend in each bin.
    
    Parameters
    ----------
    data : pandas.DataFrame
    bins : array-like
        Metallicity bin edges
    min_count : int, optional [default: 10]
        Minimum number of stars in a bin required to calculate a fit.
    xcol : str, optional [default: 'age']
        Column for the independent fit variable.
    ycol : str, optional [default: 'ce_mg']
        Column for the dependent fit variable.
    met_col : str, optional [default: 'm_h_atm']
        Column for the binning variable.
    age_fit_range : tuple of floats, optional [default: (1, 8)]
        Range of ages considered valid for fit procedure. Data outside this
        range will not contribute to the fit.
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
            (data[MET_COL] >= met_lim[0]) & 
            (data[MET_COL] < met_lim[1]) &
            (data[xcol] >= age_fit_range[0]) &
            (data[xcol] < age_fit_range[1])
        ]
        if subset.shape[0] > min_count:
            # Fit linear age trend
            regress = stats.linregress(subset[xcol], subset[ycol], **kwargs)
            fits.append(regress)
            bin_centers.append(np.mean(met_lim))
    return fits, bin_centers


if __name__ == '__main__':
    main()
