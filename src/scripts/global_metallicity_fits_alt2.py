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
from plotting import ONE_COLUMN_WIDTH, colored_text_legend, ABUNDANCE_COLORMAP, insert_colorbar_axes
import paths

RBINS = [(3, 5), (5, 7), (7, 9), (9, 11), (11, 13), (13, 15)]
ZBINS = [(1, 2), (0.5, 1), (0, 0.5)] # top to bottom
MET_COL = 'fe_h' # Column with metallicity values
MET_LABEL = '[Fe/H]'
AGE_FIT_RANGE = (1, 8) # Range of ages to fit linear trend
MIN_COUNT = 20 # Minimum number of stars in each bin for trend fitting
AGE_DELTA = 5 # Gyr, linear age shift for regression
SOLAR_AGE = 4.6 # Gyr

def main(style='paper', cmap=ABUNDANCE_COLORMAP):
    plt.style.use(paths.styles / f'{style}.mplstyle')

    # Metallicity bins
    met_bin_edges = np.arange(-0.5, 0.51, 0.2)
    dr = 1
    radius_bin_edges = np.arange(2.5, 15.5+dr, dr)

    # Import MWM sample
    mwm_rgb = pd.read_csv(paths.data / 'sample.csv')
    mwm_rgb = mwm_rgb[mwm_rgb['good_age']].copy()
    solar_sample = mwm_rgb[ # Solar metallicity sample
        (mwm_rgb[MET_COL] >= -0.1) &
        (mwm_rgb[MET_COL] < 0.1) &
        (mwm_rgb['z_max'] < 0.5) &
        (mwm_rgb['low_alpha']) # restrict age trends to low-alpha only
    ]
    solar_fits, solar_mets = fit_radial_bins(solar_sample, radius_bin_edges)
    solar_slopes = [f.slope for f in solar_fits]

    # Set up figure
    fig, axs = plt.subplots(
        len(ZBINS), 1,
        figsize=(ONE_COLUMN_WIDTH, 2*ONE_COLUMN_WIDTH), 
        sharex=True, sharey=True,
        gridspec_kw={'hspace': 0, 'wspace': 0},
    )
    cmap = plt.get_cmap(cmap)
    norm = BoundaryNorm(met_bin_edges, cmap.N)
    cax = insert_colorbar_axes(fig, orientation='horizontal', pad=0.03)
    # plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95)

    for i, ax in enumerate(axs):
        zlim = ZBINS[i]
        for j in range(len(met_bin_edges)-1):
            met_lim = met_bin_edges[j:j+2]
            mean_met = np.mean(met_lim)
            region = mwm_rgb[
                (mwm_rgb[MET_COL] >= met_lim[0]) &
                (mwm_rgb[MET_COL] < met_lim[1]) &
                (mwm_rgb['z_max'] >= zlim[0]) &
                (mwm_rgb['z_max'] < zlim[1]) &
                (mwm_rgb['low_alpha']) # restrict age trends to low-alpha only
            ]
            # Bin by metallicity and fit linear trend to stars
            met_fits, radii = fit_radial_bins(region, radius_bin_edges)
            slopes = np.array([f.slope for f in met_fits])
            errors = np.array([f.stderr for f in met_fits])
            # Plot fit slopes
            color = cmap(norm(mean_met))
            ax.plot(
                radii, slopes, '.-',
                color=color,
                label=str(round(mean_met, 1))
            )
            ax.fill_between(
                radii, slopes - errors, slopes + errors,
                color=color, 
                alpha=0.5, 
                edgecolor='none'
            )
        # Plot Solar neighborhood fits for comparison
        if i<2:
            ax.plot(solar_mets, solar_slopes, ls='--', color=cmap(norm(0)))
        # Dotted horizontal line
        ax.axhline(0, ls=':', c='gray', zorder=0)
    
    # Add colorbar
    fig.colorbar(
        ScalarMappable(norm, cmap),
        cax=cax,
        orientation='horizontal',
        label=MET_LABEL
    )

    # Format axes
    axs[0].set_xlim((2, 16))
    axs[0].set_ylim((-0.09, 0.06))
    axs[0].xaxis.set_major_locator(MultipleLocator(5))
    axs[0].xaxis.set_minor_locator(MultipleLocator(1))
    axs[0].yaxis.set_major_locator(MultipleLocator(0.05))
    axs[0].yaxis.set_minor_locator(MultipleLocator(0.01))
    axs[-1].set_xlabel('Guiding radius [kpc]')
    for i, ax in enumerate(axs):
        ax.set_title(r'$%s\leq z_{\rm max}<%s$ kpc' % ZBINS[i], y=0.83)
        ax.set_ylabel('Slope [dex/Gyr]')
    # colored_text_legend(axs[0], loc='center right', frameon=True, title=MET_LABEL)

    plt.savefig(paths.figures / 'global_metallicity_fits_alt2')


def fit_radial_bins(
        data, 
        bins, 
        min_count=MIN_COUNT, 
        xcol='age', 
        ycol='ce_mg_corr', 
        radius_col='Rg',
        age_fit_range=AGE_FIT_RANGE,
        age_delta=AGE_DELTA,
        **kwargs
    ):
    """
    Bin data by radius and fit a linear trend in each bin.
    
    Parameters
    ----------
    data : pandas.DataFrame
    bins : array-like
        Radius bin edges in kpc
    min_count : int, optional [default: 20]
        Minimum number of stars in a bin required to calculate a fit.
    xcol : str, optional [default: 'age']
        Column for the independent fit variable.
    ycol : str, optional [default: 'ce_mg_corr']
        Column for the dependent fit variable.
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
        rlim = bins[k:k+2]
        subset = data[
            (data[radius_col] >= rlim[0]) & 
            (data[radius_col] < rlim[1]) &
            (data[xcol] >= age_fit_range[0]) &
            (data[xcol] < age_fit_range[1])
        ]
        if subset.shape[0] > min_count:
            # Fit linear age trend
            regress = stats.linregress(
                subset[xcol] - age_delta, subset[ycol], **kwargs
            )
            fits.append(regress)
            bin_centers.append(np.mean(rlim))
    return fits, bin_centers


if __name__ == '__main__':
    main()
