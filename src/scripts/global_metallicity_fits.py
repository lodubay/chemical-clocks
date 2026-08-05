"""
This script plots the slope of the age-[Ce/Mg] trend as a function of 
metallicity for each region of the Galaxy.
"""
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import ScalarMappable
import statsmodels.api as sm

from utils import import_sample, get_bin_centers
from plotting import ONE_COLUMN_WIDTH, RADIUS_COLORMAP, insert_colorbar_axes
# from stats import deming_regression, bootstrap_standard_error
import paths

ZLIM = (0, 0.5) # global z_max limits
AGE_FIT_RANGE = (1, 8) # Range of ages to fit linear trend
MIN_COUNT = 20 # Minimum number of stars in each bin for trend fitting
AGE_DELTA = 5 # Gyr, linear age shift for regression


def main(style='paper', cmap=RADIUS_COLORMAP):
    plt.style.use(paths.styles / f'{style}.mplstyle')
    savedir = {
        'paper': paths.figures,
        'presentation': paths.extra/'presentation'
    }[style]
    savedir.mkdir(exist_ok=True)

    # Metallicity bins
    met_bin_edges = np.arange(-0.85, 0.56, 0.1)
    dr = 2
    radius_bin_edges = np.arange(3, 15+dr, dr)

    # Import MWM sample
    mwm_sample = import_sample(good_ages=True)

    # Set up figure
    fig, axs = plt.subplots(
        2,
        figsize=(ONE_COLUMN_WIDTH, 1.5*ONE_COLUMN_WIDTH), 
        sharex=True,
        gridspec_kw={'hspace': 0}
    )
    cmap = plt.get_cmap(cmap)
    norm = BoundaryNorm(radius_bin_edges, cmap.N)
    cax = insert_colorbar_axes(fig, orientation='horizontal', pad=0.05)

    for j in range(len(radius_bin_edges)-1):
        rlim = radius_bin_edges[j:j+2]
        mean_radius = np.mean(rlim)
        color = cmap(norm(mean_radius))
        region = mwm_sample[
            (mwm_sample['Rg'] >= rlim[0]) &
            (mwm_sample['Rg'] < rlim[1]) &
            (mwm_sample['z_max'] >= ZLIM[0]) &
            (mwm_sample['z_max'] < ZLIM[1]) &
            (mwm_sample['high_ia']) # restrict age trends to low-alpha only
        ]
        # Bin by metallicity and fit linear trend to stars
        params, errors, mets = fit_metallicity_bins(region, met_bin_edges)
        # Plot fit slopes
        slopes = params[:,1]
        slope_errs = errors[:,1]
        axs[0].plot(
            mets, slopes, '.-',
            color=color,
            label=f'{int(mean_radius)} kpc'
        )
        axs[0].fill_between(
            mets, slopes - slope_errs, slopes + slope_errs,
            color=color, 
            alpha=0.5, 
            edgecolor='none'
        )
        # Plot intercepts
        intercepts = params[:,0]
        int_errs = errors[:,0]
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
    axs[0].set_xlim((-0.7, 0.5))
    axs[0].set_ylim((-0.1, 0.05))
    axs[0].xaxis.set_major_locator(MultipleLocator(0.5))
    axs[0].xaxis.set_minor_locator(MultipleLocator(0.1))
    axs[0].yaxis.set_major_locator(MultipleLocator(0.05))
    axs[0].yaxis.set_minor_locator(MultipleLocator(0.01))
    axs[0].set_ylabel('Slope [dex/Gyr]')
    axs[1].set_ylim((-0.2, 0.34))
    axs[1].yaxis.set_major_locator(MultipleLocator(0.1))
    axs[1].yaxis.set_minor_locator(MultipleLocator(0.02))
    axs[1].set_ylabel(r'[Ce/Mg] at $\tau=5$ Gyr')
    axs[1].set_xlabel('[Fe/H]')
    # colored_text_legend(ax, loc='center right', frameon=True)

    plt.savefig(savedir / 'global_metallicity_fits')


def fit_metallicity_bins(
        data, 
        bins, 
        min_count=MIN_COUNT, 
        age_fit_range=AGE_FIT_RANGE,
        age_delta=AGE_DELTA,
        **kwargs
    ):
    """
    Bin data by metallicity and fit a linear trend in each bin.
    
    Parameters
    ----------
    data : pandas.DataFrame
    bins : array-like of length N
        Metallicity bin edges
    min_count : int, optional [default: 20]
        Minimum number of stars in a bin required to calculate a fit.
    age_fit_range : tuple of floats, optional [default: (1, 8)]
        Range of ages considered valid for fit procedure. Data outside this
        range will not contribute to the fit.
    age_delta : float, optional [default: 5]
        Linear age shift in Gyr to center regression
    **kwargs passed to statsmodels.WLS
    
    Returns
    -------
    params : (N-1) x 2 array
        Linear fit parameters for each metallicity bin. Intercepts are stored
        in params[:,0], and slopes in params[:,1].
    errors : (N-1) x 2 array
        Standard errors on linear fit parameters.
    bin_centers : (N-1) array
        Mean of each metallicity bin for which a fit was performed.
    """
    params = []
    errors = []
    bin_centers = []
    for k in range(len(bins)-1):
        met_lim = bins[k:k+2]
        subset = data[
            (data['fe_h_corr'] >= met_lim[0]) & 
            (data['fe_h_corr'] < met_lim[1]) &
            (data['age'] >= age_fit_range[0]) &
            (data['age'] < age_fit_range[1])
        ]
        if subset.shape[0] > min_count:
            x = subset['age'].values - age_delta
            X = x[:,np.newaxis]
            X = sm.add_constant(X)
            y = subset['ce_mg_corr'].values
            xerr = subset['e_mean_age'].values
            yerr = subset['e_ce_mg'].values
            # dem_reg = deming_regression(x, y, xerr, yerr)
            # dem_err = bootstrap_standard_error(deming_regression, x, y, xerr, yerr)
            # params.append(dem_reg)
            # errors.append(dem_err)

            # Initial fit without xerr
            weights = 1 / (yerr**2)
            wls_model = sm.WLS(y, X, weights=weights)
            results = wls_model.fit()

            # Re-fit with x-errors, adopting previous best-fit slope, until
            # old and new parameters converge within 1%
            r = 1
            i = 0
            m = results.params[1]
            while r > 0.01 and i < 10:
                weights = 1 / (yerr**2 + m**2 * xerr**2)
                wls_model = sm.WLS(y, X, weights=weights)
                results = wls_model.fit()
                i += 1
                r = abs((results.params[1] - m) / m)
                m = results.params[1]

            params.append(results.params)
            errors.append(results.bse)
            bin_centers.append(np.mean(met_lim))
    return np.array(params), np.array(errors), np.array(bin_centers)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Plot fits to the age--[Ce/Mg] relation across radial bins.'
    )
    parser.add_argument('--style',
        choices=('paper', 'presentation'),
        default='paper',
        help='Plot style to use (default: "paper").'
    )
    parser.add_argument('--cmap',
        default=RADIUS_COLORMAP,
        type=str,
        help='Colormap for the radial bins (default: "managua").'
    )
    args = parser.parse_args()
    main(**vars(args))
