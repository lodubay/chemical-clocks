"""
This script contains code for calculating residual abundances. If run as a 
script, it makes a plot explaining the calculation of the residual Ce abundance.
"""
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from utils import binned_quantiles, sample_rows, import_sample
from plotting import colored_text_legend, iterate_rz_bins, ONE_COLUMN_WIDTH
from colormaps import paultol
import paths

RBINS = [(3, 5), (5, 7), (7, 9), (9, 11), (11, 13)] # left to right
ZBINS = [(1, 2), (0.5, 1), (0, 0.5)] # top to bottom
SAMPLE_FRACTION = 0.25 # fraction of stars to plot in each panel

def main(style='paper', sample_fraction=SAMPLE_FRACTION):
    # Import MWM sample
    mwm_rgb = import_sample(good_ages=False)
    # Solar neighborhood sample
    mwm_rgb_local = mwm_rgb[
        (mwm_rgb['Rg'] >= 7) &
        (mwm_rgb['Rg'] < 9) &
        (mwm_rgb['z_max'] < 0.5)
    ].copy()
    # Calculate residual abundances (Solar neighborhood only)
    mwm_rgb_local = residual_abundances(
        mwm_rgb_local, 
        col='ce_h_corr', 
        newcol='delta_ce_h',
        rbins=[(7, 9)],
        zbins=[(0, 0.5)]
    )
    # Scatter plot random sample of points
    sample = sample_rows(
        mwm_rgb_local, 
        int(sample_fraction * mwm_rgb_local.shape[0])
    )
    mg_bin_edges = np.arange(-0.75, 0.56, 0.1)
    age_bin_edges = np.arange(0.5, 11.6, 1)

    # Set up figure
    plt.style.use(paths.styles / f'{style}.mplstyle')
    savedir = {
        'paper': paths.figures,
        'presentation': paths.extra/'presentation'
    }[style]
    savedir.mkdir(exist_ok=True)
    fig, axs = plt.subplots(2, 2, 
        figsize=(ONE_COLUMN_WIDTH, ONE_COLUMN_WIDTH),
        sharex='col', sharey='row', 
        gridspec_kw={'hspace': 0, 'wspace': 0}
    )
    # scatterplot style arguments
    kwargs = dict(s=1, marker='.', rasterized=True, edgecolor='none')
    # Label each panel
    labels = ['(a)', '(b)', '(c)', '(d)']
    for i, ax in enumerate(axs.flatten()):
        ax.set_title(labels[i], y=0.93, x=0.07, ha='left', va='top', pad=0)

    # Loop through low- and high-alpha populations
    pops = ['high_ia', 'low_ia']
    colors = [paultol.highcontrast.colors[0], paultol.highcontrast.colors[2]]
    labels = ['High-Ia', 'Low-Ia']
    formats = ['s-', 'o-']
    ms = 3
    for pop, color, label, fmt in zip(pops, colors, labels, formats):
        local_pop = mwm_rgb_local[mwm_rgb_local[pop]]
        sample_pop = sample[sample[pop]]
        # Plot [Mg/H] vs [Ce/H]
        axs[0,0].scatter(
            sample_pop['mg_h'], sample_pop['ce_h_corr'], 
            c=color, **kwargs
        )
        # Plot median trend with [Mg/H]
        local_pop_medians = binned_quantiles(
            local_pop, 'ce_h_corr', 'mg_h', 
            q=0.5, bin_edges=mg_bin_edges, min_count=10
        )
        axs[0,0].plot(
            *local_pop_medians, 
            fmt, color=color, label=label, ms=ms
        )
        # Plot [Ce/H] residuals vs [Mg/H]
        axs[1,0].scatter(
            sample_pop['mg_h'], sample_pop['delta_ce_h'],
            c=color, **kwargs
        )
        # Plot median and 1-sigma bands
        for q, ls in zip([0.16, 0.5, 0.84], ['--', '-', '--']):
            local_pop_quantile = binned_quantiles(
                local_pop, 'delta_ce_h', 'mg_h',
                q=q, bin_edges=mg_bin_edges, min_count=10
            )
            axs[1,0].plot(*local_pop_quantile, ls, color=color, zorder=6)
        # Plot [Ce/H] vs age
        sample_pop_ages = sample_pop[sample_pop['good_age']]
        axs[0,1].scatter(
            sample_pop_ages['age'], sample_pop_ages['ce_h_corr'],
            c=color, **kwargs
        )
        # Plot median trends with age
        local_pop_ages = local_pop[local_pop['good_age']]
        pop_age_medians = binned_quantiles(
            local_pop_ages, 'ce_h_corr', 'age',
            q=0.5, bin_edges=age_bin_edges, min_count=10
        )
        axs[0,1].plot(*pop_age_medians, fmt, color=color, zorder=6, ms=ms)
        # Plot [Ce/H] residuals vs age
        axs[1,1].scatter(
            sample_pop_ages['age'], sample_pop_ages['delta_ce_h'],
            c=color, **kwargs
        )
        # Plot median trends with age
        pop_res_age_medians = binned_quantiles(
            local_pop_ages, 'delta_ce_h', 'age',
            q=0.5, bin_edges=age_bin_edges, min_count=10
        )
        axs[1,1].plot(*pop_res_age_medians, fmt, color=color, zorder=6, ms=ms)
        
    # Horizontal lines for reference
    axs[1,0].plot([-0.7, 0.6], [0, 0], linestyle=':', color='gray', zorder=5)
    axs[1,1].plot([-1, 12], [0, 0], linestyle=':', color='gray', zorder=5)

    # Axes labels
    axs[0,0].set_ylabel('[Ce/H]')
    axs[1,0].set_ylabel(r'$\Delta$[Ce/H]')
    axs[1,0].set_xlabel('[Mg/H]')
    axs[1,1].set_xlabel('Age [Gyr]')

    # Axes limits
    axs[0,0].set_xlim((-0.7, 0.6))
    axs[0,0].set_ylim((-0.8, 0.8))
    axs[1,0].set_ylim((-0.8, 0.8))
    axs[0,1].set_xlim((-1, 12))

    # Axes ticks
    axs[0,0].xaxis.set_major_locator(MultipleLocator(0.5))
    axs[0,0].xaxis.set_minor_locator(MultipleLocator(0.1))
    axs[0,0].yaxis.set_major_locator(MultipleLocator(0.5))
    axs[0,0].yaxis.set_minor_locator(MultipleLocator(0.1))
    axs[1,0].yaxis.set_major_locator(MultipleLocator(0.5))
    axs[1,0].yaxis.set_minor_locator(MultipleLocator(0.1))
    axs[0,1].xaxis.set_major_locator(MultipleLocator(5))
    axs[0,1].xaxis.set_minor_locator(MultipleLocator(1))

    leg = colored_text_legend(
        axs[0,0], 
        loc='lower right',
        frameon=True,
        framealpha=1,
        edgecolor='none'
    )

    plt.savefig(savedir / 'residual_abundances')


def residual_abundances(
        catalog, 
        col='ce_h_corr', 
        newcol='delta_ce_h', 
        rbins=RBINS,
        zbins=ZBINS
    ):
    """
    Calculate residual [Ce/H] abundances for high- and low-alpha populations
    separately in each Galactic region (defined by bins in Rg and z_max).
    
    Parameters
    ----------
    catalog : pandas.DataFrame
        Full MWM catalog.
    col : str, optional [default: 'ce_h_corr']
        Column with abundances to calculate the residuals for.
    newcol : str, optional [default: 'delta_ce_h']
        Name of new column with residual abundances
    rbins : list of tuples, optional
        List of bins in guiding radius (Rg) in which to calculate residuals.
    zbins : list of tuples, optional
        List of bins in z_max in which to calculate residuals.

    Returns
    -------
    catalog : pandas.DataFrame
        MWM catalog with residual abundance column appended.
    """
    mg_bin_edges = np.arange(-0.75, 0.76, 0.1)
    # Calculate residuals separately in each region for low-alpha
    res_abund = []
    for i, j, zlim, rlim in iterate_rz_bins(rbins=rbins, zbins=zbins):
        subset = catalog[
            (catalog['Rg'] >= rlim[0]) &
            (catalog['Rg'] < rlim[1]) &
            (catalog['z_max'] >= zlim[0]) &
            (catalog['z_max'] < zlim[1]) &
            (catalog['high_ia'])
        ].copy()
        if subset.shape[0] >= 100:
            # Calculate median trend with [Mg/H]
            mgh_medians = binned_quantiles(
                subset, col, 'mg_h', 
                q=0.5, bin_edges=mg_bin_edges, min_count=10
            )
            # Calculate residual Ce abundance
            subset[newcol] = subset[col] - np.interp(
                subset['mg_h'], *mgh_medians
            )
            res_abund.append(subset[[newcol]].copy())
    # Calculate high-alpha residuals all together
    all_low_ia = catalog[catalog['low_ia']].copy()
    low_ia_medians = binned_quantiles(
        all_low_ia, col, 'mg_h', 
        q=0.5, bin_edges=mg_bin_edges, min_count=10
    )
    all_low_ia[newcol] = all_low_ia[col] - np.interp(
        all_low_ia['mg_h'], *low_ia_medians
    )
    res_abund.append(all_low_ia[[newcol]].copy())
    # Join residual abundances to catalog DataFrame
    res_abund = pd.concat(res_abund)
    catalog = catalog.join(res_abund)
    return catalog


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Explainer plot for the residual abundance calculation.'
    )
    parser.add_argument('--style',
        choices=('paper', 'presentation'),
        default='paper',
        help='Plot style to use (default: "paper").'
    )
    parser.add_argument('--sample-fraction',
        type=float,
        default=SAMPLE_FRACTION,
        help='Fraction of stars to randomly sample in each panel (default: 0.25).'
    )
    args = parser.parse_args()
    main(**vars(args))
