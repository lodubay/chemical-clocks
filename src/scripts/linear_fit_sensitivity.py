"""
This script generates an appendix plot illustrating the sensitivity of the
metallicity-dependent linear fits to various sample selection factors.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

import paths
from utils import import_sample, get_bin_centers
from colormaps import paultol
from plotting import ONE_COLUMN_WIDTH, colored_text_legend
from global_metallicity_fits import fit_metallicity_bins


def main(style='paper'):
    sample = import_sample(good_ages=True, cut_limits=True)
    local_sample = sample[(sample['Rg']>=7) & (sample['Rg']<9) & (sample['z_max']<0.5)]
    local_betterage = local_sample[local_sample['training_density'] > 1e10]
    local_highsn = local_sample[local_sample['snr'] > 250]
    local_rconly = local_sample[(local_sample['logg'] < 2.5) & (local_sample['logg'] > 2.2)]

    plt.style.use(paths.styles / f'{style}.mplstyle')
    plt.rcParams['axes.prop_cycle'] = plt.cycler('color', paultol.bright.colors)
    fig, axs = plt.subplots(
        2, 
        figsize=(ONE_COLUMN_WIDTH, 1.6*ONE_COLUMN_WIDTH), 
        sharex=True,
        gridspec_kw={'hspace': 0}
    )

    met_bins = np.arange(-0.45, 0.46, 0.1)
    age_fit_range = (1, 8)
    age_delta = 5 # Gyr

    ycols = ['ce_mg_corr'] * 4 + ['ce_mg']
    metcols = ['fe_h_corr'] * 4 + ['fe_h']
    labels = ['Full sample', r'Age training density $>10^{10}$', r'$S/N > 250$', r'$2.2 < \log(g) < 2.5$', r'No $\log(g)$ corrections']
    for i, df in enumerate([local_sample, local_betterage, local_highsn, local_rconly, local_sample]):
        df_high_ia = df[df['high_ia']]
        # Bin by metallicity and fit linear trend to stars within good age range
        params, errors, met_bin_centers = fit_metallicity_bins(
            df_high_ia, 
            met_bins, 
            ycol=ycols[i],
            metcol=metcols[i],
            age_fit_range=age_fit_range, 
            age_delta=age_delta
        )
        # Plot slopes
        slopes = params[:,1]
        slope_errs = errors[:,1]
        met_bin_centers = get_bin_centers(met_bins)
        axs[0].errorbar(
            met_bin_centers, slopes, yerr=slope_errs, 
            label=labels[i],
            marker='o', linestyle='-', capsize=0
        )
        # Plot intercepts
        intercepts = params[:,0]
        intercept_errs = errors[:,0]
        axs[1].errorbar(
            met_bin_centers, intercepts, yerr=intercept_errs, 
            label=labels[i],
            marker='s', linestyle='-', capsize=0
        )

    colored_text_legend(axs[0])
    axs[1].set_xlabel('[Fe/H]')
    axs[0].set_ylabel(r'Slope [dex Gyr$^{-1}$]')
    axs[1].set_ylabel(r'[Ce/Mg] at $\tau=%s$ Gyr' % age_delta)
    axs[0].set_xlim((-0.5, 0.5))
    axs[0].set_ylim((-0.08, 0.05))
    axs[1].set_ylim((-0.2, 0.2))
    axs[0].xaxis.set_major_locator(MultipleLocator(0.5))
    axs[0].xaxis.set_minor_locator(MultipleLocator(0.1))
    axs[0].yaxis.set_major_locator(MultipleLocator(0.05))
    axs[0].yaxis.set_minor_locator(MultipleLocator(0.01))
    axs[1].yaxis.set_major_locator(MultipleLocator(0.1))
    axs[1].yaxis.set_minor_locator(MultipleLocator(0.02))
    plt.savefig(paths.figures / 'linear_fit_sensitivity')
    plt.close()


if __name__ == '__main__':
    main()
