"""
This script plots the sensitivity of the radially-binned [Ce/Mg]-age trend
to various sample selection factors and to the log(g) abundance corrections.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MultipleLocator
from matplotlib.cm import ScalarMappable

from plotting import TWO_COLUMN_WIDTH, RADIUS_COLORMAP, insert_colorbar_axes
from utils import import_sample, binned_quantiles
import paths

def main(style='paper'):
    full_sample = import_sample(good_ages=True, cut_limits=True)
    lowz_ages = full_sample[full_sample['z_max'] < 0.5]
    better_ages = lowz_ages[lowz_ages['training_density'] > 1e10]
    highsn_ages = lowz_ages[lowz_ages['snr'] > 250]
    rc_ages = lowz_ages[(lowz_ages['logg'] < 2.5) & (lowz_ages['logg'] > 2.2)]

    radius_bin_edges = np.arange(3, 15.1, 2)
    age_bin_edges = np.arange(-0.5, 11.6, 1)
    radial_cmap = plt.get_cmap(RADIUS_COLORMAP)
    norm = BoundaryNorm(radius_bin_edges, radial_cmap.N)

    plt.style.use(paths.styles / f'{style}.mplstyle')
    fig, axs = plt.subplots(
        2, 4, 
        figsize=(TWO_COLUMN_WIDTH, 0.6*TWO_COLUMN_WIDTH), 
        sharex='row', sharey=True, 
        gridspec_kw={'wspace': 0, 'hspace': 0.45, 'left': 0.05}
    )
    xlim = (0, 11)
    ylim = (-0.5, 0.8)

    # Top row: different sample selection choices
    labels = ['Full sample', r'Training density $>10^{10}$', r'$S/N > 250$', r'$2.2 < \log(g) < 2.5$']
    for i, df in enumerate([lowz_ages, better_ages, highsn_ages, rc_ages]):
        # Plot all stars
        pcm = axs[0,i].hexbin(
            df['age'], df['ce_mg_corr'],
            C=np.ones(df.shape[0]),
            reduce_C_function=np.sum,
            gridsize=20,
            cmap='binary',
            linewidths=0.2,
            mincnt=0,
            extent=[xlim[0], xlim[1], ylim[0], ylim[1]]
        )
        # Plot median age trends binned by radius
        for j in range(len(radius_bin_edges)-1):
            radius_bin = radius_bin_edges[j:j+2]
            mean_radius = np.mean(radius_bin)
            df_subset = df[
                (df['Rg'] >= radius_bin[0]) &
                (df['Rg'] < radius_bin[1])
            ]
            age_medians = binned_quantiles(
                df_subset, 'ce_mg_corr', 'age',
                q=0.5, bin_edges=age_bin_edges, min_count=20, est_errors=True
            )
            axs[0,i].plot(
                *age_medians[:-1], '-', 
                color=radial_cmap(norm(mean_radius)), zorder=1,
                label=f'{int(mean_radius)} kpc'
            )
            axs[0,i].fill_between(
                age_medians[0],
                age_medians[1]+age_medians[2],
                age_medians[1]-age_medians[2],
                color=radial_cmap(norm(mean_radius)),
                alpha=0.5, edgecolor='none', zorder=2
            )
            # Compare vs median trends in full sample
            full_subset = lowz_ages[
                (lowz_ages['Rg'] >= radius_bin[0]) &
                (lowz_ages['Rg'] < radius_bin[1])
            ]
            full_age_medians = binned_quantiles(
                full_subset, 'ce_mg_corr', 'age',
                q=0.5, bin_edges=age_bin_edges, min_count=20, est_errors=True
            )
            axs[0,i].plot(
                *full_age_medians[:-1], '--', 
                color=radial_cmap(norm(mean_radius)), zorder=1,
            )
        # Sample size
        axs[0,i].text(1, -0.4, r'$N=%s$' % df.shape[0])
        axs[0,i].text(5.5, 0.7, labels[i], ha='center', va='top')

    # Bottom row: bins in log(g) without corrective offsets
    labels = ['Full sample', r'$1.5 < \log(g) < 2.0$', r'$2.0 < \log(g) < 2.5$', r'$2.5 < \log(g) < 3.0$']
    logg_bins = [(1, 3), (1.5, 2), (2, 2.5), (2.5, 3)]
    for i, logg_lim in enumerate(logg_bins):
        df = lowz_ages[(lowz_ages['logg'] > logg_lim[0]) & (lowz_ages['logg'] < logg_lim[1])]
        # Plot all stars
        pcm = axs[1,i].hexbin(
            df['age'], df['ce_mg'],
            C=np.ones(df.shape[0]),
            reduce_C_function=np.sum,
            gridsize=20,
            cmap='binary',
            linewidths=0.2,
            mincnt=0,
            extent=[xlim[0], xlim[1], ylim[0], ylim[1]]
        )
        # Plot median age trends binned by radius
        for j in range(len(radius_bin_edges)-1):
            radius_bin = radius_bin_edges[j:j+2]
            mean_radius = np.mean(radius_bin)
            df_subset = df[
                (df['Rg'] >= radius_bin[0]) &
                (df['Rg'] < radius_bin[1])
            ]
            age_medians = binned_quantiles(
                df_subset, 'ce_mg', 'age',
                q=0.5, bin_edges=age_bin_edges, min_count=20, est_errors=True
            )
            axs[1,i].plot(
                *age_medians[:-1], '-', 
                color=radial_cmap(norm(mean_radius)), zorder=1,
                label=f'{int(mean_radius)} kpc'
            )
            axs[1,i].fill_between(
                age_medians[0],
                age_medians[1]+age_medians[2],
                age_medians[1]-age_medians[2],
                color=radial_cmap(norm(mean_radius)),
                alpha=0.5, edgecolor='none', zorder=2
            )
            # Compare vs median trends in full sample (logg-corrected)
            full_subset = lowz_ages[
                (lowz_ages['Rg'] >= radius_bin[0]) &
                (lowz_ages['Rg'] < radius_bin[1])
            ]
            full_age_medians = binned_quantiles(
                full_subset, 'ce_mg_corr', 'age',
                q=0.5, bin_edges=age_bin_edges, min_count=20, est_errors=True
            )
            axs[1,i].plot(
                *full_age_medians[:-1], '--', 
                color=radial_cmap(norm(mean_radius)), zorder=1,
            )
        # Sample size
        axs[1,i].text(1, -0.4, r'$N=%s$' % df.shape[0])
        axs[1,i].text(5.5, 0.7, labels[i], ha='center', va='top')

    cax = insert_colorbar_axes(fig, width=0.015, pad=0.02)
    fig.colorbar(ScalarMappable(norm, radial_cmap), cax=cax, label='Guiding radius [kpc]')

    axs[0,1].set_title(r'With $\log(g)$ corrections', x=1)
    axs[1,1].set_title(r'Without $\log(g)$ corrections', x=1)
    for ax in axs[:,0]:
        ax.set_ylabel('[Ce/Mg]')
    for ax in axs.flatten():
        ax.set_xlabel('Age [Gyr]')
    axs[0,0].set_xlim(xlim)
    axs[1,0].set_xlim(xlim)
    axs[0,0].set_ylim(ylim)
    axs[0,0].xaxis.set_major_locator(MultipleLocator(5))
    axs[0,0].xaxis.set_minor_locator(MultipleLocator(1))
    axs[1,0].xaxis.set_major_locator(MultipleLocator(5))
    axs[1,0].xaxis.set_minor_locator(MultipleLocator(1))
    axs[0,0].yaxis.set_major_locator(MultipleLocator(0.5))
    axs[0,0].yaxis.set_minor_locator(MultipleLocator(0.1))
    # colored_text_legend(axs[0,0], ncols=2, fontsize=7, columnspacing=1)
    plt.savefig(paths.figures / 'age_trend_sensitivity')
    plt.close()

if __name__ == '__main__':
    main()
