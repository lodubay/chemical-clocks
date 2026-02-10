"""
Compare median trends in [Ce/Mg] and residual [Ce/H] as a function of age
and guiding-center radius.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import BoundaryNorm, Normalize

from utils import apply_alpha_cut, binned_quantiles, good_ages
from plotting import insert_colorbar_axes, colored_text_legend, ONE_COLUMN_WIDTH
from colormaps import paultol
import paths

def main(style='paper', cmap='viridis_r'):
    plt.style.use(paths.styles / f'{style}.mplstyle')
    # Import MWM sample
    mwm_rgb = pd.read_csv(paths.data / 'MWM' / 'sample.csv')
    radius_bin_edges = np.arange(3, 15.1, 2)
    age_bin_edges = np.arange(-0.5, 11.6, 1)
    mg_bin_edges = np.arange(-0.75, 0.76, 0.1)

    # Select only low-alpha, near-midplane stars
    mwm_rgb = apply_alpha_cut(mwm_rgb)
    mwm_rgb['delta_ce_h'] = np.nan * np.ones(mwm_rgb.shape[0])
    all_lowz = mwm_rgb[(mwm_rgb['z_max'] < 0.5)].copy()

    # Calculate residual [Ce/H] for high- and low-alpha stars
    for i in range(len(radius_bin_edges)-1):
        radius_bin = radius_bin_edges[i:i+2]
        low_alpha_subset = all_lowz[
            (all_lowz['Rg'] >= radius_bin[0]) &
            (all_lowz['Rg'] < radius_bin[1]) &
            (all_lowz['low_alpha'])
        ]
        low_alpha_medians = binned_quantiles(
            low_alpha_subset, 'ce_h_corr', 'mg_h',
            q=0.5, bin_edges=mg_bin_edges, min_count=10
        )
        all_lowz.loc[low_alpha_subset.index, 'delta_ce_h'] = \
            low_alpha_subset['ce_h_corr'] - np.interp(
                low_alpha_subset['mg_h'], *low_alpha_medians
            )
        high_alpha_subset = all_lowz[
            (all_lowz['Rg'] >= radius_bin[0]) &
            (all_lowz['Rg'] < radius_bin[1]) &
            (all_lowz['high_alpha'])
        ]
        high_alpha_medians = binned_quantiles(
            high_alpha_subset, 'ce_h_corr', 'mg_h',
            q=0.5, bin_edges=mg_bin_edges, min_count=10
        )
        all_lowz.loc[high_alpha_subset.index, 'delta_ce_h'] = \
            high_alpha_subset['ce_h_corr'] - np.interp(
                high_alpha_subset['mg_h'], *high_alpha_medians
            )
    
    # Set up figure
    fig, axs = plt.subplots(
        2, 1,
        figsize=(ONE_COLUMN_WIDTH, 1.5*ONE_COLUMN_WIDTH),
        sharex=True,
        gridspec_kw={'hspace': 0.}
    )
    cax = insert_colorbar_axes(fig, 'horizontal', pad=0.05)
    radial_cmap = plt.get_cmap(cmap)
    norm = BoundaryNorm(radius_bin_edges, radial_cmap.N)
    xlim = (0, 12)
    ylim = [(-0.5, 0.7), (-0.6, 0.6)]

    lowz_ages = good_ages(all_lowz)
    low_alpha_ages = lowz_ages[lowz_ages['low_alpha']]
    high_alpha_ages = lowz_ages[lowz_ages['high_alpha']]
    for i, col in enumerate(['ce_mg', 'delta_ce_h']):
        # Plot all stars
        pcm = axs[i].hexbin(
            lowz_ages['age'], lowz_ages[col],
            C=np.ones(lowz_ages.shape[0]),
            reduce_C_function=np.sum,
            gridsize=(30, 12),
            cmap='binary',
            norm=Normalize(vmin=0, vmax=400),
            linewidths=0.2,
            mincnt=1,
            extent=[xlim[0], xlim[1], ylim[i][0], ylim[i][1]]
        )
        # fig.colorbar(pcm, ax=axs[i])

        for j in range(len(radius_bin_edges)-1):
            radius_bin = radius_bin_edges[j:j+2]
            mean_radius = np.mean(radius_bin)
            # Plot low alpha trends
            low_alpha_subset = low_alpha_ages[
                (low_alpha_ages['Rg'] >= radius_bin[0]) &
                (low_alpha_ages['Rg'] < radius_bin[1])
            ]
            low_alpha_age_medians = binned_quantiles(
                low_alpha_subset, col, 'age',
                q=0.5, bin_edges=age_bin_edges, min_count=10, est_errors=True
            )
            # axs[i].plot(*low_alpha_age_medians, '-', color='w', linewidth=2)
            axs[i].plot(
                *low_alpha_age_medians[:-1], '-', 
                color=radial_cmap(norm(mean_radius)), zorder=4,
                label=f'{int(mean_radius)} kpc'
            )
            axs[i].fill_between(
                low_alpha_age_medians[0],
                low_alpha_age_medians[1]+low_alpha_age_medians[2],
                low_alpha_age_medians[1]-low_alpha_age_medians[2],
                color=radial_cmap(norm(mean_radius)),
                alpha=0.5, edgecolor='none', zorder=2
            )
            # Plot high alpha trends
            high_alpha_subset = high_alpha_ages[
                (high_alpha_ages['Rg'] >= radius_bin[0]) &
                (high_alpha_ages['Rg'] < radius_bin[1])
            ]
            high_alpha_age_medians = binned_quantiles(
                high_alpha_subset, col, 'age',
                q=0.5, bin_edges=age_bin_edges, min_count=10, est_errors=True
            )
            # axs[i].plot(*high_alpha_age_medians[:-1], '-', color='w', linewidth=2)
            axs[i].plot(
                *high_alpha_age_medians[:-1], '--', 
                color=radial_cmap(norm(mean_radius)), zorder=3
            )
            axs[i].fill_between(
                high_alpha_age_medians[0],
                high_alpha_age_medians[1]+high_alpha_age_medians[2],
                high_alpha_age_medians[1]-high_alpha_age_medians[2],
                color=radial_cmap(norm(mean_radius)),
                edgecolor='none', alpha=0.5, zorder=1
            )
    fig.colorbar(pcm, cax=cax, orientation='horizontal', label='Number of stars')
    # Indicate median abundance errors
    mwm_rgb_ages = good_ages(mwm_rgb)
    age_err_low = np.median(mwm_rgb_ages['age'] - mwm_rgb_ages['e_n_age'])
    age_err_high = np.median(mwm_rgb_ages['e_p_age'] - mwm_rgb_ages['age'])
    med_abund_err = mwm_rgb_ages['e_ce_h'].median()
    axs[0].errorbar(
        9.5, 0.55, 
        xerr=[[age_err_low], [age_err_high]], 
        yerr=med_abund_err, 
        c='gray', capsize=0, #elinewidth=0.5,
    )

    axs[0].set_xlim(xlim)
    axs[0].set_ylim(ylim[0])
    axs[1].set_ylim(ylim[1])

    axs[0].yaxis.set_major_locator(MultipleLocator(0.5))
    axs[0].yaxis.set_minor_locator(MultipleLocator(0.1))
    axs[1].yaxis.set_major_locator(MultipleLocator(0.5))
    axs[1].yaxis.set_minor_locator(MultipleLocator(0.1))
    axs[1].xaxis.set_major_locator(MultipleLocator(5))
    axs[1].xaxis.set_minor_locator(MultipleLocator(1))

    axs[0].set_ylabel('[Ce/Mg]')
    axs[1].set_ylabel(r'$\Delta$[Ce/H]')
    axs[1].set_xlabel('Age [Gyr]')

    for ax in axs:
        colored_text_legend(ax, loc='center right')

    plt.savefig(paths.figures / 'median_age_trends')
    plt.close()


if __name__ == '__main__':
    main()
