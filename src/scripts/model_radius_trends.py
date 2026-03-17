"""
Compare [Ce/Mg]-age trends at different radii predicted by the model to the data
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import BoundaryNorm, Normalize
import vice

from utils import apply_alpha_cut, binned_quantiles, good_ages, get_bin_centers
from plotting import colored_text_legend, ONE_COLUMN_WIDTH
from colormaps import paultol
import paths
from multizone._globals import ZONE_WIDTH

OUTPUT_NAME = 'karakas16-mscale'

def main(style='paper', cmap='viridis_r'):
    plt.style.use(paths.styles / f'{style}.mplstyle')
    # Import MWM sample
    mwm_rgb = pd.read_csv(paths.data / 'MWM' / 'sample.csv')
    radius_bin_edges = np.arange(3, 15.1, 2)
    age_bin_edges = np.arange(-0.5, 11.6, 1)

    # Select only low-alpha, near-midplane stars
    mwm_rgb = apply_alpha_cut(mwm_rgb)
    mwm_rgb['delta_ce_h'] = np.nan * np.ones(mwm_rgb.shape[0])
    all_lowz = mwm_rgb[(mwm_rgb['z_max'] < 0.5)].copy()
    
    # Set up figure
    fig, ax = plt.subplots(figsize=(ONE_COLUMN_WIDTH, ONE_COLUMN_WIDTH))
    radial_cmap = plt.get_cmap(cmap)
    norm = BoundaryNorm(radius_bin_edges, radial_cmap.N)
    xlim = (0, 13)
    ylim = (-0.5, 0.8)

    lowz_ages = good_ages(all_lowz)
    low_alpha_ages = lowz_ages[lowz_ages['low_alpha']]
    high_alpha_ages = lowz_ages[lowz_ages['high_alpha']]
    # Plot all stars
    pcm = ax.hexbin(
        lowz_ages['age'], lowz_ages['ce_mg_corr'],
        C=np.ones(lowz_ages.shape[0]),
        reduce_C_function=np.sum,
        gridsize=(30, 12),
        cmap='binary',
        linewidths=0.2,
        mincnt=0,
        extent=[xlim[0], xlim[1], ylim[0], ylim[1]]
    )
    # Plot low-alpha trends binned by radius
    for j in range(len(radius_bin_edges)-1):
        radius_bin = radius_bin_edges[j:j+2]
        mean_radius = np.mean(radius_bin)
        low_alpha_subset = low_alpha_ages[
            (low_alpha_ages['Rg'] >= radius_bin[0]) &
            (low_alpha_ages['Rg'] < radius_bin[1])
        ]
        low_alpha_age_medians = binned_quantiles(
            low_alpha_subset, 'ce_mg_corr', 'age',
            q=0.5, bin_edges=age_bin_edges, min_count=10, est_errors=True
        )
        # axs[i].plot(*low_alpha_age_medians, '-', color='w', linewidth=2)
        ax.plot(
            *low_alpha_age_medians[:-1], '--', 
            color=radial_cmap(norm(mean_radius)), zorder=1,
            label=f'{int(mean_radius)} kpc'
        )
    fig.colorbar(pcm, ax=ax, orientation='horizontal', label='Number of stars')
    # Indicate median abundance errors
    mwm_rgb_ages = good_ages(mwm_rgb)
    age_err_low = np.median(mwm_rgb_ages['age'] - mwm_rgb_ages['e_n_age'])
    age_err_high = np.median(mwm_rgb_ages['e_p_age'] - mwm_rgb_ages['age'])
    med_abund_err = mwm_rgb_ages['e_ce_h'].median()
    ax.errorbar(
        10, 0.6, 
        xerr=[[age_err_low], [age_err_high]], 
        yerr=med_abund_err, 
        c='gray', capsize=0, #elinewidth=0.5,
    )

    # Plot multizon evolution
    for radius in get_bin_centers(radius_bin_edges):
        zone = int(radius / ZONE_WIDTH)
        zone_path = str(
            paths.multizone / OUTPUT_NAME / 'diskmodel.vice' / ('zone%d' % zone)
        )
        hist = vice.history(zone_path)
        # ax.plot(hist['lookback'], hist['[ce/mg]'], 'w-', lw=2)
        ax.plot(
            hist['lookback'], hist['[ce/mg]'], 
            color=radial_cmap(norm(radius)), ls='-'
        )

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(1))

    ax.set_ylabel('[Ce/Mg]')
    ax.set_xlabel('Age [Gyr]')

    handles, labels = ax.get_legend_handles_labels()
    leg = colored_text_legend(ax, invert=True, loc='center right')

    plt.savefig(paths.figures / 'model_radius_trends')
    plt.close()


if __name__ == '__main__':
    main()
