"""
Compare [Ce/Mg]-age trends at different radii predicted by the model to the data
"""
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import BoundaryNorm
import vice

from utils import binned_quantiles, get_bin_centers
from plotting import colored_text_legend, ONE_COLUMN_WIDTH
import paths
from multizone._globals import ZONE_WIDTH

OUTPUT_NAMES = ['yZ1-c11-mscale', 'yZ2-k16-mscale']
LABELS = [r'$y/Z_\odot=1$, C11+C15', r'$y/Z_\odot=2$, KL16+K18']

def main(style='paper', cmap='viridis_r'):
    plt.style.use(paths.styles / f'{style}.mplstyle')
    # Import MWM sample
    mwm_rgb = pd.read_csv(paths.data / 'MWM' / 'sample.csv')
    radius_bin_edges = np.arange(3, 15.1, 2)
    age_bin_edges = np.arange(-0.5, 11.6, 1)

    # Select stars near the midplane with good ages
    lowz_ages = mwm_rgb[(mwm_rgb['z_max'] < 0.5) & (mwm_rgb['good_age'])].copy()
    age_err_low = np.median(lowz_ages['age'] - lowz_ages['e_n_age'])
    age_err_high = np.median(lowz_ages['e_p_age'] - lowz_ages['age'])
    med_abund_err = lowz_ages['e_ce_h'].median()
    
    # Set up figure
    fig, axs = plt.subplots(
        2, 1,
        figsize=(ONE_COLUMN_WIDTH, 1.8*ONE_COLUMN_WIDTH),
        sharex=True, sharey=True,
        gridspec_kw={'hspace': 0.}
    )
    radial_cmap = plt.get_cmap(cmap)
    norm = BoundaryNorm(radius_bin_edges, radial_cmap.N)
    xlim = (0, 13)
    ylim = (-0.7, 0.8)

    for ax in axs:
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
        # Plot median age trends binned by radius
        for j in range(len(radius_bin_edges)-1):
            radius_bin = radius_bin_edges[j:j+2]
            mean_radius = np.mean(radius_bin)
            lowz_subset = lowz_ages[
                (lowz_ages['Rg'] >= radius_bin[0]) &
                (lowz_ages['Rg'] < radius_bin[1])
            ]
            age_medians = binned_quantiles(
                lowz_subset, 'ce_mg_corr', 'age',
                q=0.5, bin_edges=age_bin_edges, min_count=20, est_errors=True
            )
            ax.plot(
                *age_medians[:-1], '--', 
                color=radial_cmap(norm(mean_radius)), zorder=1,
                label=f'{int(mean_radius)} kpc'
            )
        # Indicate median abundance errors
        ax.errorbar(
            3, -0.5, 
            xerr=[[age_err_low], [age_err_high]], 
            yerr=med_abund_err, 
            c='gray', capsize=0,
        )
    fig.colorbar(
        pcm, 
        ax=axs, 
        orientation='horizontal', 
        pad=0.1,
        label='Number of stars'
    )

    # Plot multizone evolution
    for i, output_name in enumerate(OUTPUT_NAMES):
        for radius in get_bin_centers(radius_bin_edges):
            zone = int(radius / ZONE_WIDTH)
            zone_path = str(
                paths.multizone / output_name / 'diskmodel.vice' / ('zone%d' % zone)
            )
            hist = vice.history(zone_path)
            axs[i].plot(
                hist['lookback'], hist['[ce/mg]'], 
                color=radial_cmap(norm(radius)), ls='-'
            )
            axs[i].set_title(LABELS[i], y=0.95, pad=0, va='top')

    axs[0].set_xlim(xlim)
    axs[0].set_ylim(ylim)

    axs[0].yaxis.set_major_locator(MultipleLocator(0.5))
    axs[0].yaxis.set_minor_locator(MultipleLocator(0.1))
    axs[0].xaxis.set_major_locator(MultipleLocator(5))
    axs[0].xaxis.set_minor_locator(MultipleLocator(1))

    axs[0].set_ylabel('[Ce/Mg]')
    axs[1].set_ylabel('[Ce/Mg]')
    axs[1].set_xlabel('Age [Gyr]')

    for ax in axs:
        handles, labels = ax.get_legend_handles_labels()
        leg = colored_text_legend(ax, invert=True, loc='center right')
    fig.suptitle(r'Models with $M_{\rm AGB}\times0.5$', y=0.92)

    plt.savefig(paths.figures / 'model_radius_trends')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Plot [Ce/Mg]-age trends at many radii predicted by models.'
    )
    parser.add_argument('--style',
        choices=('paper', 'poster'),
        default='paper',
        help='Plot style to use (default: paper).'
    )
    parser.add_argument('--cmap',
        default='viridis_r',
        help='Colormap to use for radial dimension.'
    )
    args = parser.parse_args()
    main(**vars(args))
