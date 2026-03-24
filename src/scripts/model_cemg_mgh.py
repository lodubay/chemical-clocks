"""
This script plots a Hayden-style [Ce/Mg]-[Mg/H] plot of a multizone model
compared to MWM data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MultipleLocator

from multizone_stars import MultizoneStars
from plotting import TWO_COLUMN_WIDTH, insert_colorbar_axes
from utils import plot_gas_abundance
from contours import plot_kde2D_contours
import paths

OUTPUT_NAME = 'yZ2-k16-mscale/diskmodel'
RBINS = [(3, 5), (7, 9), (11, 13)]
ZLIM = (0, 2)


def main(style='paper', cmap='Spectral_r'):
    # Import MWM sample
    sample = pd.read_csv(paths.data / 'MWM' / 'sample.csv')
    # Set up figure
    plt.style.use(paths.styles / f'{style}.mplstyle')
    fig, axs = plt.subplots(
        1, 3, 
        figsize=(TWO_COLUMN_WIDTH, 0.33*TWO_COLUMN_WIDTH),
        sharex=True, sharey=True,
        gridspec_kw={'wspace': 0.}
    )
    # cax = insert_colorbar_axes(fig, pad=0.02)
    age_bounds = np.arange(0, 12.1, 2)
    cmap = plt.get_cmap(cmap)
    cbar = fig.colorbar(
        ScalarMappable(
            BoundaryNorm(age_bounds, cmap.N, extend='max'), 
            cmap
        ), 
        ax=axs,
        # shrink=0.6, 
        aspect=15, 
        fraction=0.05, 
        pad=0.02,
        orientation='vertical',
        label='Age [Gyr]'
    )
    # Plot multizone output
    mzs = MultizoneStars.from_output(OUTPUT_NAME)
    mzs.model_uncertainty(sample, inplace=True)
    for i, rlim in enumerate(RBINS):
        ax = axs[i]
        ax.set_title(r'$%s\leq R_g<%s$ kpc' % rlim)
        mzs_subset = mzs.region(rlim, absz_lim=ZLIM)
        mzs_subset.scatter_plot(
            ax, '[mg/h]', '[ce/mg]', color='age',
            cmap=cbar.cmap, norm=cbar.norm,
            markersize=0.5,
        )
        plot_gas_abundance(ax, mzs_subset, '[mg/h]', '[ce/mg]', c='k', ls='--')
        # Plot MWM data contours
        sample_subset = sample[
            (sample['Rg'] >= rlim[0]) &
            (sample['Rg'] < rlim[1]) &
            (sample['z_max'] >= ZLIM[0]) &
            (sample['z_max'] < ZLIM[1])
        ]
        plot_kde2D_contours(ax, sample_subset, 'mg_h', 'ce_mg_corr')
    
    # Format axes
    axs[0].set_xlim((-0.9, 0.6))
    axs[0].set_ylim((-0.9, 0.7))
    for ax in axs:
        ax.set_xlabel('[Mg/H]')
    axs[0].set_ylabel('[Ce/Mg]', labelpad=-2)
    # Set x-axis ticks
    axs[0].xaxis.set_major_locator(MultipleLocator(0.5))
    axs[0].xaxis.set_minor_locator(MultipleLocator(0.1))
    # Set y-axis ticks
    axs[0].yaxis.set_major_locator(MultipleLocator(0.5))
    axs[0].yaxis.set_minor_locator(MultipleLocator(0.1))
    
    plt.savefig(paths.figures / 'model_cemg_mgh.pdf')
    plt.close()


if __name__ == '__main__':
    main()
