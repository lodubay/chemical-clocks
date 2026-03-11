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
from plotting import setup_hayden_plot, iterate_rz_bins
from utils import plot_gas_abundance
from contours import plot_kde2D_contours
import paths

OUTPUT_NAME = 'agb-mscale/diskmodel'


def main(style='paper', cmap='Spectral_r'):
    # Import MWM sample
    sample = pd.read_csv(paths.data / 'MWM' / 'sample.csv')
    # Set up figure
    plt.style.use(paths.styles / f'{style}.mplstyle')
    fig, axs = setup_hayden_plot()
    # Enlarge for colorbar
    figsize = fig.get_size_inches()
    fig.set_size_inches((figsize[0], figsize[1] * 1.25))
    age_bounds = np.arange(0, 12.1, 2)
    cmap = plt.get_cmap(cmap)
    cbar = fig.colorbar(
        ScalarMappable(
            BoundaryNorm(age_bounds, cmap.N, extend='max'), 
            cmap
        ), 
        ax=axs,
        shrink=0.6, 
        aspect=30, 
        fraction=0.1, 
        pad=0.1,
        orientation='horizontal',
        label='Age [Gyr]'
    )
    # Plot multizone output
    mzs = MultizoneStars.from_output(OUTPUT_NAME)
    mzs.model_uncertainty(sample, inplace=True)
    for i, j, zlim, rlim in iterate_rz_bins():
        ax = axs[i,j]
        mzs_subset = mzs.region(rlim, zlim)
        mzs_subset.scatter_plot(
            ax, '[mg/h]', '[ce/mg]', color='age',
            cmap=cbar.cmap, norm=cbar.norm
        )
        plot_gas_abundance(ax, mzs_subset, '[mg/h]', '[ce/mg]', c='k', ls='--')
        # Plot MWM data contours
        sample_subset = sample[
            (sample['Rg'] >= rlim[0]) &
            (sample['Rg'] < rlim[1]) &
            (sample['z_max'] >= zlim[0]) &
            (sample['z_max'] < zlim[1])
        ]
        plot_kde2D_contours(ax, sample_subset, 'mg_h', 'ce_mg_corr')
    
    # Format axes
    axs[0,0].set_xlim((-0.8, 0.6))
    axs[0,0].set_ylim((-0.8, 0.8))
    for ax in axs[-1]:
        ax.set_xlabel('[Mg/H]')
    for ax in axs[:,0]:
        ax.set_ylabel('[Ce/Mg]', labelpad=-2)
    # Set x-axis ticks
    axs[0,0].xaxis.set_major_locator(MultipleLocator(0.5))
    axs[0,0].xaxis.set_minor_locator(MultipleLocator(0.1))
    # Set y-axis ticks
    axs[0,0].yaxis.set_major_locator(MultipleLocator(0.5))
    axs[0,0].yaxis.set_minor_locator(MultipleLocator(0.1))
    
    plt.savefig(paths.figures / 'model_cemg_mgh.pdf')
    plt.close()


if __name__ == '__main__':
    main()
