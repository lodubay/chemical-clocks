"""
This script plots a comparison of log(g)-calibrated abundances against the
default calibrated abundances in MWM DR19. Calibrations by Tawny Sit.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MultipleLocator
from matplotlib.cm import ScalarMappable

from utils import get_bin_centers, binned_quantiles, import_sample, alpha_cut
from plotting import truncate_colormap, insert_colorbar_axes, ONE_COLUMN_WIDTH
import paths

AXES_LIM = {
    'mg_h': (-0.8, 0.5),
    'ce_mg': (-0.8, 0.8),
    'fe_mg': (-0.4, 0.2)
}


def main(style='paper', cmap_name='autumn'):
    # Initialize grid of log(g), [Mg/H] values
    MgH_bin_edges = np.round(np.linspace(-0.75, 0.45, 13, endpoint=True), 2)
    MgH_bin_centers = get_bin_centers(MgH_bin_edges)
    logg_bin_edges = np.linspace(0, 3.5, 8, endpoint=True)
    logg_bin_centers = get_bin_centers(logg_bin_edges)

    # Load MWM data
    calib_data = import_sample(good_ages=False)

    # Plot
    plt.style.use(paths.styles / f'{style}.mplstyle')
    fig, axs = plt.subplots(
        2, 2, 
        sharex=True, sharey='row',
        figsize=(ONE_COLUMN_WIDTH, 1.2 * ONE_COLUMN_WIDTH),
        gridspec_kw={'hspace': 0, 'wspace': 0}
    )
    cmap = truncate_colormap(plt.get_cmap(cmap_name), minval=0.1, maxval=0.9)
    norm = BoundaryNorm(logg_bin_edges[2:], cmap.N)
    hexbin_kw = dict(gridsize=20, linewidths=0.2, cmap='binary')
    ms = 2
    for i, ycol in enumerate(['ce_mg', 'ce_mg_corr', 'fe_mg', 'fe_mg_corr']):
        ax = axs.flatten()[i]
        # 2D hexbin of all stars
        ax.hexbin(
            calib_data['mg_h'], calib_data[ycol], 
            extent=[*AXES_LIM['mg_h'], *AXES_LIM[ycol.replace('_corr', '')]], 
            **hexbin_kw
        )
        # Plot median trends binned by log(g)
        # Ignore log(g) < 1 as that is outside the sample cut
        for j in range(2, logg_bin_centers.shape[0]):
            logg_lim = logg_bin_edges[j:j+2]
            logg_center = np.sum(logg_bin_edges[j:j+2])/2
            logg_subset = calib_data[
                (calib_data['logg'] > logg_lim[0]) & 
                (calib_data['logg'] < logg_lim[1])
            ]
            # High-alpha median trends, binned by [Mg/H]
            low_ia_uncorr_med = binned_quantiles(
                logg_subset[logg_subset['low_ia']], ycol, 'mg_h', 
                q=0.5, bin_edges=MgH_bin_edges, min_count=10, est_errors=True
            )
            ax.errorbar(
                *low_ia_uncorr_med, 
                fmt='o--', markersize=ms, 
                capsize=0,
                color=cmap(norm(logg_center)),
                zorder=10-j
            )
            # Low-alpha median trends, binned by [Mg/H]
            high_ia_uncorr_med = binned_quantiles(
                logg_subset[logg_subset['high_ia']], ycol, 'mg_h', 
                q=0.5, bin_edges=MgH_bin_edges, min_count=10, est_errors=True
            )
            ax.errorbar(
                *high_ia_uncorr_med, 
                fmt='s-', markersize=ms,
                capsize=0,
                color=cmap(norm(logg_center)),
                zorder=10-j
            )
    # Indicate low- and high-Ia cut
    mgh_arr = np.arange(-1, 0.51, 0.01)
    # Convert cut from [Fe/H] to [Mg/H]
    alpha_cut_slope = 0.13
    alpha_cut_mgh_scale = 1 - alpha_cut_slope
    for ax in axs[1]:
        ax.plot(mgh_arr, alpha_cut(mgh_arr) / alpha_cut_mgh_scale, 'k--', lw=0.5)
    # Label populations
    axs[1,0].text(-0.7, 0.1, 'High-Ia')
    axs[1,0].text(0.05, -0.325, 'Low-Ia')
    # Add colorbar
    cax = insert_colorbar_axes(
        fig, 
        pad=0.06, width=0.03, 
        orientation='horizontal'
    )
    fig.colorbar(
        ScalarMappable(norm, cmap), 
        cax=cax, 
        label=r'$\log(g)$', 
        orientation='horizontal'
    )
    cax.yaxis.set_inverted(True)
    # Plot labels
    axs[0,0].set_title('No offsets')
    axs[0,1].set_title('With offsets')
    axs[0,0].set_ylabel('[Ce/Mg]')
    axs[1,0].set_ylabel('[Fe/Mg]')
    axs[1,0].set_xlabel('[Mg/H]')
    axs[1,1].set_xlabel('[Mg/H]')
    # Axes limits
    axs[0,0].set_xlim(AXES_LIM['mg_h'])
    axs[0,0].set_ylim(AXES_LIM['ce_mg'])
    axs[1,0].set_ylim(AXES_LIM['fe_mg'])
    # Axes tickers
    axs[0,0].xaxis.set_major_locator(MultipleLocator(0.5))
    axs[0,0].xaxis.set_minor_locator(MultipleLocator(0.1))
    axs[0,0].yaxis.set_major_locator(MultipleLocator(0.5))
    axs[0,0].yaxis.set_minor_locator(MultipleLocator(0.1))
    axs[1,0].yaxis.set_major_locator(MultipleLocator(0.2))
    axs[1,0].yaxis.set_minor_locator(MultipleLocator(0.05))
    plt.savefig(paths.figures / 'logg_calibrations')
    plt.close()


if __name__ == '__main__':
    main()
