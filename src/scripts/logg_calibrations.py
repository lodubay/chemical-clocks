"""
This script plots a comparison of log(g)-calibrated abundances against the
default calibrated abundances in MWM DR19. Calibrations by Tawny Sit.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MultipleLocator
from scipy.interpolate import interpn

from utils import get_bin_centers, binned_quantiles, apply_alpha_cut, truncate_colormap
import paths
from _globals import ONE_COLUMN_WIDTH

AXES_LIM = {
    'mg_h': (-0.8, 0.5),
    'ce_mg': (-0.7, 0.7),
    'fe_mg': (-0.4, 0.2)
}


def main(style='paper', cmap_name='autumn'):
    # Initialize grid of log(g), [Mg/H] values
    MgH_bin_edges = np.round(np.linspace(-0.75, 0.45, 13, endpoint=True), 2)
    MgH_bin_centers = get_bin_centers(MgH_bin_edges)
    logg_bin_edges = np.linspace(0, 3.5, 8, endpoint=True)
    logg_bin_centers = get_bin_centers(logg_bin_edges)
    grid = (MgH_bin_centers, logg_bin_centers)
    # Load calibration grids
    fe_offsets = np.load(paths.data / 'MWM' / 'fe_offset_grid.npy')
    ce_offsets = np.load(paths.data / 'MWM' / 'ce_offset_grid.npy')

    # Load MWM data
    calib_data = pd.read_csv(paths.data / 'MWM' / 'MWM_RGB.csv')
    calib_data = calib_data[calib_data['snr'] > 100]
    calib_data = apply_alpha_cut(calib_data)

    # Interpolate & apply log(g) corrections
    feh_corr = np.empty(calib_data.shape[0])
    ceh_corr = np.empty(calib_data.shape[0])
    for i in range(calib_data.shape[0]):
        feh_corr[i] = apply_elem_offsets(
            calib_data['mg_h'].iloc[i], 
            calib_data['logg'].iloc[i], 
            calib_data['fe_h'].iloc[i], 
            fe_offsets,
            grid
        )
        ceh_corr[i] = apply_elem_offsets(
            calib_data['mg_h'].iloc[i], 
            calib_data['logg'].iloc[i], 
            calib_data['ce_h'].iloc[i], 
            ce_offsets,
            grid
        )
    
    calib_data['fe_h_corr'] = feh_corr
    calib_data['ce_h_corr'] = ceh_corr
    calib_data['fe_mg_corr'] = calib_data['fe_h_corr'] - calib_data['mg_h']
    calib_data['ce_mg_corr'] = calib_data['ce_h_corr'] - calib_data['mg_h']
    calib_data['fe_mg'] = -calib_data['mg_fe']

    # Plot
    plt.style.use(paths.styles / f'{style}.mplstyle')
    fig, axs = plt.subplots(
        2, 2, 
        sharex=True, sharey='row',
        figsize=(ONE_COLUMN_WIDTH, ONE_COLUMN_WIDTH),
        gridspec_kw={'hspace': 0, 'wspace': 0}
    )
    # high_alpha_cmap = plt.get_cmap('Reds')
    # low_alpha_cmap = plt.get_cmap('Blues')
    cmap = truncate_colormap(plt.get_cmap(cmap_name), minval=0., maxval=0.8)
    norm = BoundaryNorm(logg_bin_edges, cmap.N)
    hexbin_kw = dict(gridsize=20, linewidths=0.2, cmap='binary')
    ms = 3
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
            high_alpha_uncorr_med = binned_quantiles(
                logg_subset[logg_subset['high_alpha']], ycol, 'mg_h', 
                q=0.5, bin_edges=MgH_bin_edges, min_count=10
            )
            ax.plot(
                *high_alpha_uncorr_med, 
                'o--', ms=ms, 
                color=cmap(norm(logg_center)),
                zorder=10-j,
                # label=logg_center
            )
            # Low-alpha median trends, binned by [Mg/H]
            low_alpha_uncorr_med = binned_quantiles(
                logg_subset[logg_subset['low_alpha']], ycol, 'mg_h', 
                q=0.5, bin_edges=MgH_bin_edges, min_count=10
            )
            ax.plot(
                *low_alpha_uncorr_med, 
                's-', ms=ms,
                color=cmap(norm(logg_center)),
                zorder=10-j,
                label=logg_center
            )
    # Add colorbar
    plt.subplots_adjust(right=0.75, bottom=0.25)
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
    axs[0,1].legend(title=r'$\log(g)$', loc='upper left', bbox_to_anchor=(1, 1))
    plt.savefig(paths.figures / 'logg_calibrations')
    plt.close()


def apply_elem_offsets(mgh, logg, xh, offsets, grid):
    interp_point = np.array([mgh, logg])
    star_offset = interpn(grid, offsets, interp_point, bounds_error=False, fill_value=None)[0]
    corr_xh = xh + star_offset
    return corr_xh


if __name__ == '__main__':
    main()
