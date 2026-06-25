"""
This script plots the MWM [Ce/Mg]-[Mg/H] distribution color-coded by age.
"""
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm, BoundaryNorm
from matplotlib.ticker import MultipleLocator

from utils import import_sample
from plotting import ONE_COLUMN_WIDTH, ABUNDANCE_COLORMAP, AGE_COLORMAP
import paths

def main(style='paper'):
    mwm_sample = import_sample(good_ages=True)
    plt.style.use(paths.styles / f'{style}.mplstyle')
    savedir = {
        'paper': paths.figures,
        'presentation': paths.extra/'presentation'
    }[style]
    savedir.mkdir(exist_ok=True)
    fig, axs = plt.subplots(
        2, 
        figsize=(ONE_COLUMN_WIDTH, 1.5*ONE_COLUMN_WIDTH), 
        sharey=True,
        gridspec_kw={'hspace': 0.25, 'top': 0.95}
    )

    # First panel: [Ce/Mg] vs [Mg/H], color-coded by median age
    cmap = plt.get_cmap(AGE_COLORMAP)
    norm = BoundaryNorm(np.arange(0, 11, 1), cmap.N, extend='max')
    xlim = (-0.7, 0.5)
    ylim = (-0.9, 0.9)
    axs[0].scatter(
        mwm_sample['mg_h'], mwm_sample['ce_mg_corr'], 
        c=mwm_sample['age'], cmap=cmap, norm=norm,
        s=1, rasterized=True, edgecolors='none', marker='o', zorder=0
    )
    pc, contours = hexbin_contours(
        axs[0], mwm_sample['mg_h'], mwm_sample['ce_mg_corr'], mwm_sample['age'],
        gridsize=30, extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
        cmap=cmap, norm=norm, mincnt=10, contours=4,
    )
    fig.colorbar(pc, ax=axs[0], label='Age [Gyr]')
    # Indicate median abundance errors
    axs[0].errorbar(
        0.3, 0.7, 
        xerr=mwm_sample['e_mg_h'].median(), 
        yerr=mwm_sample['e_ce_mg'].median(), 
        c='k', capsize=0, elinewidth=1,
    )
    axs[0].set_xlabel('[Mg/H]')
    axs[0].set_ylabel('[Ce/Mg]')
    axs[0].set_xlim(xlim)
    axs[0].set_ylim(ylim)
    axs[0].xaxis.set_major_locator(MultipleLocator(0.5))
    axs[0].xaxis.set_minor_locator(MultipleLocator(0.1))
    axs[0].yaxis.set_major_locator(MultipleLocator(0.5))
    axs[0].yaxis.set_minor_locator(MultipleLocator(0.1))

    # Second panel: [Ce/Mg] vs age, color-coded by [Fe/H]
    cmap = plt.get_cmap(ABUNDANCE_COLORMAP)
    norm = BoundaryNorm(np.arange(-0.6, 0.41, 0.1), cmap.N, extend='both')
    xlim = (0, 12)
    axs[1].scatter(
        mwm_sample['age'], mwm_sample['ce_mg_corr'], 
        c=mwm_sample['fe_h_corr'], cmap=cmap, norm=norm,
        s=1, rasterized=True, edgecolors='none', marker='o', zorder=0
    )
    pc, contours = hexbin_contours(
        axs[1], mwm_sample['age'], mwm_sample['ce_mg_corr'], mwm_sample['fe_h_corr'],
        gridsize=30, extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
        cmap=cmap, norm=norm, mincnt=10, contours=4,
    )
    fig.colorbar(pc, ax=axs[1], label='[Fe/H]')
    # Indicate median abundance errors
    age_err_low = np.median(mwm_sample['age'] - mwm_sample['e_n_age'])
    age_err_high = np.median(mwm_sample['e_p_age'] - mwm_sample['age'])
    med_abund_err = mwm_sample['e_ce_h'].median()
    axs[1].errorbar(
        10, 0.7, 
        xerr=[[age_err_low], [age_err_high]], 
        yerr=med_abund_err, 
        c='k', capsize=0, elinewidth=1,
    )
    axs[1].set_xlabel('Age [Gyr]')
    axs[1].set_ylabel('[Ce/Mg]')
    axs[1].set_xlim(xlim)
    axs[1].xaxis.set_major_locator(MultipleLocator(5))
    axs[1].xaxis.set_minor_locator(MultipleLocator(1))
    axs[1].yaxis.set_major_locator(MultipleLocator(0.5))
    axs[1].yaxis.set_minor_locator(MultipleLocator(0.1))

    plt.savefig(savedir / 'cemg_mgh_age')
    plt.close()


def hexbin_contours(
        ax, x, y, C, 
        Cfunc=np.median, 
        gridsize=100,
        extent=None, 
        cmap=None, 
        norm=None, 
        vmin=None,
        vmax=None,
        mincnt=0,
        contours=5, 
        contour_cmap='binary', 
        contour_lw=2,
    ):
    """
    Generate a hexbin plot with density contours.

    This is done by layering a number of hexbin plots on top of each other,
    each with a higher minimum count corresponding to the contour level.
    """
    # Config for all hexbin layers
    cfg = dict(
        x=x,
        y=y,
        C=C,
        reduce_C_function=Cfunc,
        gridsize=gridsize,
        extent=extent,
        cmap=cmap,
        norm=norm,
        vmin=vmin,
        vmax=vmax,
    )
    # Base layer - standard hexbin with color-coding
    pc = ax.hexbin(**cfg, linewidths=0.2, zorder=1, mincnt=mincnt)
    # Ensure consistent normalization for each hexbin layer
    if cfg['norm'] is None:
        cfg['vmin'] = pc.get_clim()[0]
        cfg['vmax'] = pc.get_clim()[1]
    # Divide contours evenly
    if isinstance(contours, int):
        # Invisible hexbin for density values
        density = ax.hexbin(x, y, gridsize=gridsize, extent=extent, zorder=0, alpha=0)
        contours = np.linspace(mincnt, density.get_clim()[1], contours+2, endpoint=True)[1:-1]
    # Get colors for contour outlines
    contour_norm = LogNorm(vmin=contours[0], vmax=contours[-1])
    contour_cmap = plt.get_cmap(contour_cmap)
    colors = contour_cmap(contour_norm(contours))
    for i, count in enumerate(contours):
        ax.hexbin(**cfg, edgecolors=colors[i], lw=contour_lw, zorder=i+1, mincnt=count)
        ax.hexbin(**cfg, lw=0.2, zorder=i+2, mincnt=count)
    return pc, contours


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Plot [Ce/Mg] vs [Mg/H] for halo stars.'
    )
    parser.add_argument('--style',
        choices=('paper', 'presentation'),
        default='paper',
        help='Plot style to use (default: paper).'
    )
    args = parser.parse_args()
    main(**vars(args))
