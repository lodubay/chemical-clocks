"""
This script plots the MWM [Ce/Mg]-[Mg/H] distribution color-coded by age.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm, BoundaryNorm
from matplotlib.ticker import MultipleLocator

from _globals import ONE_COLUMN_WIDTH
from plotting import get_color_list
from utils import good_ages
import paths

XLIM = (-0.7, 0.5)
YLIM = (-0.8, 0.8)

def main(style='paper'):
    mwm_rgb = pd.read_csv(paths.data / 'MWM' / 'MWM_RGB.csv')
    mwm_rgb = good_ages(mwm_rgb)
    plt.style.use(paths.styles / f'{style}.mplstyle')
    fig, ax = plt.subplots(figsize=(ONE_COLUMN_WIDTH, 0.7*ONE_COLUMN_WIDTH))
    cmap = plt.get_cmap('Spectral_r')
    norm = BoundaryNorm(np.arange(0, 11, 1), cmap.N, extend='max')
    ax.scatter(
        mwm_rgb['mg_h'], mwm_rgb['ce_mg'], 
        c=mwm_rgb['age'], cmap=cmap, norm=norm,
        s=1, rasterized=True, edgecolors='none', marker='o', zorder=0
    )
    pc, contours = hexbin_contours(
        ax, mwm_rgb['mg_h'], mwm_rgb['ce_mg'], mwm_rgb['age'],
        gridsize=30, extent=[XLIM[0], XLIM[1], YLIM[0], YLIM[1]],
        cmap=cmap, norm=norm, mincnt=10, contours=4,
    )
    print(contours)
    fig.colorbar(pc, ax=ax, label='StarFlow Age [Gyr]')
    ax.set_xlabel('[Mg/H]')
    ax.set_ylabel('[Ce/Mg]')
    ax.set_xlim(XLIM)
    ax.set_ylim(YLIM)
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    plt.savefig(paths.figures / 'cemg_mgh_age')
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
    main()
