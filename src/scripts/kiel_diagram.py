"""
This script plots a Kiel diagram of the MWM sample.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

import paths
from _globals import TWO_COLUMN_WIDTH
from mwm_sample import LOGG_CUT, TEFF_CUT
from plotting import colored_text_legend, insert_colorbar_axes
from utils import good_ages
from colormaps import paultol


def main(style='paper'):
    mwm_rgb = pd.read_csv(paths.data / 'MWM' / 'MWM_RGB.csv')
    # mwm_rgb = good_ages(mwm_rgb)
    plt.style.use(paths.styles / f'{style}.mplstyle')
    fig, axs = plt.subplots(
        1, 2, 
        figsize=(0.8*TWO_COLUMN_WIDTH, 0.5*TWO_COLUMN_WIDTH),
        sharex=True, sharey=True,
        gridspec_kw={'wspace': 0.}
    )
    # First panel: compare full sample to sample with good ages
    axs[0].scatter(
        mwm_rgb['teff'], mwm_rgb['logg'],
        # c=mwm_rgb['ce_h'], cmap='viridis', vmin=-1.5, vmax=0.5,
        rasterized=True, s=1, marker='.', edgecolors='none', c='k', 
        label='Full sample'
    )
    # Indicate stars with good ages
    mwm_ages = good_ages(mwm_rgb)
    axs[0].scatter(
        mwm_ages['teff'], mwm_ages['logg'],
        rasterized=True, s=1, marker='.', edgecolors='none', 
        c=paultol.vibrant.colors[3], label='Good ages'
    )
    # Second panel: color-code full sample by [Ce/H]
    pc = axs[1].scatter(
        mwm_rgb['teff'], mwm_rgb['logg'],
        c=mwm_rgb['ce_h'], cmap='viridis', vmin=-1.5, vmax=0.5,
        rasterized=True, s=1, marker='.', edgecolors='none'
    )
    cax = insert_colorbar_axes(fig)
    fig.colorbar(
        pc, cax=cax, extend='both', label='[Ce/H]'
    )
    cax.yaxis.set_major_locator(MultipleLocator(0.5))
    cax.yaxis.set_minor_locator(MultipleLocator(0.1))
    axs[0].set_xlim((5400, 4000))
    # axs[0].xaxis.set_inverted(True)
    axs[0].set_ylim(LOGG_CUT)
    axs[0].yaxis.set_inverted(True)
    axs[0].xaxis.set_major_locator(MultipleLocator(500))
    axs[0].xaxis.set_minor_locator(MultipleLocator(100))
    axs[0].yaxis.set_major_locator(MultipleLocator(0.5))
    axs[0].yaxis.set_minor_locator(MultipleLocator(0.1))
    axs[0].set_xlabel(r'$T_{\rm eff}$ [K]')
    axs[1].set_xlabel(r'$T_{\rm eff}$ [K]')
    axs[0].set_ylabel(r'$\log g$')
    colored_text_legend(axs[0])
    plt.savefig(paths.figures / 'kiel_diagram')
    plt.close()


if __name__ == '__main__':
    main()
