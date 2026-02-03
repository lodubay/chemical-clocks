"""
This script plots a Kiel diagram of the MWM sample.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

import paths
from _globals import ONE_COLUMN_WIDTH
from mwm_sample import LOGG_CUT, TEFF_CUT


def main(style='paper'):
    mwm_rgb = pd.read_csv(paths.data / 'MWM' / 'MWM_RGB.csv')
    plt.style.use(paths.styles / f'{style}.mplstyle')
    fig, ax = plt.subplots(figsize=(ONE_COLUMN_WIDTH, 1.8*ONE_COLUMN_WIDTH))
    pc = ax.scatter(
        mwm_rgb['teff'], mwm_rgb['logg'],
        c=mwm_rgb['m_h_atm'], cmap='viridis', vmin=-1.5,
        rasterized=True, s=0.5, marker='.', edgecolors='none'
    )
    fig.colorbar(
        pc, ax=ax, orientation='horizontal', extend='min', pad=0.1,
        label=r'[M/H]$_{\rm atm}$'
    )
    ax.set_xlim(TEFF_CUT)
    ax.xaxis.set_inverted(True)
    ax.set_ylim(LOGG_CUT)
    ax.yaxis.set_inverted(True)
    ax.set_xlabel(r'$T_{\rm eff}$ [K]')
    ax.set_ylabel(r'$\log g$')
    plt.savefig(paths.figures / 'kiel_diagram')
    plt.close()


if __name__ == '__main__':
    main()
