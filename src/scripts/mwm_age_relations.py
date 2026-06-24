"""
Plot [Mg/H], [Fe/Mg], and [Ce/Mg] vs age.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from utils import import_sample
import paths

RLIM = (7, 9)
ZLIM = (0, 2)

def main(style='poster'):
    mwm_sample = import_sample(good_ages=True)
    local_sample = mwm_sample[
        (mwm_sample['Rg'] >= RLIM[0]) & 
        (mwm_sample['Rg'] < RLIM[1]) & 
        (mwm_sample['z_max'] >= ZLIM[0]) &
        (mwm_sample['z_max'] < ZLIM[1])
    ].copy()
    print(local_sample.shape[0])
    plt.style.use(paths.styles / f'{style}.mplstyle')
    fig, axs = plt.subplots(
        1, 3, figsize=(8, 2.5), sharex=True, constrained_layout=True
    )
    cols = ['mg_h', 'mg_fe', 'ce_mg']
    labels = ['[Mg/H]', '[Mg/Fe]', '[Ce/Mg]']
    xlim = (0, 11)
    ylims = [(-1.1, 0.6), (-0.2, 0.4), (-0.8, 0.8)]
    for ax, col, label, ylim in zip(axs, cols, labels, ylims):
        # ax.scatter(
        #     local_sample['age'], local_sample[col], 
        #     rasterized=True, marker='.', c='k', s=1, edgecolors='none',
        # )
        ax.hexbin(
            local_sample['age'], local_sample[col],
            gridsize=30, cmap='binary', linewidths=0.2,
            extent=[xlim[0], xlim[1], ylim[0], ylim[1]]
        )
        ax.set_xlabel('Age [Gyr]')
        ax.set_ylabel(label)
        ax.set_ylim(ylim)
    axs[0].set_xlim(xlim)
    axs[0].xaxis.set_major_locator(MultipleLocator(5))
    axs[0].xaxis.set_minor_locator(MultipleLocator(1))
    axs[0].yaxis.set_major_locator(MultipleLocator(0.5))
    axs[0].yaxis.set_minor_locator(MultipleLocator(0.1))
    axs[1].yaxis.set_major_locator(MultipleLocator(0.2))
    axs[1].yaxis.set_minor_locator(MultipleLocator(0.05))
    axs[2].yaxis.set_major_locator(MultipleLocator(0.5))
    axs[2].yaxis.set_minor_locator(MultipleLocator(0.1))
    plt.savefig(paths.figures / 'mwm_age_relations')
    plt.close()


if __name__ == '__main__':
    main()
