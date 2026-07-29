"""
This script plots the distribution of stellar ages, metallicities, and 
guiding radii for the full sample.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from utils import import_sample
from plotting import ONE_COLUMN_WIDTH, colored_text_legend
from colormaps import paultol
import paths

def main(style='paper'):
    sample = import_sample(good_ages=False)
    sample_ages = sample[sample['good_age']]
    plt.style.use(paths.styles / f'{style}.mplstyle')
    fig, axs = plt.subplots(
        3, 
        figsize=(ONE_COLUMN_WIDTH, 1.5*ONE_COLUMN_WIDTH), 
        constrained_layout=True
    )
    good_age_color = paultol.vibrant.colors[3]
    rwidth = 1.0
    # First panel: age distributions
    age_bins = np.arange(0, 14.5, 0.5)
    axs[0].hist(
        sample['age'], 
        bins=age_bins, 
        color='k', 
        histtype='step', 
        label='Full sample'
    )
    axs[0].hist(
        sample_ages['age'], 
        bins=age_bins, 
        color=good_age_color, 
        histtype='bar', 
        rwidth=rwidth,
        label='Good ages'
    )
    axs[0].set_xlabel('Age [Gyr]')
    axs[0].set_xlim((min(age_bins), max(age_bins)))
    axs[0].xaxis.set_major_locator(MultipleLocator(5))
    axs[0].xaxis.set_minor_locator(MultipleLocator(1))
    colored_text_legend(axs[0], loc='upper right')
    # Second panel: [Fe/H] distributions
    feh_bins = np.arange(-1.5, 0.55, 0.05)
    axs[1].hist(
        sample['fe_h'], 
        bins=feh_bins, 
        color='k',
        histtype='step'
    )
    axs[1].hist(
        sample_ages['fe_h'], 
        bins=feh_bins, 
        color=good_age_color, 
        histtype='bar',
        rwidth=rwidth
    )
    axs[1].set_xlabel('[Fe/H]')
    axs[1].set_xlim((min(feh_bins), max(feh_bins)))
    axs[1].xaxis.set_major_locator(MultipleLocator(0.5))
    axs[1].xaxis.set_minor_locator(MultipleLocator(0.1))
    # Third panel: guiding radius
    Rg_bins = np.arange(0, 20.5, 0.5)
    axs[2].hist(
        sample['Rg'], 
        bins=Rg_bins, 
        color='k', 
        histtype='step'
    )
    axs[2].hist(
        sample_ages['Rg'], 
        bins=Rg_bins, 
        color=good_age_color, 
        histtype='bar',
        rwidth=rwidth
    )
    axs[2].set_xlabel('Guiding radius [kpc]')
    axs[2].set_xlim((min(Rg_bins), max(Rg_bins)))
    axs[2].set_ylim((0, 11000))
    axs[2].xaxis.set_major_locator(MultipleLocator(5))
    axs[2].xaxis.set_minor_locator(MultipleLocator(1))
    for ax in axs:
        ax.yaxis.set_major_locator(MultipleLocator(5000))
        ax.yaxis.set_minor_locator(MultipleLocator(1000))
        ax.set_ylabel(r'$N$', rotation='horizontal', labelpad=4)
        ax.set_ylim((0, 11000))
    plt.savefig(paths.figures / 'sample_distributions')
    plt.close()


if __name__ == '__main__':
    main()
