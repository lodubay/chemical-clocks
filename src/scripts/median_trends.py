"""
Plot median [Ce/Mg] vs [Mg/H] in different Galactic regions and for
high- and low-alpha populations.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from utils import alpha_cut, binned_quantiles, sample_rows
from plotting import colored_text_legend
from _globals import TWO_COLUMN_WIDTH
from colormaps import paultol
import paths

RBINS = [(3, 5), (5, 7), (7, 9), (9, 11), (11, 13)] # left to right
ZBINS = [(1, 2), (0.5, 1), (0, 0.5)] # top to bottom
ALPHA_BUFFER = 0.02 # dex, buffer around the [Mg/Fe] dividing line
# SAMPLE_FRACTION = 0.25 # fraction of stars to plot in each panel
SAMPLE_SIZE = 1000 # number of stars to plot in each panel, randomly sampled

def main(style='paper'):
    plt.style.use(paths.styles / f'{style}.mplstyle')
    # Import MWM sample
    mwm_rgb = pd.read_csv(paths.data / 'MWM' / 'MWM_RGB.csv')
    # Divide by low/high alpha
    mwm_rgb['low_alpha'] = mwm_rgb['mg_fe'] < alpha_cut(mwm_rgb['fe_h']) - ALPHA_BUFFER
    mwm_rgb['high_alpha'] = mwm_rgb['mg_fe'] > alpha_cut(mwm_rgb['fe_h']) + ALPHA_BUFFER
    # Local sample for comparison
    local_sample = mwm_rgb[
        (mwm_rgb['Rg'] >= 7) &
        (mwm_rgb['Rg'] < 9) &
        (mwm_rgb['z_max'] < 0.5)
    ]
    mg_bin_edges = np.arange(-0.75, 0.76, 0.1)
    local_low_alpha_medians = binned_quantiles(
        local_sample[local_sample['low_alpha']], 'ce_mg_corr', 'mg_h', 
        q=0.5, bin_edges=mg_bin_edges, min_count=10
    )
    local_high_alpha_medians = binned_quantiles(
        local_sample[local_sample['high_alpha']], 'ce_mg_corr', 'mg_h', 
        q=0.5, bin_edges=mg_bin_edges, min_count=10
    )

    fig, axs = plt.subplots(
        len(ZBINS), len(RBINS),
        figsize=(TWO_COLUMN_WIDTH, 0.6*TWO_COLUMN_WIDTH), 
        sharex=True, sharey=True,
        gridspec_kw={'hspace': 0, 'wspace': 0},
    )
    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95)
    # scatterplot style arguments
    kwargs = dict(s=1, marker='.', rasterized=True, edgecolor='none')
    high_alpha_color = paultol.highcontrast.colors[2]
    low_alpha_color = paultol.highcontrast.colors[0]

    for i, row in enumerate(axs):
        zlim = ZBINS[i]
        for j, ax in enumerate(row):
            rlim = RBINS[j]
            subset = mwm_rgb[
                (mwm_rgb['Rg'] >= rlim[0]) &
                (mwm_rgb['Rg'] < rlim[1]) &
                (mwm_rgb['z_max'] >= zlim[0]) &
                (mwm_rgb['z_max'] < zlim[1])
            ]
            low_alpha = subset[subset['low_alpha']]
            high_alpha = subset[subset['high_alpha']]
            # Scatter plot random sample of points
            # sample = sample_rows(subset, int(SAMPLE_FRACTION * subset.shape[0]))
            sample = sample_rows(subset, SAMPLE_SIZE)
            low_alpha_sample = sample[sample['low_alpha']]
            ax.scatter(
                low_alpha_sample['mg_h'], low_alpha_sample['ce_mg_corr'], 
                c=low_alpha_color, **kwargs
            )
            high_alpha_sample = sample[sample['high_alpha']]
            ax.scatter(
                high_alpha_sample['mg_h'], high_alpha_sample['ce_mg_corr'], 
                c=high_alpha_color, **kwargs
            )
            # Plot local trends for comparison
            ax.plot(
                *local_low_alpha_medians, 
                linestyle='--', color=low_alpha_color,
            )
            ax.plot(
                *local_high_alpha_medians, 
                linestyle='--', color=high_alpha_color,
            )
            # Plot median trends
            if low_alpha.shape[0] >= 100:
                ax.plot(
                    *binned_quantiles(
                        low_alpha, 'ce_mg_corr', 'mg_h', 
                        q=0.5, bin_edges=mg_bin_edges, min_count=10
                    ), 
                    '.-', color=low_alpha_color, label='High-Ia'
                )
            if high_alpha.shape[0] >= 100:
                ax.plot(
                    *binned_quantiles(
                        high_alpha, 'ce_mg_corr', 'mg_h', 
                        q=0.5, bin_edges=mg_bin_edges, min_count=10
                    ), 
                    '.-', color=high_alpha_color, label='Low-Ia'
                )
    # Indicate median abundance errors
    axs[0,0].errorbar(
        0.4, -0.5, 
        xerr=mwm_rgb['e_mg_h'].median(), 
        yerr=mwm_rgb['e_ce_mg'].median(), 
        c='gray', capsize=0, elinewidth=1,
    )

    # Format axes
    axs[0,0].set_xlim((-0.8, 0.6))
    axs[0,0].set_ylim((-0.7, 0.9))
    axs[0,0].xaxis.set_major_locator(MultipleLocator(0.5))
    axs[0,0].xaxis.set_minor_locator(MultipleLocator(0.1))
    axs[0,0].yaxis.set_major_locator(MultipleLocator(0.5))
    axs[0,0].yaxis.set_minor_locator(MultipleLocator(0.1))
    for ax in axs[-1,:]:
        ax.set_xlabel('[Mg/H]')
    for i, ax in enumerate(axs[0,:]):
        ax.set_title(r'$%s\leq R_{\rm guide}<%s$ kpc' % RBINS[i], fontsize=8)
    for ax in axs[:,0]:
        ax.set_ylabel(r'[Ce/Mg]$_{\rm corr}$')
    for i, ax in enumerate(axs[:,-1]):
        ax.yaxis.set_label_position('right')
        ax.set_ylabel(
            r'$%s\leq z_{\rm max}<%s$ kpc' % ZBINS[i], 
            fontsize=8, labelpad=6
        )
    # Text-only lengend
    leg = colored_text_legend(axs[0,0], loc='upper right')

    plt.savefig(paths.figures / 'median_trends')


if __name__ == '__main__':
    main()
