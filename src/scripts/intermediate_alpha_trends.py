"""
Plot median [Ce/Mg] vs [Mg/H] in different Galactic regions and for
high- and low-alpha populations.
"""
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from utils import binned_quantiles, sample_rows, import_sample
from plotting import setup_hayden_plot, iterate_rz_bins, colored_text_legend
from colormaps import paultol
import paths

# SAMPLE_FRACTION = 0.25 # fraction of stars to plot in each panel
SAMPLE_SIZE = 1000 # number of stars to plot in each panel, randomly sampled
YCOL = 'ce_mg_corr'

def main(style='paper'):
    plt.style.use(paths.styles / f'{style}.mplstyle')
    savedir = {
        'paper': paths.figures,
        'presentation': paths.extra/'presentation'
    }[style]
    savedir.mkdir(exist_ok=True)
    # Import MWM sample
    mwm_rgb = import_sample(good_ages=False)
    # Intermediate-alpha stars
    # mwm_rgb['int_ia'] = (
    #     (mwm_rgb['mg_fe'] > mwm_rgb['fe_h'] * -0.3 + 0.05) &
    #     (mwm_rgb['mg_fe'] > 0.05) &
    #     (mwm_rgb['mg_fe'] < 0.2) &
    #     (mwm_rgb['fe_h'] > -0.5)
    # )
    mwm_rgb['int_ia'] = (~mwm_rgb['low_ia']) & (~mwm_rgb['high_ia'])
    # Local sample for comparison
    local_sample = mwm_rgb[
        (mwm_rgb['Rg'] >= 7) &
        (mwm_rgb['Rg'] < 9) &
        (mwm_rgb['z_max'] < 0.5)
    ]
    mg_bin_edges = np.arange(-0.75, 0.76, 0.1)
    local_high_ia_medians = binned_quantiles(
        local_sample[local_sample['high_ia']], YCOL, 'mg_h', 
        q=0.5, bin_edges=mg_bin_edges, min_count=10
    )
    local_int_ia_medians = binned_quantiles(
        local_sample[local_sample['int_ia']], YCOL, 'mg_h', 
        q=0.5, bin_edges=mg_bin_edges, min_count=10
    )
    local_low_ia_medians = binned_quantiles(
        local_sample[local_sample['low_ia']], YCOL, 'mg_h', 
        q=0.5, bin_edges=mg_bin_edges, min_count=10
    )

    fig, axs = setup_hayden_plot()
    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95)
    # scatterplot style arguments
    kwargs = dict(s=1, marker='.', rasterized=True, edgecolor='none')
    low_ia_color = paultol.highcontrast.colors[2]
    high_ia_color = paultol.highcontrast.colors[0]
    int_ia_color = paultol.highcontrast.colors[1]

    for i, j, zlim, rlim in iterate_rz_bins():
        ax = axs[i,j]
        subset = mwm_rgb[
            (mwm_rgb['Rg'] >= rlim[0]) &
            (mwm_rgb['Rg'] < rlim[1]) &
            (mwm_rgb['z_max'] >= zlim[0]) &
            (mwm_rgb['z_max'] < zlim[1])
        ]
        high_ia = subset[subset['high_ia']]
        low_ia = subset[subset['low_ia']]
        int_ia = subset[subset['int_ia']]
        # Scatter plot random sample of points
        # sample = sample_rows(subset, int(SAMPLE_FRACTION * subset.shape[0]))
        sample = sample_rows(subset, SAMPLE_SIZE)
        high_ia_sample = sample[sample['high_ia']]
        ax.scatter(
            high_ia_sample['mg_h'], high_ia_sample[YCOL], 
            c=high_ia_color, **kwargs
        )
        low_ia_sample = sample[sample['low_ia']]
        ax.scatter(
            low_ia_sample['mg_h'], low_ia_sample[YCOL], 
            c=low_ia_color, **kwargs
        )
        int_ia_sample = sample[sample['int_ia']]
        ax.scatter(
            int_ia_sample['mg_h'], int_ia_sample[YCOL], 
            c=int_ia_color, **kwargs
        )
        # Plot local trends for comparison
        ax.plot(
            *local_high_ia_medians, 
            linestyle='--', color=high_ia_color, label='High-Ia',
        )
        ax.plot(
            *local_int_ia_medians, 
            linestyle='--', color=int_ia_color, label='Int.-Ia',
        )
        ax.plot(
            *local_low_ia_medians, 
            linestyle='--', color=low_ia_color, label='Low-Ia',
        )
        # Plot median trends
        if high_ia.shape[0] >= 100:
            ax.errorbar(
                *binned_quantiles(
                    high_ia, YCOL, 'mg_h', 
                    q=0.5, bin_edges=mg_bin_edges, min_count=10, 
                    est_errors=True
                ), 
                fmt='s-', markersize=3, color=high_ia_color, capsize=0
            )
        if low_ia.shape[0] >= 100:
            ax.errorbar(
                *binned_quantiles(
                    low_ia, YCOL, 'mg_h', 
                    q=0.5, bin_edges=mg_bin_edges, min_count=10,
                    est_errors=True
                ), 
                fmt='o-', markersize=3, color=low_ia_color, capsize=0
            )
        if int_ia.shape[0] >= 100:
            ax.errorbar(
                *binned_quantiles(
                    int_ia, YCOL, 'mg_h', 
                    q=0.5, bin_edges=mg_bin_edges, min_count=10,
                    est_errors=True
                ), 
                fmt='^-', markersize=3, color=int_ia_color, capsize=0
            )
        # Indicate number of low- and high-Ia stars in region
        if style == 'paper':
            if i==j==0:
                sample_size_low_ia = r'$N=%s$' % low_ia.shape[0]
                sample_size_high_ia = r'$N=%s$' % high_ia.shape[0]
                sample_size_int_ia = r'$N=%s$' % int_ia.shape[0]
            else:
                sample_size_low_ia = str(low_ia.shape[0])
                sample_size_high_ia = str(high_ia.shape[0])
                sample_size_int_ia = str(int_ia.shape[0])
            ax.text(
                0.91, 0.91, sample_size_high_ia, 
                color=high_ia_color, 
                ha='right', va='top', transform=ax.transAxes,
            )
            ax.text(
                0.91, 0.79, sample_size_int_ia, 
                color=int_ia_color, 
                ha='right', va='top', transform=ax.transAxes,
            )
            ax.text(
                0.91, 0.67, sample_size_low_ia, 
                color=low_ia_color, 
                ha='right', va='top', transform=ax.transAxes,
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
    for ax in axs[:,0]:
        ax.set_ylabel('[Ce/Mg]', labelpad=-2)
    # Text-only lengend
    leg = colored_text_legend(axs[0,0], loc='upper left')

    # plt.savefig(savedir / 'median_trends_grid')
    plt.savefig(paths.extra / 'intermediate_alpha_trends')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Plot median trends in [Ce/Mg] vs [Mg/H] across the Galaxy.'
    )
    parser.add_argument('--style',
        choices=('paper', 'presentation'),
        default='paper',
        help='Plot style to use (default: "paper").'
    )
    args = parser.parse_args()
    main(**vars(args))
