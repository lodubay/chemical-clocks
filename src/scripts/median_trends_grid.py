"""
Plot median [Ce/Mg] vs [Mg/H] in different Galactic regions and for
high- and low-alpha populations.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from utils import binned_quantiles, sample_rows
from plotting import setup_hayden_plot, iterate_rz_bins, colored_text_legend
from colormaps import paultol
import paths

# SAMPLE_FRACTION = 0.25 # fraction of stars to plot in each panel
SAMPLE_SIZE = 1000 # number of stars to plot in each panel, randomly sampled

def main(style='paper'):
    plt.style.use(paths.styles / f'{style}.mplstyle')
    # Import MWM sample
    mwm_rgb = pd.read_csv(paths.data / 'MWM' / 'sample.csv')
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

    fig, axs = setup_hayden_plot()
    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95)
    # scatterplot style arguments
    kwargs = dict(s=1, marker='.', rasterized=True, edgecolor='none')
    high_alpha_color = paultol.highcontrast.colors[2]
    low_alpha_color = paultol.highcontrast.colors[0]

    for i, j, zlim, rlim in iterate_rz_bins():
        ax = axs[i,j]
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
            linestyle='--', color=low_alpha_color, label='High-Ia',
        )
        ax.plot(
            *local_high_alpha_medians, 
            linestyle='--', color=high_alpha_color, label='Low-Ia',
        )
        # Plot median trends
        if low_alpha.shape[0] >= 100:
            ax.errorbar(
                *binned_quantiles(
                    low_alpha, 'ce_mg_corr', 'mg_h', 
                    q=0.5, bin_edges=mg_bin_edges, min_count=10, 
                    est_errors=True
                ), 
                fmt='s-', markersize=3, color=low_alpha_color, capsize=0
            )
        if high_alpha.shape[0] >= 100:
            ax.errorbar(
                *binned_quantiles(
                    high_alpha, 'ce_mg_corr', 'mg_h', 
                    q=0.5, bin_edges=mg_bin_edges, min_count=10,
                    est_errors=True
                ), 
                fmt='o-', markersize=3, color=high_alpha_color, capsize=0
            )
        # Indicate number of low- and high-Ia stars in region
        if i==j==0:
            sample_size_high_alpha = r'$N=%s$' % high_alpha.shape[0]
            sample_size_low_alpha = r'$N=%s$' % low_alpha.shape[0]
        else:
            sample_size_high_alpha = str(high_alpha.shape[0])
            sample_size_low_alpha = str(low_alpha.shape[0])
        ax.text(
            0.91, 0.91, sample_size_low_alpha, 
            color=low_alpha_color, 
            ha='right', va='top', transform=ax.transAxes,
        )
        ax.text(
            0.91, 0.79, sample_size_high_alpha, 
            color=high_alpha_color, 
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

    plt.savefig(paths.figures / 'median_trends_grid')


if __name__ == '__main__':
    main()
