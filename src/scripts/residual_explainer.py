"""
Plot explaining the calculation of the residual abundance Delta [Ce/H].
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from utils import binned_quantiles, apply_alpha_cut, sample_rows, good_ages, colored_text_legend
from colormaps import paultol
from _globals import ONE_COLUMN_WIDTH
import paths

SAMPLE_FRACTION = 0.25

def main(style='paper'):
    # Import MWM sample
    mwm_rgb = pd.read_csv(paths.data / 'MWM' / 'MWM_RGB.csv')
    # Divide by low/high alpha
    mwm_rgb = apply_alpha_cut(mwm_rgb)

    mg_bin_edges = np.arange(-0.75, 0.76, 0.1)
    age_bin_edges = np.arange(0.5, 11.6, 1)

    # Solar neighborhood sample
    mwm_rgb_local = mwm_rgb[
        (mwm_rgb['Rg'] >= 7) &
        (mwm_rgb['Rg'] < 9) &
        (mwm_rgb['z_max'] < 0.5)
    ].copy()
    local_low_alpha = mwm_rgb_local[mwm_rgb_local['low_alpha']].copy()
    local_high_alpha = mwm_rgb_local[mwm_rgb_local['high_alpha']].copy()
    # Calculate median trend with [Mg/H]
    local_low_alpha_medians = binned_quantiles(
        local_low_alpha, 'ce_h_corr', 'mg_h', 
        q=0.5, bin_edges=mg_bin_edges, min_count=10
    )
    local_high_alpha_medians = binned_quantiles(
        local_high_alpha, 'ce_h_corr', 'mg_h', 
        q=0.5, bin_edges=mg_bin_edges, min_count=10
    )
    # Calculate residual Ce abundance
    local_low_alpha['delta_ce_h'] = local_low_alpha['ce_h_corr'] - np.interp(
        local_low_alpha['mg_h'], *local_low_alpha_medians
    )
    local_high_alpha['delta_ce_h'] = local_high_alpha['ce_h_corr'] - np.interp(
        local_high_alpha['mg_h'], *local_high_alpha_medians
    )

    # Set up figure
    plt.style.use(paths.styles / f'{style}.mplstyle')
    fig, axs = plt.subplots(2, 2, 
        figsize=(ONE_COLUMN_WIDTH, ONE_COLUMN_WIDTH),
        sharex='col', sharey='row', 
        gridspec_kw={'hspace': 0, 'wspace': 0}
    )
    # scatterplot style arguments
    kwargs = dict(s=1, marker='.', rasterized=True, edgecolor='none')
    high_alpha_color = paultol.highcontrast.colors[2]
    low_alpha_color = paultol.highcontrast.colors[0]

    # Scatter plot random sample of points
    sample = sample_rows(
        good_ages(mwm_rgb_local), 
        int(SAMPLE_FRACTION * mwm_rgb_local.shape[0])
    )
    # sample = sample_rows(subset, SAMPLE_SIZE)
    low_alpha_sample = local_low_alpha.loc[sample[sample['low_alpha']].index]
    high_alpha_sample = local_high_alpha.loc[sample[sample['high_alpha']].index]
    
    # Plot [Mg/H] vs [Ce/H]
    axs[0,0].scatter(
        high_alpha_sample['mg_h'], high_alpha_sample['ce_h_corr'], 
        c=high_alpha_color, **kwargs
    )
    axs[0,0].scatter(
        low_alpha_sample['mg_h'], low_alpha_sample['ce_h_corr'], 
        c=low_alpha_color, **kwargs
    )
    axs[0,0].plot(
        *local_low_alpha_medians, 
        '.-', color=low_alpha_color, label='High-Ia'
    )
    axs[0,0].plot(
        *local_high_alpha_medians, 
        '.-', color=high_alpha_color, label='Low-Ia'
    )

    # Plot [Ce/H] residuals vs [Mg/H]
    axs[1,0].scatter(
        low_alpha_sample['mg_h'], low_alpha_sample['delta_ce_h'],
        c=low_alpha_color, **kwargs
    )
    axs[1,0].scatter(
        high_alpha_sample['mg_h'], high_alpha_sample['delta_ce_h'],
        c=high_alpha_color, **kwargs
    )
    # Plot 1-sigma bands
    low_alpha_sigma_low = binned_quantiles(
        local_low_alpha, 'delta_ce_h', 'mg_h',
        q=0.16, bin_edges=mg_bin_edges, min_count=10
    )
    axs[1,0].plot(*low_alpha_sigma_low, '--', color=low_alpha_color)
    low_alpha_sigma_med = binned_quantiles(
        local_low_alpha, 'delta_ce_h', 'mg_h',
        q=0.5, bin_edges=mg_bin_edges, min_count=10
    )
    axs[1,0].plot(*low_alpha_sigma_med, '-', color=low_alpha_color)
    low_alpha_sigma_high = binned_quantiles(
        local_low_alpha, 'delta_ce_h', 'mg_h',
        q=0.84, bin_edges=mg_bin_edges, min_count=10
    )
    axs[1,0].plot(*low_alpha_sigma_high, '--', color=low_alpha_color)
    high_alpha_sigma_low = binned_quantiles(
        local_high_alpha, 'delta_ce_h', 'mg_h',
        q=0.16, bin_edges=mg_bin_edges, min_count=10
    )
    axs[1,0].plot(*high_alpha_sigma_low, '--', color=high_alpha_color)
    high_alpha_sigma_med = binned_quantiles(
        local_high_alpha, 'delta_ce_h', 'mg_h',
        q=0.5, bin_edges=mg_bin_edges, min_count=10
    )
    axs[1,0].plot(*high_alpha_sigma_med, '-', color=high_alpha_color)
    high_alpha_sigma_high = binned_quantiles(
        local_high_alpha, 'delta_ce_h', 'mg_h',
        q=0.84, bin_edges=mg_bin_edges, min_count=10
    )
    axs[1,0].plot(*high_alpha_sigma_high, '--', color=high_alpha_color)

    # Plot [Ce/H] vs age
    axs[0,1].scatter(
        low_alpha_sample['age'], low_alpha_sample['ce_h_corr'],
        c=low_alpha_color, **kwargs
    )
    axs[0,1].scatter(
        high_alpha_sample['age'], high_alpha_sample['ce_h_corr'],
        c=high_alpha_color, **kwargs
    )
    # Plot median trends with age
    low_alpha_age_medians = binned_quantiles(
        good_ages(local_low_alpha), 'ce_h_corr', 'age',
        q=0.5, bin_edges=age_bin_edges, min_count=10
    )
    axs[0,1].plot(
        *low_alpha_age_medians, '.-', color=low_alpha_color, zorder=6,
        label='High-Ia'
    )
    high_alpha_age_medians = binned_quantiles(
        good_ages(local_high_alpha), 'ce_h_corr', 'age',
        q=0.5, bin_edges=age_bin_edges, min_count=10
    )
    axs[0,1].plot(
        *high_alpha_age_medians, '.-', color=high_alpha_color, zorder=6,
        label='Low-Ia'
    )

    # Plot [Ce/H] residuals vs age
    axs[1,1].scatter(
        low_alpha_sample['age'], low_alpha_sample['delta_ce_h'],
        c=low_alpha_color, **kwargs
    )
    axs[1,1].scatter(
        high_alpha_sample['age'], high_alpha_sample['delta_ce_h'],
        c=high_alpha_color, **kwargs
    )
    # Plot median trends with age
    low_alpha_age_medians = binned_quantiles(
        good_ages(local_low_alpha), 'delta_ce_h', 'age',
        q=0.5, bin_edges=age_bin_edges, min_count=10
    )
    axs[1,1].plot(
        *low_alpha_age_medians, '.-', color=low_alpha_color, zorder=6,
        label='High-Ia'
    )
    high_alpha_age_medians = binned_quantiles(
        good_ages(local_high_alpha), 'delta_ce_h', 'age',
        q=0.5, bin_edges=age_bin_edges, min_count=10
    )
    axs[1,1].plot(
        *high_alpha_age_medians, '.-', color=high_alpha_color, zorder=6,
        label='Low-Ia'
    )

    # Axes labels
    axs[0,0].set_ylabel(r'[Ce/H]$_{\rm corr}$')
    axs[1,0].set_ylabel(r'$\Delta$[Ce/H]$_{\rm corr}$')
    axs[1,0].set_xlabel('[Mg/H]')
    axs[1,1].set_xlabel('Age [Gyr]')

    # Axes limits
    axs[0,0].set_xlim((-0.8, 0.6))
    axs[0,0].set_ylim((-0.9, 0.8))
    axs[1,0].set_ylim((-0.7, 0.7))
    axs[0,1].set_xlim((-1, 12))

    # Axes ticks
    axs[0,0].xaxis.set_major_locator(MultipleLocator(0.5))
    axs[0,0].xaxis.set_minor_locator(MultipleLocator(0.1))
    axs[0,0].yaxis.set_major_locator(MultipleLocator(0.5))
    axs[0,0].yaxis.set_minor_locator(MultipleLocator(0.1))
    axs[1,0].yaxis.set_major_locator(MultipleLocator(0.5))
    axs[1,0].yaxis.set_minor_locator(MultipleLocator(0.1))
    axs[0,1].xaxis.set_major_locator(MultipleLocator(5))
    axs[0,1].xaxis.set_minor_locator(MultipleLocator(1))

    leg = colored_text_legend(axs[0,0], loc='upper left')

    plt.savefig(paths.figures / 'residual_explainer')


if __name__ == '__main__':
    main()
