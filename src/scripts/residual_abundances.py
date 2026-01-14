"""
Plot metallicity-corrected residual Ce abundances as a function of
position in the Galaxy.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from utils import alpha_cut, binned_quantiles, sample_rows, good_ages
from plotting import colored_text_legend
from _globals import TWO_COLUMN_WIDTH
from colormaps import paultol
import paths

RBINS = [(3, 5), (5, 7), (7, 9), (9, 11), (11, 13)] # left to right
ZBINS = [(1, 2), (0.5, 1), (0, 0.5)] # top to bottom
ALPHA_BUFFER = 0.02 # dex, buffer around the [Mg/Fe] dividing line
# SAMPLE_FRACTION = 0.25 # fraction of stars to plot in each panel
SAMPLE_SIZE = 1000 # number of stars to plot in each panel, randomly selected

def main(style='paper'):
    plt.style.use(paths.styles / f'{style}.mplstyle')
    # Import MWM sample
    mwm_rgb = pd.read_csv(paths.data / 'MWM' / 'MWM_RGB.csv')
    # Divide by low/high alpha
    mwm_rgb['low_alpha'] = mwm_rgb['mg_fe'] < alpha_cut(mwm_rgb['fe_h']) - ALPHA_BUFFER
    mwm_rgb['high_alpha'] = mwm_rgb['mg_fe'] > alpha_cut(mwm_rgb['fe_h']) + ALPHA_BUFFER

    mg_bin_edges = np.arange(-0.75, 0.76, 0.1)
    age_bin_edges = np.arange(0.5, 11.6, 1)

    # Solar neighborhood sample
    local_low_alpha = mwm_rgb[
        (mwm_rgb['Rg'] >= 7) &
        (mwm_rgb['Rg'] < 9) &
        (mwm_rgb['z_max'] < 0.5) &
        (mwm_rgb['low_alpha'])
    ].copy()
    local_high_alpha = mwm_rgb[
        (mwm_rgb['Rg'] >= 7) &
        (mwm_rgb['Rg'] < 9) &
        (mwm_rgb['z_max'] < 0.5) &
        (mwm_rgb['high_alpha'])
    ].copy()
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
    # Calculate median trend with age (only stars with good ages)
    local_low_alpha_age_medians = binned_quantiles(
        good_ages(local_low_alpha), 'delta_ce_h', 'age',
        q=0.5, bin_edges=age_bin_edges, min_count=10
    )
    local_high_alpha_age_medians = binned_quantiles(
        good_ages(local_high_alpha), 'delta_ce_h', 'age',
        q=0.5, bin_edges=age_bin_edges, min_count=10
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
            low_alpha = subset[subset['low_alpha']].copy()
            high_alpha = subset[subset['high_alpha']].copy()
            # Select random sample of stars for scatter plot (good ages only)
            sample = sample_rows(
                good_ages(subset), 
                # int(SAMPLE_FRACTION * subset.shape[0])
                SAMPLE_SIZE
            )
            low_alpha_sample = sample[sample['low_alpha']]
            high_alpha_sample = sample[sample['high_alpha']]
            if low_alpha.shape[0] >= 100:
                # Calculate median trend with [Mg/H]
                low_alpha_medians = binned_quantiles(
                    low_alpha, 'ce_h_corr', 'mg_h', 
                    q=0.5, bin_edges=mg_bin_edges, min_count=10
                )
                # Calculate residual Ce abundance
                # (for all stars, including those with no/poor ages)
                low_alpha['delta_ce_h'] = low_alpha['ce_h_corr'] - np.interp(
                    low_alpha['mg_h'], *low_alpha_medians
                )
                # Scatter plot random sample of points
                ax.scatter(
                    low_alpha.loc[low_alpha_sample.index, 'age'], 
                    low_alpha.loc[low_alpha_sample.index, 'delta_ce_h'],
                    c=low_alpha_color, zorder=2, **kwargs
                )
                # Plot median trend with age
                age_medians = binned_quantiles(
                    good_ages(low_alpha), 'delta_ce_h', 'age',
                    q=0.5, bin_edges=age_bin_edges, min_count=10
                )
                ax.plot(
                    *age_medians, '.-', color=low_alpha_color, zorder=6,
                    label='High-Ia'
                )
            if high_alpha.shape[0] >= 100:
                # Calculate median trend with [Mg/H]
                high_alpha_medians = binned_quantiles(
                    high_alpha, 'ce_h_corr', 'mg_h', 
                    q=0.5, bin_edges=mg_bin_edges, min_count=10
                )
                # Calculate residual Ce abundance
                high_alpha['delta_ce_h'] = high_alpha['ce_h_corr'] - np.interp(
                    high_alpha['mg_h'], *high_alpha_medians
                )
                # Scatter plot random sample of points
                ax.scatter(
                    high_alpha.loc[high_alpha_sample.index, 'age'], 
                    high_alpha.loc[high_alpha_sample.index, 'delta_ce_h'],
                    c=high_alpha_color, zorder=1, **kwargs
                )
                # Plot median trend with age
                age_medians = binned_quantiles(
                    good_ages(high_alpha), 'delta_ce_h', 'age',
                    q=0.5, bin_edges=age_bin_edges, min_count=10
                )
                ax.plot(
                    *age_medians, '.-', color=high_alpha_color, zorder=5,
                    label='Low-Ia'
                )
            # Plot local low and high-alpha trends for comparison
            ax.plot(
                *local_low_alpha_age_medians, 
                linestyle='--', color=low_alpha_color, zorder=4,
            )
            ax.plot(
                *local_high_alpha_age_medians, 
                linestyle='--', color=high_alpha_color, zorder=3,
            )
            # Horizontal line for reference
            ax.plot([0, 12], [0, 0], linestyle=':', color='gray', zorder=0)
    # Indicate median abundance errors
    mwm_rgb_ages = good_ages(mwm_rgb)
    age_err_low = np.median(mwm_rgb_ages['age'] - mwm_rgb_ages['e_n_age'])
    age_err_high = np.median(mwm_rgb_ages['e_p_age'] - mwm_rgb_ages['age'])
    med_abund_err = mwm_rgb_ages['e_ce_h'].median()
    axs[0,0].errorbar(
        3, -0.4, 
        xerr=[[age_err_low], [age_err_high]], 
        yerr=med_abund_err, 
        c='gray', capsize=0, elinewidth=0.5,
    )

    # Format axes
    axs[0,0].set_xlim((0, 12))
    axs[0,0].set_ylim((-0.6, 0.6))
    axs[0,0].xaxis.set_major_locator(MultipleLocator(5))
    axs[0,0].xaxis.set_minor_locator(MultipleLocator(1))
    axs[0,0].yaxis.set_major_locator(MultipleLocator(0.5))
    axs[0,0].yaxis.set_minor_locator(MultipleLocator(0.1))
    for ax in axs[-1,:]:
        ax.set_xlabel('Age [Gyr]')
    for i, ax in enumerate(axs[0,:]):
        ax.set_title(r'$%s\leq R_{\rm guide}<%s$ kpc' % RBINS[i], fontsize=8)
    for ax in axs[:,0]:
        ax.set_ylabel(r'$\Delta$[Ce/H]')
    for i, ax in enumerate(axs[:,-1]):
        ax.yaxis.set_label_position('right')
        ax.set_ylabel(
            r'$%s\leq z_{\rm max}<%s$ kpc' % ZBINS[i], 
            fontsize=8, labelpad=6
        )
    # Text-only lengend
    leg = colored_text_legend(axs[0,0], loc='upper left')

    plt.savefig(paths.figures / 'residual_abundances')


if __name__ == '__main__':
    main()
