"""
Plot the local [Ce/Mg]--age relation and fit a linear trend in bins
of metallicity.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MultipleLocator
from scipy import stats

from _globals import ONE_COLUMN_WIDTH
from utils import good_ages, apply_alpha_cut, sample_rows
import paths

MET_COL = 'fe_h_corr' # Column with metallicity values
MET_LABEL = r'[Fe/H]'
AGE_FIT_RANGE = (1, 8) # Range of ages to fit linear trend
SAMPLE_FRACTION = 1 # fraction of stars to plot in each panel
RLIM = (7, 9)
ZLIM = (0, 0.5)


def main(style='paper'):
    plt.style.use(paths.styles / f'{style}.mplstyle')

    # Data
    mwm_rgb = pd.read_csv(paths.data / 'MWM' / 'sample.csv')
    mwm_rgb = good_ages(mwm_rgb).copy()
    mwm_rgb = apply_alpha_cut(mwm_rgb)
    local_sample = mwm_rgb[
        (mwm_rgb['Rg'] >= RLIM[0]) &
        (mwm_rgb['Rg'] < RLIM[1]) &
        (mwm_rgb['z_max'] >= ZLIM[0]) &
        (mwm_rgb['z_max'] < ZLIM[1])
    ]
    # Restrict age trends to low-alpha stars only
    local_low_alpha = local_sample[local_sample['low_alpha']]
    local_high_alpha = local_sample[~local_sample['low_alpha']] # include border stars
    # Randomly sample fraction of stars to plot (still fit to full sample)
    plot_sample = sample_rows(
        local_sample, int(SAMPLE_FRACTION * local_sample.shape[0])
    )
    # plot_low_alpha = plot_sample[plot_sample['low_alpha']]
    # plot_high_alpha = plot_sample[~plot_sample['low_alpha']] # include border stars

    # Metallicity bins
    met_bins = np.arange(-0.5, 0.51, 0.2)
    age_arr = np.arange(0, 12.1, 0.1)

    nrows = len(met_bins)-1
    fig, axs = plt.subplots(
        nrows, 1,
        figsize=(ONE_COLUMN_WIDTH, 0.6*nrows*ONE_COLUMN_WIDTH),
        sharex=True, sharey=True, gridspec_kw={'hspace': 0.},
    )
    cmap = plt.get_cmap('viridis')
    norm = BoundaryNorm(met_bins, cmap.N, extend='both')
    xlim = (0, 11)
    ylim = (-0.7, 0.9)

    fits = []
    for i, ax in enumerate(axs):
        # Underlying scatter plot of all low-alpha stars
        pcm = axs[i].hexbin(
            local_sample['age'], local_sample['ce_mg_corr'],
            C=np.ones(local_sample.shape[0]),
            reduce_C_function=np.sum,
            gridsize=(30, 12),
            cmap='binary',
            linewidths=0.2,
            mincnt=1,
            extent=[xlim[0], xlim[1], ylim[0], ylim[1]]
        )
        # ax.scatter(
        #     local_low_alpha['age'], local_low_alpha['ce_mg_corr'],
        #     c='gray', s=1, marker='o', rasterized=True, edgecolor='none'
        # )
        # Plot high-alpha stars for reference (not fit)
        # ax.scatter(
        #     local_high_alpha['age'], local_high_alpha['ce_mg_corr'], 
        #     edgecolors='gray', 
        #     s=1, marker='o', rasterized=True, facecolors='w', linewidths=0.3
        # )
        # Bin by metallicity and fit linear trend to stars within good age range
        met_lim = (np.round(met_bins[-(i+2)], 2), np.round(met_bins[-(i+1)], 2))
        met_center = np.mean(met_lim) # mean metallicity of bin
        color = cmap(norm(met_center))
        # Scatter plot of stars in metallicity range
        subset_low_alpha = local_sample[
            (local_sample[MET_COL] >= met_lim[0]) &
            (local_sample[MET_COL] < met_lim[1]) &
            (local_sample['low_alpha'])
        ]
        ax.scatter(
            subset_low_alpha['age'], subset_low_alpha['ce_mg_corr'],
            color=color, s=3, marker='o', rasterized=True, edgecolor='none'
        )
        # Plot high-alpha stars for reference (not fit)
        subset_high_alpha = local_sample[
            (local_sample[MET_COL] >= met_lim[0]) &
            (local_sample[MET_COL] < met_lim[1]) &
            (~local_sample['low_alpha'])
        ]
        ax.scatter(
            subset_high_alpha['age'], subset_high_alpha['ce_mg_corr'], 
            edgecolors=color, 
            s=3, marker='o', rasterized=True, facecolors='w', linewidths=0.5
        )
        # Casali et al. (2025) relation for comparison
        ax.plot(age_arr, casali_relation(age_arr, met_center), 'w-', lw=2)
        ax.plot(
            age_arr, casali_relation(age_arr, met_center), 'k:', 
            label='Casali et al. (2025)'
        )
        # Fit linear age trend
        subset_fit = local_low_alpha[
            (local_low_alpha[MET_COL] >= met_lim[0]) & 
            (local_low_alpha[MET_COL] < met_lim[1]) &
            (local_low_alpha['age'] >= AGE_FIT_RANGE[0]) &
            (local_low_alpha['age'] < AGE_FIT_RANGE[1])
        ]
        regress = stats.linregress(subset_fit['age'], subset_fit['ce_mg_corr'])
        fits.append(regress)
        # Plot linear regression
        yfit = age_arr * regress.slope + regress.intercept
        # White outline for plot legibility
        ax.plot(age_arr, yfit, linestyle='-', linewidth=2, color='w')
        ax.plot( # extends beyond fit region
            age_arr[age_arr < AGE_FIT_RANGE[0]], 
            yfit[age_arr < AGE_FIT_RANGE[0]], 
            linestyle='--', 
            color=color
        )
        ax.plot( # segment within fit region
            age_arr[(AGE_FIT_RANGE[0] <= age_arr) & (age_arr < AGE_FIT_RANGE[1])], 
            yfit[(AGE_FIT_RANGE[0] <= age_arr) & (age_arr < AGE_FIT_RANGE[1])], 
            linestyle='-', 
            color=color
        )
        ax.plot( # extends beyond fit region
            age_arr[age_arr >= AGE_FIT_RANGE[1]], 
            yfit[age_arr >= AGE_FIT_RANGE[1]], 
            linestyle='--', 
            color=color
        )
        ax.set_title(
            r'$%s\leq$%s$<%s$' % (met_lim[0], MET_LABEL, met_lim[1]),
            y=0.95, pad=0, va='top',
            bbox={
                'facecolor': 'w', 
                'edgecolor': 'none', 
                'alpha': 1.,
                'pad': 0.2,
                'boxstyle': 'round'
            }
        )

    axs[0].set_xlim(xlim)
    axs[0].set_ylim(ylim)
    axs[0].xaxis.set_major_locator(MultipleLocator(5))
    axs[0].xaxis.set_minor_locator(MultipleLocator(1))
    axs[0].yaxis.set_major_locator(MultipleLocator(0.5))
    axs[0].yaxis.set_minor_locator(MultipleLocator(0.1))
    axs[-1].set_xlabel('Age [Gyr]')
    for ax in axs:
        ax.set_ylabel(r'[Ce/Mg]$_{\rm corr}$')

    plt.savefig(paths.figures / 'metallicity_fits')


def casali_relation(age, feh):
    """
    Global fit to the age-[Ce/Mg] relation from Casali et al. (2025).
    """
    return -0.032 * age + 0.194 * feh + 0.092


if __name__ == '__main__':
    main()
