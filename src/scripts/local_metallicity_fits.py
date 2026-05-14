"""
Plot the local [Ce/Mg]-age relation and fit a linear trend in metallicity bins.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MultipleLocator
from matplotlib.gridspec import GridSpec
from scipy import stats

from plotting import TWO_COLUMN_WIDTH
from utils import get_bin_centers
import paths

MET_COL = 'fe_h_corr' # Column with metallicity values
MET_LABEL = '[Fe/H]'
AGE_FIT_RANGE = (1, 8) # Range of ages to fit linear trend
RLIM = (7, 9)
ZLIM = (0, 0.5)
AGE_DELTA = 5 # Gyr, linear age shift for regression
SOLAR_AGE = 4.6 # Gyr


def main(style='paper'):
    # Data
    mwm_rgb = pd.read_csv(paths.data / 'MWM' / 'sample.csv')
    local_sample = mwm_rgb[
        (mwm_rgb['Rg'] >= RLIM[0]) &
        (mwm_rgb['Rg'] < RLIM[1]) &
        (mwm_rgb['z_max'] >= ZLIM[0]) &
        (mwm_rgb['z_max'] < ZLIM[1]) &
        (mwm_rgb['good_age'])
    ].copy()
    local_sample['e_age'] = 0.5 * (
        (local_sample['e_p_age'] - local_sample['age']) + 
        (local_sample['age'] - local_sample['e_n_age'])
    )

    # Set up figure
    plt.style.use(paths.styles / f'{style}.mplstyle')
    fig, axs = plt.subplots(
        3, 3, 
        figsize=(TWO_COLUMN_WIDTH, 0.5*TWO_COLUMN_WIDTH),
        sharex=True, sharey=True,
        gridspec_kw=dict(left=0.1, right=0.66, hspace=0, wspace=0)
    )

    # Metallicity bins
    met_bins = np.arange(-0.45, 0.46, 0.1)
    age_arr = np.arange(0, 12.1, 0.1)
    cmap = plt.get_cmap('viridis')
    norm = BoundaryNorm(met_bins, cmap.N, extend='both')

    # Left panels: individual fits in metallicity bins
    fits = []
    # gs0 = GridSpec(3, 3, figure=fig, right=0.7, hspace=0, wspace=0)
    for i, ax in enumerate(axs.flatten()):
        # Underlying scatter plot of all low-alpha stars
        ax.scatter(
            local_sample['age'], local_sample['ce_mg_corr'],
            color='gray', s=1, marker='o', rasterized=True, edgecolor='none'
        )
        # Bin by metallicity and fit linear trend to stars within good age range
        met_lim = tuple(np.round(met_bins[i:i+2], 2))
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
        # ax.plot(age_arr, casali_relation(age_arr, met_center), 'w-', lw=2)
        # ax.plot(
        #     age_arr, casali_relation(age_arr, met_center), 'k:', 
        #     label='Casali et al. (2025)'
        # )
        # Fit linear age trend
        subset_fit = local_sample[
            (local_sample[MET_COL] >= met_lim[0]) & 
            (local_sample[MET_COL] < met_lim[1]) &
            (local_sample['age'] >= AGE_FIT_RANGE[0]) &
            (local_sample['age'] < AGE_FIT_RANGE[1]) &
            (local_sample['low_alpha'])
        ]
        regress = stats.linregress(
            subset_fit['age'] - AGE_DELTA, subset_fit['ce_mg_corr']
        )
        fits.append(regress)
        # Plot linear regression
        yfit = (age_arr - AGE_DELTA) * regress.slope + regress.intercept
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
        # indicate Solar value
        if met_lim[0] <= 0 < met_lim[1]:
            ax.plot(SOLAR_AGE, 0, 'wo', zorder=9)
            ax.text(
                SOLAR_AGE, 0, r'$\odot$',
                va='center', ha='center', zorder=10, weight='bold', usetex=True
            )
        ax.text(
            0.5, 0.95,
            # r'$%s\leq{\rm%s}<%s$' % (met_lim[0], MET_LABEL, met_lim[1]),
            r'$[%s,%s)$' % met_lim,
            # y=0.95, pad=0, va='top',
            va='top', ha='center',
            transform=ax.transAxes,
            bbox={
                'facecolor': 'w', 
                'edgecolor': 'none', 
                'alpha': 0.8,
                'pad': 0.2,
                'boxstyle': 'round'
            }
        )
        # 

    # Indicate median abundance and age errors
    # age_err_low = np.median(local_sample['age'] - local_sample['e_n_age'])
    # age_err_high = np.median(local_sample['e_p_age'] - local_sample['age'])
    # med_abund_err = local_sample['e_ce_h'].median()
    # axs[0,0].errorbar(
    #     9, 0.5, 
    #     xerr=[[age_err_low], [age_err_high]], 
    #     yerr=med_abund_err, 
    #     c='gray', capsize=0, #elinewidth=0.5,
    # )

    axs[0,0].set_xlim((0, 11))
    axs[0,0].set_ylim((-0.7, 0.9))
    axs[0,0].xaxis.set_major_locator(MultipleLocator(5))
    axs[0,0].xaxis.set_minor_locator(MultipleLocator(1))
    axs[0,0].yaxis.set_major_locator(MultipleLocator(0.5))
    axs[0,0].yaxis.set_minor_locator(MultipleLocator(0.1))
    axs[-1,1].set_xlabel('Age [Gyr]')
    axs[1,0].set_ylabel('[Ce/Mg]')
    axs[0,1].set_title('Bins in [Fe/H]')
    # for ax in axs[-1]:
    #     ax.set_xlabel('Age [Gyr]')
    # for ax in axs[:,0]:
    #     ax.set_ylabel('[Ce/Mg]')

    # Right panels: plot slope and intercept vs metallicity
    gs = GridSpec(2, 1, figure=fig, left=0.75, right=0.98, hspace=0)
    ax0 = fig.add_subplot(gs[0])
    slopes = [f.slope for f in fits]
    slope_errs = [f.stderr for f in fits]
    met_bin_centers = get_bin_centers(met_bins)
    ax0.plot(met_bin_centers, slopes, 'k-', label='This work')
    for i, met_mean in enumerate(met_bin_centers):
        ax0.errorbar(
            met_mean, slopes[i], yerr=slope_errs[i],
            c=cmap(norm(met_mean)),
            marker='o', capsize=0
        )
    # Plot intercepts
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    intercepts = [f.intercept for f in fits]
    intercept_errs = [f.intercept_stderr for f in fits]
    ax1.plot(met_bin_centers, intercepts, 'k-')
    for i, met_mean in enumerate(met_bin_centers):
        ax1.errorbar(
            met_mean, intercepts[i], yerr=intercept_errs[i],
            c=cmap(norm(met_mean)),
            marker='s', capsize=0
        )
    # indicate Solar value
    ax1.plot(0, 0, 'wo', zorder=9)
    ax1.text(
        0, 0, r'$\odot$',
        va='center', ha='center', zorder=10, weight='bold', usetex=True
    )
    # Casali et al. (2025) comparison
    xarr = np.arange(-0.5, 0.51, 0.01)
    ax0.plot(
        xarr, casali_fit(1, xarr) - casali_fit(0, xarr), 
        'k--', zorder=1, label='Casali et al. (2025)'
    )
    ax1.plot(
        xarr, casali_fit(AGE_DELTA, xarr),
        'k--', zorder=1, label='Casali et al. (2025)'
    )
    ax0.legend()
    ax1.set_xlabel('[Fe/H]')
    ax0.set_ylabel(r'Slope [dex Gyr$^{-1}$]')
    ax1.set_ylabel(r'[Ce/Mg] at $\tau=%s$ Gyr' % AGE_DELTA)
    ax0.set_xlim((-0.5, 0.5))
    ax0.tick_params(axis='x', labelbottom=False)
    ax0.set_ylim((-0.07, 0.05))
    ax1.set_ylim((-0.18, 0.22))
    ax0.xaxis.set_major_locator(MultipleLocator(0.5))
    ax0.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax0.yaxis.set_major_locator(MultipleLocator(0.05))
    ax0.yaxis.set_minor_locator(MultipleLocator(0.01))
    ax1.yaxis.set_major_locator(MultipleLocator(0.1))
    ax1.yaxis.set_minor_locator(MultipleLocator(0.02))

    plt.savefig(paths.figures / 'local_metallicity_fits')


def casali_fit(age, feh):
    """
    Global fit to the age-[Ce/Mg] relation from Casali et al. (2025).
    """
    return -0.032 * age + 0.194 * feh + 0.092


if __name__ == '__main__':
    main()
