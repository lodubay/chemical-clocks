"""
This script fits a linear trend to the age-[Ce/Mg] relation in metallicity bins
for stars in the Solar neighborhood.
"""
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MultipleLocator
from scipy import stats

from plotting import ONE_COLUMN_WIDTH, insert_colorbar_axes
from colormaps import paultol
from utils import sample_rows, get_bin_centers
import paths

MET_COL = 'fe_h' # Column with metallicity values
# MET_LABEL = r'[M/H]$_{\rm atm}$'
MET_LABEL = '[Fe/H]'
AGE_FIT_RANGE = (1, 8) # Range of ages to fit linear trend
SAMPLE_FRACTION = 1 # fraction of stars to plot in each panel
RLIM = (7, 9)
ZLIM = (0, 0.5)
AGE_DELTA = 5 # Gyr, linear age shift for regression


def main(style='paper', cmap='viridis'):
    plt.style.use(paths.styles / f'{style}.mplstyle')

    # Data
    mwm_rgb = pd.read_csv(paths.data / 'MWM' / 'sample.csv')
    mwm_rgb = mwm_rgb[mwm_rgb['good_age']].copy()
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
    plot_low_alpha = plot_sample[plot_sample['low_alpha']]
    plot_high_alpha = plot_sample[~plot_sample['low_alpha']] # include border stars

    # Metallicity bins
    met_bins = np.arange(-0.55, 0.46, 0.1)

    fig, axs = plt.subplots(2, 1,
        figsize=(ONE_COLUMN_WIDTH, 1.5*ONE_COLUMN_WIDTH),
        gridspec_kw={'hspace': 0.25}
    )
    plt.subplots_adjust(right=0.75)
    # cax = insert_colorbar_axes(fig)
    cmap = plt.get_cmap(cmap)
    norm = BoundaryNorm(met_bins, cmap.N, extend='both')

    # Top panel: fit age trends binned by metallicity
    pc = axs[0].scatter(
        plot_low_alpha['age'], plot_low_alpha['ce_mg_corr'], 
        c=plot_low_alpha[MET_COL], 
        cmap=cmap, norm=norm,
        s=1, marker='o', rasterized=True, edgecolor='none'
    )
    # Plot high-alpha stars for reference (not fit)
    c = cmap(norm(plot_high_alpha[MET_COL]))
    axs[0].scatter(
        plot_high_alpha['age'], plot_high_alpha['ce_mg_corr'], 
        edgecolors=c, 
        s=1, marker='o', rasterized=True, facecolors='w', linewidths=0.3
    )
    cbar = fig.colorbar(pc, ax=axs[0], fraction=0.05, label=MET_LABEL)

    # Indicate median abundance errors
    age_err_low = np.median(mwm_rgb['age'] - mwm_rgb['e_n_age'])
    age_err_high = np.median(mwm_rgb['e_p_age'] - mwm_rgb['age'])
    med_abund_err = mwm_rgb['e_ce_h'].median()
    axs[0].errorbar(
        8, 0.6, 
        xerr=[[age_err_low], [age_err_high]], 
        yerr=med_abund_err, 
        c='gray', capsize=0, #elinewidth=0.5,
    )

    # Bin by metallicity and fit linear trend to stars within good age range
    fits = []
    age_arr = np.arange(0, 12.1, 0.1)
    for i in range(len(met_bins)-1):
        met_lim = met_bins[i:i+2]
        met_center = np.mean(met_lim) # mean metallicity of bin
        color = cmap(norm(met_center))
        subset = local_low_alpha[
            (local_low_alpha[MET_COL] >= met_lim[0]) & 
            (local_low_alpha[MET_COL] < met_lim[1]) &
            (local_low_alpha['age'] >= AGE_FIT_RANGE[0]) &
            (local_low_alpha['age'] < AGE_FIT_RANGE[1])
        ]
        # Fit linear age trend
        regress = stats.linregress(subset['age'] - AGE_DELTA, subset['ce_mg_corr'])
        fits.append(regress)
        # Plot linear regression
        yfit = (age_arr - AGE_DELTA) * regress.slope + regress.intercept
        # White outline for plot legibility
        axs[0].plot(age_arr, yfit, linestyle='-', linewidth=2, color='w')
        axs[0].plot( # extends beyond fit region
            age_arr[age_arr < AGE_FIT_RANGE[0]], 
            yfit[age_arr < AGE_FIT_RANGE[0]], 
            linestyle='--', 
            color=color
        )
        axs[0].plot( # segment within fit region
            age_arr[(AGE_FIT_RANGE[0] <= age_arr) & (age_arr < AGE_FIT_RANGE[1])], 
            yfit[(AGE_FIT_RANGE[0] <= age_arr) & (age_arr < AGE_FIT_RANGE[1])], 
            linestyle='-', 
            color=color
        )
        axs[0].plot( # extends beyond fit region
            age_arr[age_arr >= AGE_FIT_RANGE[1]], 
            yfit[age_arr >= AGE_FIT_RANGE[1]], 
            linestyle='--', 
            color=color
        )
    
    axs[0].set_xlim((0, 11))
    axs[0].set_ylim((-0.8, 0.8))
    axs[0].xaxis.set_major_locator(MultipleLocator(5))
    axs[0].xaxis.set_minor_locator(MultipleLocator(1))
    axs[0].yaxis.set_major_locator(MultipleLocator(0.5))
    axs[0].yaxis.set_minor_locator(MultipleLocator(0.1))
    cbar.ax.yaxis.set_major_locator(MultipleLocator(0.1))
    cbar.ax.tick_params(axis='y', which='major', right=False)
    axs[0].set_xlabel('Age [Gyr]')
    axs[0].set_ylabel('[Ce/Mg]')
    
    # Bottom panel: plot slopes, intercepts as a function of metallicity
    # Plot intercept on same axes
    intax = axs[1].twinx()
    intercepts = [f.intercept for f in fits]
    intercept_errs = [f.intercept_stderr for f in fits]
    intercept_color = paultol.vibrant.colors[2]
    met_bin_centers = get_bin_centers(met_bins)
    intax.errorbar(
        met_bin_centers, intercepts, yerr=intercept_errs, 
        marker='s', c=intercept_color, linestyle='-', capsize=0
    )
    # Plot slopes
    slopes = [f.slope for f in fits]
    slope_errs = [f.stderr for f in fits]
    slope_color = paultol.vibrant.colors[1]
    axs[1].errorbar(
        met_bin_centers, slopes, yerr=slope_errs, 
        marker='o', c=slope_color, capsize=0, zorder=5
    )
    # Compare against Casali slope and intercept
    intax.plot(
        met_bin_centers, [0.194 * m + 0.092 for m in met_bin_centers],
        color=intercept_color, linestyle='--', zorder=1
    )
    axs[1].plot(
        met_bin_centers, [-0.032 for m in met_bin_centers],
        color=slope_color, linestyle='--', zorder=1, 
        label='Casali et al. (2025)'
    )

    axs[1].set_xlabel(MET_LABEL)
    axs[1].set_ylabel(r'Slope [dex Gyr$^{-1}$]', color=slope_color)
    xlim = (met_bin_centers[0]-0.1, met_bin_centers[-1]+0.1)
    axs[1].set_xlim(xlim)
    axs[1].set_ylim((-0.07, 0.07))
    axs[1].tick_params(axis='y', labelcolor=slope_color)
    axs[1].xaxis.set_major_locator(MultipleLocator(0.5))
    axs[1].xaxis.set_minor_locator(MultipleLocator(0.1))
    axs[1].yaxis.set_major_locator(MultipleLocator(0.05))
    axs[1].yaxis.set_minor_locator(MultipleLocator(0.01))
    intax.set_ylabel('[Ce/Mg] at %s Gyr' % AGE_DELTA, color=intercept_color)
    intax.set_ylim((-0.25, 0.25))
    intax.tick_params(axis='y', labelcolor=intercept_color)
    intax.yaxis.set_major_locator(MultipleLocator(0.1))
    intax.yaxis.set_minor_locator(MultipleLocator(0.02))

    leg = axs[1].legend()
    leg.legend_handles[0].set_color('k')

    plt.savefig(paths.figures / 'local_metallicity_trends')


def casali_global_fit(age, feh):
    """
    Global fit to the age-[Ce/Mg] relation from Casali et al. (2025).
    """
    return -0.032 * age + 0.194 * feh + 0.092


def casali_local_fit(age, feh):
    """
    Fit to the age-[Ce/Mg] relation in the 7-8 kpc bin from Casali et al. (2025).
    """
    return -0.036 * age + 0.080 * feh + 0.095


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Fit local [Ce/Mg]-age relation in metallicity bins.'
    )
    parser.add_argument('--style',
        choices=('paper', 'poster'),
        default='paper',
        help='Plot style to use (default: paper).'
    )
    parser.add_argument('--cmap',
        default='viridis',
        help='Colormap to use for metallicity dimension.'
    )
    args = parser.parse_args()
    main(**vars(args))
