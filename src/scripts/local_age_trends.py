"""
This script fits a linear trend to the age-[Ce/Mg] relation in metallicity bins
for stars in the Solar neighborhood.
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

MET_COL = 'm_h_atm' # Column with metallicity values
MET_LABEL = r'[M/H]$_{\rm atm}$'
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
    plot_low_alpha = plot_sample[plot_sample['low_alpha']]
    plot_high_alpha = plot_sample[~plot_sample['low_alpha']] # include border stars

    # Metallicity bins
    met_bins = np.arange(-0.55, 0.46, 0.1)

    fig, ax = plt.subplots(
        figsize=(ONE_COLUMN_WIDTH, 0.75*ONE_COLUMN_WIDTH),
        constrained_layout=True
    )
    cmap = plt.get_cmap('viridis')
    norm = BoundaryNorm(met_bins, cmap.N, extend='both')

    pc = ax.scatter(
        plot_low_alpha['age'], plot_low_alpha['ce_mg_corr'], 
        c=plot_low_alpha[MET_COL], 
        cmap=cmap, norm=norm,
        s=1, marker='o', rasterized=True, edgecolor='none'
    )
    # Plot high-alpha stars for reference (not fit)
    c = cmap(norm(plot_high_alpha[MET_COL]))
    ax.scatter(
        plot_high_alpha['age'], plot_high_alpha['ce_mg_corr'], 
        edgecolors=c, 
        s=1, marker='o', rasterized=True, facecolors='w', linewidths=0.3
    )
    cbar = fig.colorbar(pc, ax=ax, label=MET_LABEL)

    # Bin by metallicity and fit linear trend to stars within good age range
    fits = []
    age_arr = np.arange(0, 12.1, 0.1)
    for i in range(len(met_bins)-1):
        met_lim = met_bins[i:i+2]
        met_center = np.mean(met_lim) # mean metallicity of bin
        subset = local_low_alpha[
            (local_low_alpha[MET_COL] >= met_lim[0]) & 
            (local_low_alpha[MET_COL] < met_lim[1]) &
            (local_low_alpha['age'] >= AGE_FIT_RANGE[0]) &
            (local_low_alpha['age'] < AGE_FIT_RANGE[1])
        ]
        # Fit linear age trend
        regress = stats.linregress(subset['age'], subset['ce_mg_corr'])
        fits.append(regress)
        # Plot linear regression
        yfit = age_arr * regress.slope + regress.intercept
        # White outline for plot legibility
        ax.plot(age_arr, yfit, linestyle='-', linewidth=2, color='w')
        ax.plot( # extends beyond fit region
            age_arr[age_arr < AGE_FIT_RANGE[0]], 
            yfit[age_arr < AGE_FIT_RANGE[0]], 
            linestyle='--', 
            color=cmap(norm(met_center))
        )
        ax.plot( # segment within fit region
            age_arr[(AGE_FIT_RANGE[0] <= age_arr) & (age_arr < AGE_FIT_RANGE[1])], 
            yfit[(AGE_FIT_RANGE[0] <= age_arr) & (age_arr < AGE_FIT_RANGE[1])], 
            linestyle='-', 
            color=cmap(norm(met_center))
        )
        ax.plot( # extends beyond fit region
            age_arr[age_arr >= AGE_FIT_RANGE[1]], 
            yfit[age_arr >= AGE_FIT_RANGE[1]], 
            linestyle='--', 
            color=cmap(norm(met_center))
        )

    ax.set_xlim((0, 12))
    ax.set_ylim((-0.8, 0.8))
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    cbar.ax.yaxis.set_major_locator(MultipleLocator(0.1))
    cbar.ax.tick_params(axis='y', which='major', right=False)
    ax.set_xlabel('Age [Gyr]')
    ax.set_ylabel(r'[Ce/Mg]$_{\rm corr}$')

    plt.savefig(paths.figures / 'local_age_trends')


if __name__ == '__main__':
    main()
