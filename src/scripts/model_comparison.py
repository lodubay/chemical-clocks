"""
Plot [Ce/Mg] evolution predicted by various GCE models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import vice

from multizone._globals import ZONE_WIDTH, END_TIME
from utils import apply_alpha_cut, good_ages, plot_gas_abundance
from plotting import ONE_COLUMN_WIDTH
from colormaps import paultol
import paths

RADIUS = 8 # kpc, zone to plot gas evolution
SOLAR_CE_S_FRAC = 0.77 # Solar s-process fraction (Arlandini et al. 1999)
SOLAR_AGE = 4.6 # Gyr
OUTPUT_NAMES = [
    'fiducial', 
    'low-sfe', 
    'insideout-mscale', 
    'Zscale',
    'no-rproc'
]


def main(style='paper'):
    plt.style.use(paths.styles / f'{style}.mplstyle')
    plt.rcParams['axes.prop_cycle'] = plt.cycler('color', paultol.bright.colors)
    
    # Select Solar neighborhood & Solar metallicity stars only
    mwm_rgb = pd.read_csv(paths.data / 'MWM' / 'sample.csv')
    mwm_rgb = good_ages(mwm_rgb).copy()
    local_sample = mwm_rgb[
        (mwm_rgb['Rg'] >= 7) &
        (mwm_rgb['Rg'] < 9) &
        (mwm_rgb['z_max'] < 0.5) &
        (mwm_rgb['mg_h'] >= -0.1) &
        (mwm_rgb['mg_h'] < 0.1)
    ].copy()
    # Divide high and low alpha
    local_sample = apply_alpha_cut(local_sample, buffer=0.02)
    local_high_alpha = local_sample[local_sample['high_alpha']]
    local_low_alpha = local_sample[local_sample['low_alpha']]

    # Median errors
    age_err_low = np.median(local_sample['age'] - local_sample['e_n_age'])
    age_err_high = np.median(local_sample['e_p_age'] - local_sample['age'])
    med_abund_err = local_sample['e_ce_mg'].median()

    figwidth = ONE_COLUMN_WIDTH
    fig, ax = plt.subplots(figsize=(figwidth, 0.8 * figwidth))
    # fig.subplots_adjust(right=0.62)
    # legend_kwargs = dict(bbox_to_anchor=(1, 1), loc='upper left')

    # Plot MWM data
    datacolor = '0.3'
    scatter_kwargs = dict(
        marker='o',
        color=datacolor,
        s=1,
        linewidth=0.2,
        rasterized=True
    )
    # for ax in axs:
    ax.scatter(
        local_low_alpha['age'], local_low_alpha['ce_mg_corr'],
        **scatter_kwargs
    )
    ax.scatter(
        local_high_alpha['age'], local_high_alpha['ce_mg_corr'],
        facecolors='w', **scatter_kwargs
    )
    # median errors
    ax.errorbar(
        3, -0.6, 
        xerr=[[age_err_low], [age_err_high]], 
        yerr=med_abund_err, 
        c=datacolor, capsize=0,
    )
    # indicate Solar value
    ax.plot(SOLAR_AGE, 0, 'wo', zorder=9)
    ax.text(
        SOLAR_AGE, 0, r'$\odot$',
        va='center', ha='center', zorder=10, weight='bold', usetex=True
    )
    # indicate Solar s-process fraction
    ax.plot(SOLAR_AGE, np.log10(SOLAR_CE_S_FRAC), 'wo', zorder=9)
    ax.text(
        SOLAR_AGE, np.log10(SOLAR_CE_S_FRAC), r'$\otimes$',
        va='center', ha='center', zorder=10, weight='bold', usetex=True
    )

    # Plot multizone model abundance evolution
    zone = int(RADIUS / ZONE_WIDTH)
    for i, output_name in enumerate(OUTPUT_NAMES):
        zone_path = str(
            paths.multizone / output_name / 'diskmodel.vice' / ('zone%d' % zone)
        )
        hist = vice.history(zone_path)
        ax.plot(hist['lookback'], hist['[ce/mg]'], 'w-', lw=2)
        ax.plot(hist['lookback'], hist['[ce/mg]'], label=output_name)
    ax.legend()
    
    ax.set_xlabel('Age [Gyr]')
    ax.set_ylabel('[Ce/Mg]')
    ax.set_xlim((0, END_TIME))
    ax.set_ylim((-0.8, 1))
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))

    fig.savefig(paths.figures / 'model_comparison')
    plt.close()


if __name__ == '__main__':
    main()
