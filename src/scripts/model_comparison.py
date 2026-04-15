"""
Plot [Ce/Mg] evolution predicted by various GCE models.
"""
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import vice

from multizone._globals import ZONE_WIDTH, END_TIME
from plotting import ONE_COLUMN_WIDTH, colored_text_legend
from colormaps import paultol
import paths

RADIUS = 8 # kpc, zone to plot gas evolution
SOLAR_CE_S_FRAC = 0.77 # Solar s-process fraction (Arlandini et al. 1999)
SOLAR_AGE = 4.6 # Gyr


def main(style='paper'):
    plt.style.use(paths.styles / f'{style}.mplstyle')
    plt.rcParams['axes.prop_cycle'] = plt.cycler('color', paultol.vibrant.colors)
    
    # Select Solar neighborhood & Solar metallicity stars only
    mwm_rgb = pd.read_csv(paths.data / 'MWM' / 'sample.csv')
    mwm_rgb = mwm_rgb[mwm_rgb['good_age']].copy()
    local_sample = mwm_rgb[
        (mwm_rgb['Rg'] >= 7) &
        (mwm_rgb['Rg'] < 9) &
        (mwm_rgb['z_max'] < 0.5) &
        (mwm_rgb['mg_h'] >= -0.1) &
        (mwm_rgb['mg_h'] < 0.1)
    ].copy()
    # Divide high and low alpha
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
    # for ax, col in zip(axs, ['ce_h_corr', 'ce_mg_corr']):
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
        3, -0.5, 
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
    output_names = [
        'yZ1-c11-fiducial', 
        'yZ1-c11-onlyagb',
        # 'yZ2-cristallo11',
        # 'yZ2-cristallo11-x2',
        # 'yZ2-cristallo11-x2-mscale'
        'yZ1-c11-lowsfe', 
        'yZ1-c11-lateburst',
        'yZ1-c11-Zscale',
        'yZ1-c11-mscale', 

        # 'rdelay-plaw',
        # 'rdelay-exp5',
        # 'rdelayx2-exp5'
        # 'karakas16'
    ]
    labels = [
        'Fiducial',
        r'No $r$-process',
        # r'$y/Z_\odot=2$',
        # 'Double yields',
        # r'Double yields ($M_{\rm AGB} \times0.5$)',
        r'${\rm SFE}\times0.5$',
        'Starburst',
        r'$Z_{\rm AGB} \times2$',
        r'$M_{\rm AGB} \times0.5$',

        # 'Delayed plaw',
        # 'Delayed 5 Gyr exp',
        # 'Double r-proc.'
        # 'Karakas16'
    ]
    colors = [paultol.vibrant.colors[i] for i in [0, 4, 2, 1, 3, 5]]
    zone = int(RADIUS / ZONE_WIDTH)
    for i, output_name in enumerate(output_names):
        zone_path = str(
            paths.multizone / output_name / 'diskmodel.vice' / ('zone%d' % zone)
        )
        hist = vice.history(zone_path)
        # for ax, col in zip(axs, ['[ce/h]', '[ce/mg]']):
        ax.plot(hist['lookback'], hist['[ce/mg]'], 'w-', lw=2)
        ax.plot(
            hist['lookback'], hist['[ce/mg]'], 
            color=colors[i], ls='-', label=labels[i]
        )
    # ax.legend()
    colored_text_legend(ax)
    
    ax.set_xlabel('Age [Gyr]')
    ax.set_ylabel('[Ce/Mg]', labelpad=-2)
    ax.set_xlim((0, END_TIME))
    ax.set_ylim((-0.7, 0.9))
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.set_title(r'$y/Z_\odot=1$, C11+C15')

    fig.savefig(paths.figures / 'model_comparison')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compare ISM evolution of multi-zone models.'
    )
    parser.add_argument('--style',
        choices=('paper', 'poster'),
        default='paper',
        help='Plot style to use (default: paper).'
    )
    args = parser.parse_args()
    main(**vars(args))
