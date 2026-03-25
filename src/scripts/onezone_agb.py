"""
Plot [Ce/Mg] evolution predicted by one-zone GCE models with modifications
to the AGB yield prescriptions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import vice

from multizone.src.yields.utils import adjusted_agb
from onezone_sfh import run_singlezone, expfall
from plotting import ONE_COLUMN_WIDTH
from colormaps import paultol
import paths

# CCSN and SN Ia yields
from multizone.src.yields import W24

SFH_TIMESCALE = 15
AGB_STUDY = 'cristallo11'
END_TIME = 13.2 # Gyr
SOLAR_CE_S_FRAC = 0.77 # Solar s-process fraction (Arlandini et al. 1999)
SOLAR_AGE = 4.6 # Gyr
ETA_SUN = 0.4 # default mass-loading factor at Solar radius


def main(style='paper'):
    plt.style.use(paths.styles / f'{style}.mplstyle')
    plt.rcParams['axes.prop_cycle'] = plt.cycler('color', paultol.bright.colors)
    
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

    local_high_alpha = local_sample[local_sample['high_alpha']]
    local_low_alpha = local_sample[local_sample['low_alpha']]

    # Median errors
    age_err_low = np.median(local_sample['age'] - local_sample['e_n_age'])
    age_err_high = np.median(local_sample['e_p_age'] - local_sample['age'])
    med_abund_err = local_sample['e_ce_mg'].median()

    figwidth = ONE_COLUMN_WIDTH
    fig, axs = plt.subplots(
        3, figsize=(figwidth, 1.67 * figwidth), 
        sharex=True, sharey=True,
        gridspec_kw={'hspace': 0.}
    )
    fig.subplots_adjust(right=0.62)
    legend_kwargs = dict(bbox_to_anchor=(1, 1), loc='upper left')

    # Plot MWM data
    datacolor = '0.3'
    scatter_kwargs = dict(
        marker='o',
        color=datacolor,
        s=1,
        linewidth=0.2,
        rasterized=True
    )
    for ax in axs:
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
            10, 0.8, 
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
    axs[0].plot(SOLAR_AGE, np.log10(SOLAR_CE_S_FRAC), 'wo', zorder=9)
    axs[0].text(
        SOLAR_AGE, np.log10(SOLAR_CE_S_FRAC), r'$\otimes$',
        va='center', ha='center', zorder=10, weight='bold', usetex=True
    )

    # Plot onezone models
    vice.yields.sneia.settings['ce'] = 0
    output_dir = paths.data / 'onezone' / 'agb'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate prompt Ce enrichment (assigned to CCSNe for convenience)
    ccsn_ce_yield = vice.yields.ccsne.settings['mg'] * (
        (1 - SOLAR_CE_S_FRAC) * vice.solar_z['ce'] / vice.solar_z['mg']
    )

    # Different AGB yield scales
    yield_scales = [3, 2, 1]
    colors = [paultol.bright.colors[c] for c in [1, 2, 0]]
    vice.yields.ccsne.settings['ce'] = ccsn_ce_yield
    for i, scale in enumerate(yield_scales):
        vice.yields.agb.settings['ce'] = adjusted_agb(
            'ce', study=AGB_STUDY, amp=scale
        )
        name = f'amp{scale}'
        run_singlezone(name, expfall, output_dir=output_dir)
        hist = vice.history(str(output_dir/name))
        axs[0].plot(hist['lookback'], hist['[ce/mg]'], color='w', linewidth=2)
        axs[0].plot(hist['lookback'], hist['[ce/mg]'], linestyle='-', 
                    color=colors[i], 
                    label=r'$\times%s$' % scale
        )
    # no prompt r-process
    vice.yields.agb.settings['ce'] = adjusted_agb(
        'ce', study=AGB_STUDY, amp=1
    )
    vice.yields.ccsne.settings['ce'] = 0
    name = f'amp1-norproc'
    run_singlezone(name, expfall, output_dir=output_dir)
    hist = vice.history(str(output_dir/name))
    axs[0].plot(hist['lookback'], hist['[ce/mg]'], color='w', linewidth=2)
    axs[0].plot(hist['lookback'], hist['[ce/mg]'], linestyle='--', 
                color=paultol.bright.colors[0], label=r'No $r$-proc.'
    )
    axs[0].legend(title=r'\textbf{Enhancement}', **legend_kwargs)

    # Mass-shifted AGB enrichment
    # mass_shifts = [0, -0.5, -1, -1.5, -2]
    # colors = [paultol.bright.colors[c] for c in [0, 4, 2, 3, 1]]
    # vice.yields.ccsne.settings['ce'] = ccsn_ce_yield
    # for i, dm in enumerate(mass_shifts):
    #     vice.yields.agb.settings['ce'] = adjusted_agb(
    #         'ce', study=AGB_STUDY, dm=dm, amp=1
    #     )
    #     name = f'mshift{dm}'
    #     run_singlezone(name, expfall)
    #     hist = vice.history(str(paths.data/output_dir/name))
    #     axs[1].plot(hist['lookback'], hist['[ce/mg]'], color='w', linewidth=2)
    #     axs[1].plot(hist['lookback'], hist['[ce/mg]'], linestyle='-', 
    #                 color=colors[i], 
    #                 label=r'$%s$ M$_\odot$' % dm
    #     )
    # axs[1].legend(title=r'\textbf{Mass shift}', **legend_kwargs)

    # Mass-shifted AGB enrichment
    mass_scales = [2, 1, 0.7, 0.5, 0.3]
    colors = [paultol.bright.colors[c] for c in [4, 0, 2, 3, 1]]
    vice.yields.ccsne.settings['ce'] = ccsn_ce_yield
    for i, mscale in enumerate(mass_scales):
        vice.yields.agb.settings['ce'] = adjusted_agb(
            'ce', study=AGB_STUDY, dm=0, amp=1, mscale=mscale,
        )
        name = f'mscale{mscale}'
        run_singlezone(name, expfall, output_dir=output_dir)
        hist = vice.history(str(output_dir/name))
        axs[1].plot(hist['lookback'], hist['[ce/mg]'], color='w', linewidth=2)
        axs[1].plot(hist['lookback'], hist['[ce/mg]'], linestyle='-', 
                    color=colors[i], 
                    label=r'$\times%s$' % mscale
        )
    axs[1].legend(title=r'\textbf{Mass scale}', **legend_kwargs)

    # Metallicity-scaled AGB enrichment
    met_scales = [5, 2, 1, 0.5]
    colors = [paultol.bright.colors[c] for c in [1, 2, 0, 4]]
    vice.yields.ccsne.settings['ce'] = ccsn_ce_yield
    for i, Zscale in enumerate(met_scales):
        vice.yields.agb.settings['ce'] = adjusted_agb(
            'ce', study=AGB_STUDY, dm=0, amp=1, Zscale=Zscale
        )
        name = f'Zscale{Zscale}'
        run_singlezone(name, expfall, output_dir=output_dir)
        hist = vice.history(str(output_dir/name))
        axs[2].plot(hist['lookback'], hist['[ce/mg]'], color='w', linewidth=2)
        axs[2].plot(hist['lookback'], hist['[ce/mg]'], linestyle='-', 
                    color=colors[i], 
                    label=r'$\times%s$' % Zscale
        )
    axs[2].legend(title=r'\textbf{$Z$ scale}', **legend_kwargs)

    axs[0].set_xlim((0, END_TIME))
    axs[0].set_ylim((-0.8, 1))
    axs[0].xaxis.set_major_locator(MultipleLocator(5))
    axs[0].xaxis.set_minor_locator(MultipleLocator(1))
    axs[0].yaxis.set_major_locator(MultipleLocator(0.5))
    axs[0].yaxis.set_minor_locator(MultipleLocator(0.1))

    titles = ['(a)', '(b)', '(c)', '(d)']
    for i, ax in enumerate(axs):
        ax.set_ylabel('[Ce/Mg]')
        ax.set_title(titles[i], loc='left', x=0.05, y=0.9, va='top')
    axs[-1].set_xlabel('Age [Gyr]')

    fig.savefig(paths.figures / 'onezone_agb')
    plt.close()

    # Sanity plots
    hist = vice.history(str(output_dir / 'amp1'))
    plt.plot(hist['lookback'], hist['[mg/h]'], 'k-')
    plt.xlabel('Lookback [Gyr]')
    plt.ylabel('[Mg/H]')
    plt.ylim((-1.5, 0.5))
    plt.savefig(paths.extra / 'agb_mgh.png')
    plt.close()


if __name__ == '__main__':
    main()
