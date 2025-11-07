"""
Plot [Ce/Mg] evolution predicted by one-zone models with delayed Ce enrichment.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import vice

from agb_enrichment_models import normalize, expfall
from utils import alpha_cut, amplified_agb, latex_float
from colormaps import paultol
import paths
import _globals

SFH_TIMESCALE = 15
AGB_STUDY = 'cristallo11'
AGB_YIELD_SCALE = 1
CCSN_CE_YIELD = 0
DELAYED_CE_YIELD = 1e-8


def main(style='paper'):
    plt.style.use(paths.styles / f'{style}.mplstyle')
    plt.rcParams['axes.prop_cycle'] = plt.cycler('color', paultol.bright.colors)
    
    # Select Solar neighborhood & Solar metallicity stars only
    mwm_rgb = pd.read_csv(paths.data / 'MWM' / 'MWM_RGB.csv')
    local_sample = mwm_rgb[
        (mwm_rgb['Rg'] >= 7.5) &
        (mwm_rgb['Rg'] < 8.5) &
        (mwm_rgb['z_max'] < 0.5) &
        (mwm_rgb['mg_h'] >= -0.1) &
        (mwm_rgb['mg_h'] < 0.1)
    ].copy()

    local_high_alpha = local_sample[
        local_sample['mg_fe'] >= alpha_cut(local_sample['fe_h'])
    ]
    local_low_alpha = local_sample[
        local_sample['mg_fe'] < alpha_cut(local_sample['fe_h'])
    ]

    figwidth = _globals.ONE_COLUMN_WIDTH
    fig, axs = plt.subplots(
        2, figsize=(figwidth, 1.33 * figwidth), 
        sharex=True, sharey=True,
        gridspec_kw={'hspace': 0.}
    )
    fig.subplots_adjust(right=0.67)

    # Plot MWM data
    datacolor = '0.3'
    scatter_kwargs = dict(
        marker='o',
        color=datacolor,
        s=1,
        linewidth=0.2,
        # rasterized=True
    )
    for ax in axs:
        ax.scatter(
            local_low_alpha['age'], local_low_alpha['ce_mg'],
            **scatter_kwargs
        )
        ax.scatter(
            local_high_alpha['age'], local_high_alpha['ce_mg'],
            facecolors='w', **scatter_kwargs
        )
        # median errors
        age_err_low = np.median(local_sample['age'] - local_sample['e_n_age'])
        age_err_high = np.median(local_sample['e_p_age'] - local_sample['age'])
        med_abund_err = local_sample['e_ce_mg'].median()
        ax.errorbar(
            10, 0.8, 
            xerr=[[age_err_low], [age_err_high]], 
            yerr=med_abund_err, 
            c=datacolor, capsize=0,
        )

    # Plot onezone models
    vice.yields.ccsne.settings['mg'] = 0.0019
    vice.yields.ccsne.settings['fe'] = 0.0012
    vice.yields.ccsne.settings['ce'] = CCSN_CE_YIELD
    vice.yields.sneia.settings['mg'] = 0.
    vice.yields.sneia.settings['fe'] = 0.0024
    vice.yields.sneia.settings['ce'] = 0
    vice.yields.agb.settings['ce'] = amplified_agb(
        'ce', study=AGB_STUDY, prefactor=AGB_YIELD_SCALE
    )
    output_dir = paths.data / 'onezone'
    output_dir.mkdir(parents=True, exist_ok=True)
    dt = 0.01
    simtime = np.arange(0, 12+dt, dt)
    model_kwargs = dict(
        func=normalize(expfall),
        mode='sfr',
        elements=('fe', 'mg', 'ce'),
        IMF='kroupa',
        eta=2.5,
        RIa='exp',
        delay=0.01,
        tau_star=2,
        dt=dt,
    )

    # Different delayed enrichment scales
    delayed_ce_yields = [2e-8, 1e-8, 3e-9, 0]
    colors = [paultol.bright.colors[c] for c in [1, 0, 2, 3]]
    for i, yld in enumerate(delayed_ce_yields):
        vice.yields.sneia.settings['ce'] = yld
        name = f'3p-delayed-ce{int(yld*1e9)}'
        sz = vice.singlezone(
            name=str(output_dir / name),
            tau_ia=1,
            **model_kwargs
        )
        sz.run(simtime, overwrite=True)
        hist = vice.history(str(paths.data/output_dir/name))
        axs[0].plot(hist['lookback'], hist['[ce/mg]'], color='w', linewidth=2)
        axs[0].plot(hist['lookback'], hist['[ce/mg]'], linestyle='-', 
                    color=colors[i], label=latex_float(yld)
        )
    axs[0].legend(
        title=r'$y_{\rm Ce}^{\rm NSM}$', 
        loc='upper left', 
        bbox_to_anchor=(1, 1)
    )

    # Different delayed enrichment timescales
    tau_ce_list = [1, 2, 5, 10]
    vice.yields.sneia.settings['ce'] = DELAYED_CE_YIELD
    for i, tau_ce in enumerate(tau_ce_list):
        name = f'3p-delayed-tau{tau_ce}'
        sz = vice.singlezone(
            name=str(output_dir / name),
            tau_ia=tau_ce,
            **model_kwargs
        )
        sz.run(simtime, overwrite=True)
        hist = vice.history(str(paths.data/output_dir/name))
        axs[1].plot(hist['lookback'], hist['[ce/mg]'], color='w', linewidth=2)
        axs[1].plot(hist['lookback'], hist['[ce/mg]'], linestyle='-', 
                    label='%s Gyr' % tau_ce
        )
    axs[1].legend(
        title=r'$\tau_{\rm NSM}$', 
        loc='upper left', 
        bbox_to_anchor=(1, 1)
    )

    axs[0].set_xlim((0, 12))
    axs[0].set_ylim((-1, 1))

    for ax in axs:
        ax.set_ylabel('[Ce/Mg]')
    axs[-1].set_xlabel('Age [Gyr]')

    fig.savefig(paths.figures / 'delayed_enrichment_models')


if __name__ == '__main__':
    main()
