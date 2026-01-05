"""
Plot [Ce/Mg] evolution predicted by one-zone models with delayed Ce enrichment.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import vice

from agb_enrichment_models import normalize, expfall, exprise, constant, lateburst
from utils import alpha_cut, adjusted_agb, good_ages
from plotting import latex_float
from colormaps import paultol
import paths
import _globals

SFH_TIMESCALE = 15
AGB_STUDY = 'cristallo11'
AGB_YIELD_SCALE = 1
CCSN_CE_YIELD = 0
DELAYED_CE_YIELD = 1e-8
DELAYED_CE_TIMESCALE = 5


def main(style='paper'):
    plt.style.use(paths.styles / f'{style}.mplstyle')
    plt.rcParams['axes.prop_cycle'] = plt.cycler('color', paultol.bright.colors)
    
    # Select Solar neighborhood & Solar metallicity stars only
    mwm_rgb = pd.read_csv(paths.data / 'MWM' / 'MWM_RGB.csv')
    mwm_rgb = good_ages(mwm_rgb).copy()
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
        4, figsize=(figwidth, 2.67 * figwidth), 
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
            local_low_alpha['age'], local_low_alpha['ce_mg_corr'],
            **scatter_kwargs
        )
        ax.scatter(
            local_high_alpha['age'], local_high_alpha['ce_mg_corr'],
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
    vice.yields.agb.settings['ce'] = adjusted_agb(
        'ce', study=AGB_STUDY, amp=AGB_YIELD_SCALE
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
            tau_ia=DELAYED_CE_TIMESCALE,
            **model_kwargs
        )
        sz.run(simtime, overwrite=True)
        hist = vice.history(str(output_dir/name))
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
    colors = [paultol.bright.colors[c] for c in [3, 2, 0, 1]]
    vice.yields.sneia.settings['ce'] = DELAYED_CE_YIELD
    for i, tau_ce in enumerate(tau_ce_list):
        name = f'3p-delayed-tau{tau_ce}'
        sz = vice.singlezone(
            name=str(output_dir / name),
            tau_ia=tau_ce,
            **model_kwargs
        )
        sz.run(simtime, overwrite=True)
        hist = vice.history(str(output_dir/name))
        axs[1].plot(hist['lookback'], hist['[ce/mg]'], color='w', linewidth=2)
        axs[1].plot(hist['lookback'], hist['[ce/mg]'], linestyle='-', 
                    color=colors[i], label='%s Gyr' % tau_ce
        )
    axs[1].legend(
        title=r'$\tau_{\rm NSM}$', 
        loc='upper left', 
        bbox_to_anchor=(1, 1)
    )

    # Different outflow mass-loading factors
    eta_list = [5, 2.5, 1, 0]
    colors = [paultol.bright.colors[c] for c in [1, 0, 2, 3]]
    for i, eta in enumerate(eta_list):
        name = f'3p-eta{eta}'
        model_kwargs['eta'] = eta
        sz = vice.singlezone(
            name=str(output_dir / name),
            tau_ia=DELAYED_CE_TIMESCALE,
            **model_kwargs
        )
        sz.run(simtime, overwrite=True)
        hist = vice.history(str(output_dir/name))
        axs[2].plot(hist['lookback'], hist['[ce/mg]'], color='w', linewidth=2)
        axs[2].plot(hist['lookback'], hist['[ce/mg]'], linestyle='-', 
                    color=colors[i], label=eta)
    axs[2].legend(title=r'$\eta$', loc='upper left', bbox_to_anchor=(1, 1))

    # inset SFR plot
    axins = inset_axes(
        axs[3], width='100%', height='100%',
        loc='lower left',
        bbox_to_anchor=(1.13, 0, 0.33, 0.33),
        bbox_transform=axs[3].transAxes,
        borderpad=0,
    )
    axins.set_xlabel('Age [Gyr]')
    axins.set_title('SFR')
    # Different SFHs
    funcs = [exprise, constant, expfall, lateburst]
    names = ['exprise', 'constant', 'expfall', 'lateburst']
    labels = ['Rising', 'Constant', 'Falling', 'Burst']
    colors = [paultol.bright.colors[c] for c in [1, 2, 0, 3]]
    model_kwargs['eta'] = 2.5
    for i, name in enumerate(names):
        fullname = f'3p-{name}'
        model_kwargs['func'] = normalize(funcs[i])
        sz = vice.singlezone(
            name=str(output_dir / fullname),
            tau_ia=DELAYED_CE_TIMESCALE,
            **model_kwargs
        )
        sz.run(simtime, overwrite=True)
        hist = vice.history(str(output_dir/fullname))
        axs[3].plot(hist['lookback'], hist['[ce/mg]'], color='w', linewidth=2)
        axs[3].plot(hist['lookback'], hist['[ce/mg]'], linestyle='-', 
                    color=colors[i], label=labels[i])
        axins.plot(hist['lookback'], hist['sfr'], color=colors[i])
    axs[3].legend(title='SFH', loc='upper left', bbox_to_anchor=(1, 1))
    axins.set_xlim((0, 12))
    axins.set_ylim((0, 0.2))

    axs[0].set_xlim((0, 12))
    axs[0].set_ylim((-1, 1))

    for ax in axs:
        ax.set_ylabel(r'[Ce/Mg]$_{\rm corr}$')
    axs[-1].set_xlabel('Age [Gyr]')

    fig.savefig(paths.figures / 'delayed_enrichment_models')


if __name__ == '__main__':
    main()
