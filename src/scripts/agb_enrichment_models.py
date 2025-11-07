"""
Plot [Ce/Mg] evolution predicted by one-zone GCE models with varying parameters.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import vice

from utils import alpha_cut, amplified_agb
from colormaps import paultol
import paths
import _globals

SFH_TIMESCALE = 15
AGB_STUDY = 'karakas16'
CCSN_CE_YIELD = 3e-9


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
        3, figsize=(figwidth, 2 * figwidth), 
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
    vice.yields.sneia.settings['mg'] = 0.
    vice.yields.sneia.settings['fe'] = 0.0024
    vice.yields.sneia.settings['ce'] = 0
    output_dir = paths.data / 'onezone'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Different AGB yield scales
    yield_scales = [3, 2, 1]
    vice.yields.ccsne.settings['ce'] = CCSN_CE_YIELD
    for i, scale in enumerate(yield_scales):
        vice.yields.agb.settings['ce'] = amplified_agb(
            'ce', study=AGB_STUDY, prefactor=scale
        )
        name = f'2p-agb-x{scale}'
        run_singlezone(name, expfall)
        hist = vice.history(str(paths.data/output_dir/name))
        axs[0].plot(hist['lookback'], hist['[ce/mg]'], color='w', linewidth=2)
        axs[0].plot(hist['lookback'], hist['[ce/mg]'], linestyle='-', 
                    color=paultol.bright.colors[scale-1], 
                    label=r'$\times%s$' % scale
        )
    # no prompt r-process
    vice.yields.agb.settings['ce'] = amplified_agb(
        'ce', study=AGB_STUDY, prefactor=1
    )
    vice.yields.ccsne.settings['ce'] = 0
    name = f'2p-agb-x1-norproc'
    run_singlezone(name, expfall)
    hist = vice.history(str(paths.data/output_dir/name))
    axs[0].plot(hist['lookback'], hist['[ce/mg]'], color='w', linewidth=2)
    axs[0].plot(hist['lookback'], hist['[ce/mg]'], linestyle='--', 
                color=paultol.bright.colors[0], label=r'No $r$-proc.'
    )
    axs[0].legend(title='AGB yields', loc='upper left', bbox_to_anchor=(1, 1))

    # Different outflow mass-loading factors
    vice.yields.ccsne.settings['ce'] = CCSN_CE_YIELD
    vice.yields.agb.settings['ce'] = amplified_agb(
        'ce', study=AGB_STUDY, prefactor=1
    )
    eta_list = [5, 2.5, 1, 0]
    colors = [paultol.bright.colors[c] for c in [1, 0, 2, 3]]
    for i, eta in enumerate(eta_list):
        name = f'2p-eta{eta}'
        run_singlezone(name, expfall, eta=eta)
        hist = vice.history(str(paths.data/output_dir/name))
        axs[1].plot(hist['lookback'], hist['[ce/mg]'], color='w', linewidth=2)
        axs[1].plot(hist['lookback'], hist['[ce/mg]'], linestyle='-', 
                    color=colors[i], label=eta)
    axs[1].legend(title=r'$\eta$', loc='upper left', bbox_to_anchor=(1, 1))
    
    # inset SFR plot
    axins = inset_axes(
        axs[2], width='100%', height='100%',
        loc='lower left',
        bbox_to_anchor=(1.13, 0, 0.33, 0.33),
        bbox_transform=axs[2].transAxes,
        borderpad=0,
    )
    axins.set_xlabel('Age [Gyr]')
    axins.set_title('SFR')
    # Different star formation histories
    vice.yields.ccsne.settings['ce'] = CCSN_CE_YIELD
    vice.yields.agb.settings['ce'] = amplified_agb(
        'ce', study=AGB_STUDY, prefactor=1
    )
    funcs = [exprise, constant, expfall, lateburst]
    names = ['exprise', 'constant', 'expfall', 'lateburst']
    labels = ['Rising', 'Constant', 'Falling', 'Burst']
    colors = [paultol.bright.colors[c] for c in [1, 2, 0, 3]]
    for i, name in enumerate(names):
        fullname = f'2p-{name}'
        run_singlezone(fullname, funcs[i])
        hist = vice.history(str(paths.data/output_dir/fullname))
        axs[2].plot(hist['lookback'], hist['[ce/mg]'], color='w', linewidth=2)
        axs[2].plot(hist['lookback'], hist['[ce/mg]'], linestyle='-', 
                    color=colors[i], label=labels[i])
        axins.plot(hist['lookback'], hist['sfr'], color=colors[i])
    axs[2].legend(title='SFH', loc='upper left', bbox_to_anchor=(1, 1))
    axins.set_xlim((0, 12))
    axins.set_ylim((0, 0.2))

    axs[0].set_xlim((0, 12))
    axs[0].set_ylim((-1, 1))

    titles = ['(a)', '(b)', '(c)']
    for i, ax in enumerate(axs):
        ax.set_ylabel('[Ce/Mg]')
        ax.set_title(titles[i], loc='left', x=0.05, y=0.9, va='top')
    axs[2].set_xlabel('Age [Gyr]')

    fig.savefig(paths.figures / 'agb_enrichment_models')


def run_singlezone(name, sfh, mode='sfr', eta=2.5, output_dir=paths.data/'onezone'):
    dt = 0.01
    simtime = np.arange(0, 12+dt, dt)
    sz = vice.singlezone(
        name=str(output_dir / name),
        func=normalize(sfh),
        mode=mode,
        elements=('fe', 'mg', 'ce'),
        IMF='kroupa',
        eta=eta,
        delay=0.04,
        RIa='plaw',
        tau_star=2,
        dt=dt,
    )
    sz.run(simtime, overwrite=True)


def expfall(time):
    return np.exp(-time/SFH_TIMESCALE)

def exprise(time):
    return np.exp(time/SFH_TIMESCALE)

def constant(time):
    if isinstance(time, np.ndarray):
        return np.ones(time.shape)
    elif isinstance(time, list):
        return [1 for t in time]
    else:
        return 1

def lateburst(time):
    amplitude = 2
    mean = 10
    std = 1
    gauss = amplitude * np.exp(-(time - mean)**2 / (2 * std**2))
    return expfall(time) * (1 + gauss)

def normalize(func):
    dt = 0.01
    simtime = np.arange(0, 12+dt, dt)
    integral = np.sum(dt * func(simtime))
    f = lambda t: 1/integral * func(t)
    return f


if __name__ == '__main__':
    main()
