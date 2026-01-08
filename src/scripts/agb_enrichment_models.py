"""
Plot [Ce/Mg] evolution predicted by one-zone GCE models with varying parameters.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import vice

from utils import alpha_cut, adjusted_agb, good_ages
from colormaps import paultol
import paths
import _globals

# CCSN and SN Ia yields
from yields import yZ1

SFH_TIMESCALE = 15
AGB_STUDY = 'cristallo11'
END_TIME = 12 # Gyr
SOLAR_CE_S_FRAC = 0.77 # fraction of Ce in the Sun from the s-process
SOLAR_AGE = 4.6 # Gyr
ETA_SUN = 0.4 # default mass-loading factor at Solar radius


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

    # Median errors
    age_err_low = np.median(local_sample['age'] - local_sample['e_n_age'])
    age_err_high = np.median(local_sample['e_p_age'] - local_sample['age'])
    med_abund_err = local_sample['e_ce_mg'].median()

    figwidth = _globals.ONE_COLUMN_WIDTH
    fig, axs = plt.subplots(
        4, figsize=(figwidth, 2.67 * figwidth), 
        sharex=True, sharey=True,
        gridspec_kw={'hspace': 0.}
    )
    fig.subplots_adjust(right=0.67)
    legend_kwargs = dict(bbox_to_anchor=(1, 1), loc='upper left')

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
        ax.errorbar(
            10, 0.8, 
            xerr=[[age_err_low], [age_err_high]], 
            yerr=med_abund_err, 
            c=datacolor, capsize=0,
        )
        # indicate Solar value (s-process only)
        ax.plot(SOLAR_AGE, np.log10(SOLAR_CE_S_FRAC), 'wo', zorder=9)
        ax.text(
            SOLAR_AGE, np.log10(SOLAR_CE_S_FRAC), r'$\odot$',
            va='center', ha='center', zorder=10, weight='bold', usetex=True
        )

    # Plot onezone models
    vice.yields.sneia.settings['ce'] = 0
    output_dir = paths.data / 'onezone'
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
        name = f'2p-agb-x{scale}'
        run_singlezone(name, expfall)
        hist = vice.history(str(paths.data/output_dir/name))
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
    name = f'2p-agb-x1-norproc'
    run_singlezone(name, expfall)
    hist = vice.history(str(paths.data/output_dir/name))
    axs[0].plot(hist['lookback'], hist['[ce/mg]'], color='w', linewidth=2)
    axs[0].plot(hist['lookback'], hist['[ce/mg]'], linestyle='--', 
                color=paultol.bright.colors[0], label=r'No $r$-proc.'
    )
    axs[0].legend(title='AGB yields', **legend_kwargs)

    # Mass-shifted AGB enrichment
    mass_shifts = [0, -0.5, -1, -1.5, -2]
    colors = [paultol.bright.colors[c] for c in [0, 4, 2, 3, 1]]
    vice.yields.ccsne.settings['ce'] = ccsn_ce_yield
    for i, dm in enumerate(mass_shifts):
        vice.yields.agb.settings['ce'] = adjusted_agb(
            'ce', study=AGB_STUDY, dm=dm, amp=1
        )
        name = f'2p-agb-dm{dm}'
        run_singlezone(name, expfall)
        hist = vice.history(str(paths.data/output_dir/name))
        axs[1].plot(hist['lookback'], hist['[ce/mg]'], color='w', linewidth=2)
        axs[1].plot(hist['lookback'], hist['[ce/mg]'], linestyle='-', 
                    color=colors[i], 
                    label=r'$%s$ M$_\odot$' % dm
        )
    axs[1].legend(title='AGB masses', **legend_kwargs)


    # Different outflow mass-loading factors
    vice.yields.ccsne.settings['ce'] = ccsn_ce_yield
    vice.yields.agb.settings['ce'] = adjusted_agb(
        'ce', study=AGB_STUDY, amp=1
    )
    eta_list = [1, 0.4, 0.2, 0]
    colors = [paultol.bright.colors[c] for c in [1, 0, 2, 3]]
    for i, eta in enumerate(eta_list):
        name = f'2p-eta{eta}'
        run_singlezone(name, expfall, eta=eta)
        hist = vice.history(str(paths.data/output_dir/name))
        axs[2].plot(hist['lookback'], hist['[ce/mg]'], color='w', linewidth=2)
        axs[2].plot(hist['lookback'], hist['[ce/mg]'], linestyle='-', 
                    color=colors[i], label=eta)
    axs[2].legend(title=r'$\eta$', **legend_kwargs)
    
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
    # Different star formation histories
    vice.yields.ccsne.settings['ce'] = ccsn_ce_yield
    vice.yields.agb.settings['ce'] = adjusted_agb(
        'ce', study=AGB_STUDY, amp=1
    )
    funcs = [exprise, constant, expfall, lateburst]
    names = ['exprise', 'constant', 'expfall', 'lateburst']
    labels = ['Rising', 'Constant', 'Falling', 'Burst']
    colors = [paultol.bright.colors[c] for c in [1, 2, 0, 3]]
    for i, name in enumerate(names):
        fullname = f'2p-{name}'
        run_singlezone(fullname, funcs[i])
        hist = vice.history(str(paths.data/output_dir/fullname))
        axs[3].plot(hist['lookback'], hist['[ce/mg]'], color='w', linewidth=2)
        axs[3].plot(hist['lookback'], hist['[ce/mg]'], linestyle='-', 
                    color=colors[i], label=labels[i])
        axins.plot(hist['lookback'], hist['sfr'], color=colors[i])
    axs[3].legend(title='SFH', **legend_kwargs)
    axins.set_xlim((0, END_TIME))
    axins.set_ylim((0, 0.2))

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
    axs[3].set_xlabel('Age [Gyr]')

    fig.savefig(paths.figures / 'agb_enrichment_models')


def run_singlezone(name, sfh, mode='sfr', eta=ETA_SUN, output_dir=paths.data/'onezone'):
    dt = 0.01
    simtime = np.arange(0, END_TIME+dt, dt)
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
    mean = 8
    std = 1
    gauss = amplitude * np.exp(-(time - mean)**2 / (2 * std**2))
    return expfall(time) * (1 + gauss)

def normalize(func):
    dt = 0.01
    simtime = np.arange(0, END_TIME+dt, dt)
    integral = np.sum(dt * func(simtime))
    f = lambda t: 1/integral * func(t)
    return f


if __name__ == '__main__':
    main()
