"""
Plot [Ce/Mg] evolution predicted by one-zone GCE models with varying 
star formation history parameters.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import vice

from multizone.src.yields.utils import adjusted_agb
from utils import alpha_cut, good_ages
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
    fig.subplots_adjust(right=0.65)
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

    # Plot onezone models
    vice.yields.sneia.settings['ce'] = 0
    output_dir = paths.data / 'onezone' / 'sfh'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate prompt Ce enrichment (assigned to CCSNe for convenience)
    ccsn_ce_yield = vice.yields.ccsne.settings['mg'] * (
        (1 - SOLAR_CE_S_FRAC) * vice.solar_z['ce'] / vice.solar_z['mg']
    )
    vice.yields.ccsne.settings['ce'] = ccsn_ce_yield
    # Standard AGB assumptions
    vice.yields.agb.settings['ce'] = adjusted_agb('ce', study=AGB_STUDY)

    # Different star formation efficiencies
    taustar_list = [1, 2, 5, 10, 20]
    colors = [paultol.bright.colors[c] for c in [4, 0, 2, 3, 1]]
    for i, taustar in enumerate(taustar_list):
        name = f'taustar{taustar}'
        run_singlezone(name, expfall, tau_star=taustar, output_dir=output_dir)
        hist = vice.history(str(output_dir/name))
        axs[0].plot(hist['lookback'], hist['[ce/mg]'], color='w', linewidth=2)
        axs[0].plot(hist['lookback'], hist['[ce/mg]'], linestyle='-', 
                    color=colors[i], label=f'{taustar} Gyr')
    handles, labels = axs[0].get_legend_handles_labels()
    axs[0].legend(handles[::-1], labels[::-1], title=r'$\tau_\star$', **legend_kwargs)

    # Different outflow mass-loading factors
    vice.yields.agb.settings['ce'] = adjusted_agb(
        'ce', study=AGB_STUDY, amp=1
    )
    eta_list = [1, 0.4, 0.2, 0]
    colors = [paultol.bright.colors[c] for c in [1, 0, 2, 3]]
    for i, eta in enumerate(eta_list):
        name = f'eta{eta}'
        run_singlezone(name, expfall, eta=eta, output_dir=output_dir)
        hist = vice.history(str(output_dir/name))
        axs[1].plot(hist['lookback'], hist['[ce/mg]'], color='w', linewidth=2)
        axs[1].plot(hist['lookback'], hist['[ce/mg]'], linestyle='-', 
                    color=colors[i], label=eta)
    axs[1].legend(title=r'$\eta$', **legend_kwargs)
    
    # Different star formation histories
    # inset SFR plot
    axins = axs[2].inset_axes(
        bounds=(1.15, 0, 0.33, 0.33),
        transform=axs[2].transAxes,
    )
    axins.set_xlabel('Age [Gyr]', fontsize='small')
    axins.set_title('SFR', fontsize='small', y=0.8, pad=0.00001)
    funcs = [exprise, constant, expfall, lateburst]
    names = ['exprise', 'constant', 'expfall', 'lateburst']
    labels = ['Rising', 'Constant', 'Falling', 'Burst']
    colors = [paultol.bright.colors[c] for c in [1, 2, 0, 3]]
    for i, name in enumerate(names):
        fullname = name
        run_singlezone(fullname, funcs[i], output_dir=output_dir)
        hist = vice.history(str(output_dir/fullname))
        axs[2].plot(hist['lookback'], hist['[ce/mg]'], color='w', linewidth=2)
        axs[2].plot(hist['lookback'], hist['[ce/mg]'], linestyle='-', 
                    color=colors[i], label=labels[i])
        axins.plot(hist['lookback'], hist['sfr'], color=colors[i])
    axs[2].legend(title=r'\textbf{SFH}', **legend_kwargs)
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
    axs[-1].set_xlabel('Age [Gyr]')

    fig.savefig(paths.figures / 'onezone_sfh')
    plt.close()


def run_singlezone(name, sfh, mode='sfr', eta=ETA_SUN, tau_star=2, output_dir=paths.data/'onezone'):
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
        tau_star=tau_star,
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
    mean = 9
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
