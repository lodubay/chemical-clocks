"""
Plot [Ce/Mg] evolution predicted by one-zone models with delayed Ce enrichment.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import vice

from multizone.src.yields.utils import adjusted_agb
from onezone_sfh import normalize, expfall, exprise, constant, lateburst
from plotting import latex_float, ONE_COLUMN_WIDTH
from colormaps import paultol
import paths

# CCSN and SN Ia yields
from yields import yZ1

SFH_TIMESCALE = 15
AGB_STUDY = 'cristallo11'
END_TIME = 12 # Gyr
AGB_YIELD_SCALE = 1
AGB_MASS_SHIFT = 0
CCSN_CE_YIELD = 0
DELAYED_CE_YIELD = 3e-9
DELAYED_CE_TIMESCALE = 5
ETA_SUN = 0.4 # default mass-loading factor at Solar radius


def main(style='paper'):
    plt.style.use(paths.styles / f'{style}.mplstyle')
    plt.rcParams['axes.prop_cycle'] = plt.cycler('color', paultol.bright.colors)
    
    # Select Solar neighborhood & Solar metallicity stars only
    mwm_rgb = pd.read_csv(paths.data / 'MWM' / 'sample.csv')
    mwm_rgb = mwm_rgb[mwm_rgb['good_age']].copy()
    local_sample = mwm_rgb[
        (mwm_rgb['Rg'] >= 7.5) &
        (mwm_rgb['Rg'] < 8.5) &
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
        ax.errorbar(
            10, 0.8, 
            xerr=[[age_err_low], [age_err_high]], 
            yerr=med_abund_err, 
            c=datacolor, capsize=0,
        )

    # Plot onezone models
    vice.yields.ccsne.settings['ce'] = CCSN_CE_YIELD
    vice.yields.sneia.settings['ce'] = 0
    vice.yields.agb.settings['ce'] = adjusted_agb(
        'ce', study=AGB_STUDY, amp=AGB_YIELD_SCALE, dm=AGB_MASS_SHIFT,
    )
    output_dir = paths.data / 'onezone'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Different delayed enrichment scales
    delayed_ce_yields = [1e-8, 3e-9, 1e-9, 0]
    colors = [paultol.bright.colors[c] for c in [1, 0, 2, 3]]
    for i, yld in enumerate(delayed_ce_yields):
        vice.yields.sneia.settings['ce'] = yld
        name = f'3p-delayed-ce{int(yld*1e9)}'
        run_singlezone(name, expfall)
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
        run_singlezone(name, expfall, tau_ia=tau_ce)
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
    eta_list = [1, 0.4, 0.2, 0]
    colors = [paultol.bright.colors[c] for c in [1, 0, 2, 3]]
    for i, eta in enumerate(eta_list):
        name = f'3p-eta{eta}'
        run_singlezone(name, expfall, eta=eta)
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
    for i, name in enumerate(names):
        fullname = f'3p-{name}'
        run_singlezone(fullname, funcs[i])
        hist = vice.history(str(output_dir/fullname))
        axs[3].plot(hist['lookback'], hist['[ce/mg]'], color='w', linewidth=2)
        axs[3].plot(hist['lookback'], hist['[ce/mg]'], linestyle='-', 
                    color=colors[i], label=labels[i])
        axins.plot(hist['lookback'], hist['sfr'], color=colors[i])
    axs[3].legend(title='SFH', loc='upper left', bbox_to_anchor=(1, 1))
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

    fig.savefig(paths.figures / 'delayed_enrichment_models')


def run_singlezone(name, sfh, mode='sfr', eta=ETA_SUN, tau_ia=DELAYED_CE_TIMESCALE, output_dir=paths.data/'onezone'):
    dt = 0.01
    simtime = np.arange(0, END_TIME+dt, dt)
    sz = vice.singlezone(
        name=str(output_dir / name),
        func=normalize(sfh),
        mode=mode,
        elements=('fe', 'mg', 'ce'),
        IMF='kroupa',
        eta=eta,
        delay=0.01,
        RIa='exp',
        tau_ia=tau_ia,
        tau_star=2,
        dt=dt,
    )
    sz.run(simtime, overwrite=True)


if __name__ == '__main__':
    main()
