"""
This script plots the mass and metallicity dependence of AGB yields predicted
from a simple stellar population (SSP) model.
"""
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter
import vice

import paths
from plotting import TWO_COLUMN_WIDTH, colored_text_legend
from colormaps import paultol
from multizone._globals import END_TIME
from multizone.src.yields.utils import adjusted_agb
from multizone.src.dtds import plateau

SOLAR_Z = 0.014

def main(style='paper'):
    plt.style.use(paths.styles / f'{style}.mplstyle')
    plt.rcParams['axes.prop_cycle'] = plt.cycler('color', paultol.bright.colors)
    fig, axs = plt.subplots(
        1, 3, 
        figsize=(TWO_COLUMN_WIDTH, 0.33*TWO_COLUMN_WIDTH), 
        constrained_layout=True
    )

    # Non-AGB yields
    vice.yields.ccsne.settings['fe'] = 0
    vice.yields.ccsne.settings['ce'] = 0
    # Mean SN Ia Fe yield (Msun)
    mfeia = 0.7
    # Rate of SNe Ia per solar mass of stars
    Ria = 1.3e-3 # Maoz & Graur (2017)
    vice.yields.sneia.settings['fe'] = Ria * mfeia # absolute scale is arbitrary

    # First panel: IMF-weighted yields
    labels = ['C11+C15', 'KL16+K18']
    logprefactor = 8
    for i, study in enumerate(['cristallo11', 'karakas16']):
        interp = adjusted_agb('ce', study=study)
        masses = interp.masses[1:] # exclude 0
        yields = len(masses) * [0]
        for j in range(len(yields)):
            yields[j] = 10**logprefactor * interp(masses[j], SOLAR_Z) * masses[j]**-1.3
        axs[0].plot(masses, yields, marker='.', label=labels[i])

    axs[0].set_xlim((0.5, 6.5))
    axs[0].set_ylim((0, None))
    axs[0].set_xlabel(r'$M_{\rm ZAMS}\,[M_\odot]$')
    axs[0].set_ylabel(r'IMF-weighted Ce Yield [$\times10^{-%s}$]' % logprefactor)
    axs[0].xaxis.set_minor_locator(MultipleLocator(0.5))
    axs[0].yaxis.set_minor_locator(MultipleLocator(0.1))
    colored_text_legend(axs[0])

    # Second panel: cumulative enrichment as a function of time 
    # for a Solar metallicity SSP
    agb_studies = ['cristallo11', 'karakas16', 'cristallo11', 'karakas16']
    mass_scales = [1, 1, 0.5, 0.5]
    linestyles = ['-', '-', '--', '--']
    colors = [paultol.bright.colors[i] for i in [0, 1, 0, 1]]
    markers = ['o', 'o', 's', 's']
    ms = 3
    for i in range(len(agb_studies)):
        vice.yields.agb.settings['ce'] = adjusted_agb('ce', study=agb_studies[i], mscale=mass_scales[i])
        mass, time = vice.single_stellar_population('ce', Z=0.014, time=END_TIME, dt=1e-3)
        mass = [_ / mass[-1] for _ in mass]
        axs[1].plot(time, mass, ls=linestyles[i], c=colors[i])
        # Plot median enrichment times
        idx = 0
        while mass[idx] < 0.5: idx += 1
        axs[1].plot(time[idx], 1.0, markers[i], c=colors[i], markersize=ms)

    # Compare against Fe from SNe Ia
    mass, time = vice.single_stellar_population('fe', time=END_TIME, dt=1e-3, RIa=plateau(), delay=0.04)
    mass = [_ / mass[-1] for _ in mass]
    axs[1].plot(time, mass, ':', c='gray')
    # Plot median enrichment time
    idx = 0
    while mass[idx] < 0.5: idx += 1
    axs[1].plot(time[idx], 1.0, '^', c='gray', markersize=ms)

    axs[1].set_xscale('log')
    axs[1].set_xlim((4e-2, 20))
    axs[1].set_xlabel('Age [Gyr]')
    axs[1].set_ylabel(r'$M_{\rm Ce}/M_{\rm Ce,final}$')
    axs[1].yaxis.set_minor_locator(MultipleLocator(0.05))
    axs[1].xaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:g}'.format(y)))
    axs[1].text(
        0.47, 0.5, r'$M_{\rm AGB}\times1$', 
        transform=axs[1].transAxes, 
        ha='right', va='bottom',
        rotation=75,
    )
    axs[1].text(
        0.62, 0.45, r'$R_{\rm Ia}$', 
        transform=axs[1].transAxes, 
        ha='right', va='bottom',
        rotation=55,
    )
    axs[1].text(
        0.85, 0.2, r'$M_{\rm AGB}\times0.5$', 
        transform=axs[1].transAxes, 
        ha='right', va='bottom',
        rotation=75
    )

    # Third panel: total enrichment from an SSP as a function of metallicity
    mstar = 1e6
    logzvals = np.linspace(-2, 0.6, 1000)
    logprefactor = 8
    for i, study in enumerate(['cristallo11', 'karakas16']):
        vice.yields.agb.settings['ce'] = adjusted_agb('ce', study=study)
        color = paultol.bright.colors[i]
        # Get interpolation
        mass_yields = len(logzvals) * [0]
        for j in range(len(logzvals)):
            mass, times = vice.single_stellar_population(
                'ce', 
                Z=SOLAR_Z * 10**logzvals[j], 
                time=END_TIME,
                dt=1e-2,
                mstar=mstar
            )
            mass_yields[j] = mass[-1] / mstar * 10**logprefactor
        y, m, z = vice.yields.agb.grid('ce', study = study)
        # Indicate extrapolated yields
        idx_lower = -1
        idx_upper = -1
        for j in range(len(logzvals)):
            if idx_lower == -1 and SOLAR_Z * 10**logzvals[j] > z[0]: idx_lower = j
            if idx_upper == -1 and SOLAR_Z * 10**logzvals[j] > z[-1]: idx_upper = j
        if idx_lower == -1: idx_lower = 0
        if idx_upper == -1: idx_upper = len(logzvals) - 1
        axs[2].plot(logzvals[:idx_lower], mass_yields[:idx_lower], ':', c=color)
        axs[2].plot(logzvals[idx_lower:idx_upper], mass_yields[idx_lower:idx_upper], '-', c=color)
        axs[2].plot(logzvals[idx_upper:], mass_yields[idx_upper:], ':', c=color)
        # Plot nearest grid points
        grid_logz = []
        grid_yields = []
        for j in range(len(z)):
            logz = np.log10(z[j] / SOLAR_Z)
            diff = [abs(_ - logz) for _ in logzvals]
            idx = diff.index(min(diff))
            grid_logz.append(logzvals[idx])
            grid_yields.append(mass_yields[idx])
        axs[2].plot(grid_logz, grid_yields, '.', color=color)

    axs[2].set_xlim((-2.1, 0.6))
    axs[2].set_ylim((0, 1.5))
    axs[2].set_xlabel(r'$\log_{10}(Z/Z_\odot)$')
    axs[2].set_ylabel(r'$M_{\rm Ce}/M_\star\,[\times10^{-%s}]$' % logprefactor)
    axs[2].xaxis.set_minor_locator(MultipleLocator(0.2))
    axs[2].yaxis.set_major_locator(MultipleLocator(0.5))
    axs[2].yaxis.set_minor_locator(MultipleLocator(0.1))

    plt.savefig(paths.figures / 'ssp_yields')
    plt.close()
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Plot the mass and metallicity dependence of AGB yields.'
    )
    parser.add_argument('--style',
        choices=('paper', 'poster'),
        default='paper',
        help='Plot style to use (default: paper).'
    )
    args = parser.parse_args()
    main(**vars(args))
