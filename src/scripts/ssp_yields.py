"""
This script plots metallicity dependence of AGB yield predictions
from a simple stellar population (SSP) model.
"""
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter
import vice
from vice.toolkit.interpolation.interp_scheme_2d import interp_scheme_2d
from vice.yields.agb._grid_reader import yield_grid

import paths
from plotting import ONE_COLUMN_WIDTH, colored_text_legend
from colormaps import paultol

SOLAR_Z = 0.014
END_TIME = 13.2 # Gyr

def main(style='paper'):
    plt.style.use(paths.styles / f'{style}.mplstyle')
    plt.rcParams['axes.prop_cycle'] = plt.cycler('color', paultol.bright.colors)
    fig, ax = plt.subplots(
        figsize=(ONE_COLUMN_WIDTH, 0.7*ONE_COLUMN_WIDTH), 
        constrained_layout=True
    )

    # Non-AGB yields
    vice.yields.ccsne.settings['fe'] = 0
    vice.yields.ccsne.settings['ce'] = 0
    vice.yields.sneia.settings['ce'] = 0
    # Mean SN Ia Fe yield (Msun)
    mfeia = 0.7
    # Rate of SNe Ia per solar mass of stars
    Ria = 1.3e-3 # Maoz & Graur (2017)
    vice.yields.sneia.settings['fe'] = Ria * mfeia # absolute scale is arbitrary

    # Third panel: total enrichment from an SSP as a function of metallicity
    mstar = 1e6
    logzvals = np.linspace(-2, 0.6, 1000)
    logprefactor = 8
    labels = ['C11+C15', 'KL16+K18']
    for i, study in enumerate(['cristallo11', 'karakas16']):
        vice.yields.agb.settings['ce'] = agb_interpolator('ce', study=study)
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
        ax.plot(logzvals[:idx_lower], mass_yields[:idx_lower], ':', c=color)
        ax.plot(logzvals[idx_lower:idx_upper], mass_yields[idx_lower:idx_upper], '-', c=color)
        ax.plot(logzvals[idx_upper:], mass_yields[idx_upper:], ':', c=color)
        # Plot nearest grid points
        grid_logz = []
        grid_yields = []
        for j in range(len(z)):
            logz = np.log10(z[j] / SOLAR_Z)
            diff = [abs(_ - logz) for _ in logzvals]
            idx = diff.index(min(diff))
            grid_logz.append(logzvals[idx])
            grid_yields.append(mass_yields[idx])
        ax.plot(grid_logz, grid_yields, '.', color=color, label=labels[i])

    ax.set_xlim((-2.1, 0.6))
    ax.set_ylim((0, 1.5))
    ax.set_xlabel(r'$\log_{10}(Z/Z_\odot)$')
    ax.set_ylabel(r'$M_{\rm Ce}/M_\star\,[\times10^{-%s}]$' % logprefactor)
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    colored_text_legend(ax, loc='upper left', title='Yield tables')

    plt.savefig(paths.figures / 'ssp_yields')
    plt.close()


class agb_interpolator(interp_scheme_2d):
    """
    Custom AGB yield interpolator that forces 0 yield at 0 metallicity,
    0 yield at 0 mass, and enforces non-negative yields.
    
    Inherits from vice.toolkit.interpolator.interp_scheme_2d.
    """
    def __init__(self, element, study='cristallo11'):
        # let the grid reader function do the error handling
        yields, masses, metallicities = yield_grid(element, study=study)
        # enforce yield of 0 at 0 mass
        new_masses = [0] + list(masses)
        # enforce yield of 0 at 0 metallicity
        new_metallicities = [0] + list(metallicities)
        new_yields = [[0] * len(new_metallicities)]
        for row in yields:
            new_yields.append([0] + list(row))
        super().__init__(new_masses, new_metallicities, new_yields)
    
    def __call__(self, mass, metallicity):
        return max(super().__call__(mass, metallicity), 0)
    
    @property
    def masses(self):
        return super().xcoords
    
    @property
    def metallicities(self):
        return super().ycoords
    
    @property
    def yields(self):
        return super().zcoords


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
