"""
This script compares the linear fit parameters as a function of guiding radius
and metallicity between two different orbit catalogs: the fiducial catalog
using multiple distance sources, and a smaller catalog using only the
Bailer-Jones et al. (2021) photo-geometric distances.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D

from utils import import_sample, get_bin_centers
from plotting import ONE_COLUMN_WIDTH, RADIUS_COLORMAP, insert_colorbar_axes
from global_metallicity_fits import ZLIM, fit_metallicity_bins
from sample import add_kinematics
import paths

def main(style='paper'):
    # Import MWM sample
    full_sample = import_sample(good_ages=True, cut_limits=True)
    # Merge with photogeometric distance orbit catalog
    full_photogeo = add_kinematics(
        full_sample.reset_index(), 
        fitspath=paths.data / 'catalogs' / 'sdssv-mwm-dr19-apogee-photogeo-actions.fits', 
        suffix='_photogeo'
    )
    full_photogeo.set_index('sdss_id', inplace=True)
    # Metallicity bins
    met_bin_edges = np.arange(-0.85, 0.56, 0.1)
    dr = 2
    radius_bin_edges = np.arange(3, 15+dr, dr)

    # Set up figure
    plt.style.use(paths.styles / f'{style}.mplstyle')
    fig, axs = plt.subplots(
        2,
        figsize=(ONE_COLUMN_WIDTH, 1.5*ONE_COLUMN_WIDTH), 
        sharex=True,
        gridspec_kw={'hspace': 0}
    )
    cmap = plt.get_cmap(RADIUS_COLORMAP)
    norm = BoundaryNorm(radius_bin_edges, cmap.N)
    cax = insert_colorbar_axes(fig, orientation='horizontal', pad=0.05)

    for j in range(len(radius_bin_edges)-1):
        rlim = radius_bin_edges[j:j+2]
        mean_radius = np.mean(rlim)
        color = cmap(norm(mean_radius))
        # First, the fiducial distances
        region = full_photogeo[
            (full_photogeo['Rg'] >= rlim[0]) &
            (full_photogeo['Rg'] < rlim[1]) &
            (full_photogeo['z_max'] >= ZLIM[0]) &
            (full_photogeo['z_max'] < ZLIM[1]) &
            (full_photogeo['high_ia']) # restrict age trends to low-alpha only
        ]
        # Bin by metallicity and fit linear trend to stars
        params, errors, mets = fit_metallicity_bins(region, met_bin_edges)
        # Plot fit slopes
        slopes = params[:,1]
        slope_errs = errors[:,1]
        axs[0].plot(
            mets, slopes, '.-',
            color=color,
            label=f'{int(mean_radius)} kpc'
        )
        axs[0].fill_between(
            mets, slopes - slope_errs, slopes + slope_errs,
            color=color, 
            alpha=0.5, 
            edgecolor='none'
        )
        # Plot intercepts
        intercepts = params[:,0]
        int_errs = errors[:,0]
        axs[1].plot(mets, intercepts, '.-', color=color)
        axs[1].fill_between(
            mets, intercepts - int_errs, intercepts + int_errs,
            color=color,
            alpha=0.5,
            edgecolor='none'
        )
        # Next, the Bailer-Jones distances
        region = full_photogeo[
            (full_photogeo['Rg_photogeo'] >= rlim[0]) &
            (full_photogeo['Rg_photogeo'] < rlim[1]) &
            (full_photogeo['z_max_photogeo'] >= ZLIM[0]) &
            (full_photogeo['z_max_photogeo'] < ZLIM[1]) &
            (full_photogeo['high_ia']) # restrict age trends to low-alpha only
        ]
        # Bin by metallicity and fit linear trend to stars
        params, errors, mets = fit_metallicity_bins(region, met_bin_edges)
        # Plot fit slopes
        slopes = params[:,1]
        slope_errs = errors[:,1]
        axs[0].plot(
            mets, slopes, '--',
            color=color,
            label=f'{int(mean_radius)} kpc'
        )
        # axs[0].fill_between(
        #     mets, slopes - slope_errs, slopes + slope_errs,
        #     color=color, 
        #     alpha=0.5, 
        #     edgecolor='none'
        # )
        # Plot intercepts
        intercepts = params[:,0]
        int_errs = errors[:,0]
        axs[1].plot(mets, intercepts, '--', color=color)
        # axs[1].fill_between(
        #     mets, intercepts - int_errs, intercepts + int_errs,
        #     color=color,
        #     alpha=0.5,
        #     edgecolor='none'
        # )
    # Dotted horizontal line at 0
    axs[0].axhline(0, ls=':', c='gray', zorder=0)
    # indicate Solar value
    axs[1].plot(0, 0, 'wo', zorder=9)
    axs[1].text(
        0, 0, r'$\odot$',
        va='center', ha='center', zorder=10, weight='bold', usetex=True
    )

    # Colorbar
    fig.colorbar(
        ScalarMappable(norm, cmap), 
        cax=cax, 
        orientation='horizontal', 
        label='Guiding radius [kpc]'
    )

    # Legend
    lines = [
        Line2D([0], [0], color='k', linestyle='-'),
        Line2D([0], [0], color='k', linestyle='--'),
    ]
    labels = ['Composite distances', 'Bailer-Jones distances']
    axs[0].legend(lines, labels, loc='upper left')

    # Format axes
    axs[0].set_xlim((-0.7, 0.5))
    axs[0].set_ylim((-0.1, 0.05))
    axs[0].xaxis.set_major_locator(MultipleLocator(0.5))
    axs[0].xaxis.set_minor_locator(MultipleLocator(0.1))
    axs[0].yaxis.set_major_locator(MultipleLocator(0.05))
    axs[0].yaxis.set_minor_locator(MultipleLocator(0.01))
    axs[0].set_ylabel(r'Slope [dex Gyr$^{-1}$]')
    axs[1].set_ylim((-0.2, 0.34))
    axs[1].yaxis.set_major_locator(MultipleLocator(0.1))
    axs[1].yaxis.set_minor_locator(MultipleLocator(0.02))
    axs[1].set_ylabel(r'[Ce/Mg] at $\tau=5$ Gyr')
    axs[1].set_xlabel('[Fe/H]')
    plt.savefig(paths.figures / 'distance_source_sensitivity')
    plt.close()

if __name__ == '__main__':
    main()
