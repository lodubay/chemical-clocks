"""
This script plots spectral windows around Ce II lines for a handful stars.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from astropy.table import Table
from sdss_access import Access

import paths
from colormaps import paultol
from plotting import TWO_COLUMN_WIDTH
from mwm_sample import LOGG_CUT
from utils import vac2air

# List of Ce II lines used to calculate abundance in ASPCAP
# from Cunha et al. (2017)
CE_II_LINES = [
    # 15277.65, 
    15784.75, 
    # 15958.40, 
    # 15977.12, 
    # 16327.32, 
    16376.48, 
    16595.18, 
    # 16722.51
] # Angstroms

# Approximate bounds of Ce window masks
# https://data.sdss5.org/sas/sdsswork/mwm/spectro/astra/component_data/aspcap/masks_ipl3/Ce.mask
CE_WINDOWS = [
    (15783.55, 15786.35),
    (16375.25, 16377.50),
    (16593.9, 16596.4)
] # Angstroms

def main(style='paper', verbose=True, overwrite=False):
    mwm_sample = pd.read_csv(paths.data / 'MWM' / 'sample.csv', index_col='sdss_id')
    plt.rcParams['axes.prop_cycle'] = plt.cycler('color', paultol.bright.colors)
    colors = paultol.bright.colors
    markers = ['o', '^', 'd', 's', '*', 'p', 'P']

    # Get spectra (all have S/N ~ 200)
    sdss_id_list = [
        58834996, # logg~3
        116336280, # logg~2.5
        55254073, # logg~2
        70979365, # [Ce/Fe]~0, [Fe/H]~-1, logg~2
        62793899, # [Ce/Fe]~-0.3, [Fe/H]~-0.5, logg~2
        96579887, # [Ce/Fe]~0, [Fe/H]~+0.3, logg~2
    ]
    access = Access(release='dr19')
    access.remote()
    # Check for spectrum files
    download = False
    for sdss_id in sdss_id_list:
        if verbose: print(f'{sdss_id}: checking for downloaded spectra...')
        access_kwargs = dict(v_astra='0.6.0', component='', sdss_id=sdss_id)
        mwmStar_filename = access.full('mwmStar', **access_kwargs)
        if not access.exists('', full=mwmStar_filename) or overwrite:
            access.add('mwmStar', **access_kwargs)
            download = True
            if verbose: print('\tAdding data file to download list...')
        else:
            if verbose: print('\tFound data file!')
        astraStar_filename = access.full('astraStarASPCAP', **access_kwargs)
        if not access.exists('', full=astraStar_filename) or overwrite:
            access.add('astraStarASPCAP', **access_kwargs)
            download = True
            if verbose: print('\tAdding model file to download list...')
        else:
            if verbose: print('\tFound model file!')
    # Download spectra if needed
    if download:
        if verbose: print('Downloading spectra...')
        access.set_stream()
        access.commit()

    plt.style.use(paths.styles / f'{style}.mplstyle')
    fig = plt.figure(figsize=(TWO_COLUMN_WIDTH, 0.6*TWO_COLUMN_WIDTH))
    gs1 = GridSpec(1, 3, figure=fig, left=0.08, right=0.72, wspace=0.2)

    # Plot spectral windows
    ax0 = fig.add_subplot(gs1[0])
    ax1 = fig.add_subplot(gs1[1])
    ax2 = fig.add_subplot(gs1[2])
    spec_axs = np.array([ax0, ax1, ax2])
    for i, ax in enumerate(spec_axs):
        ax.set_title('%s Å' % CE_II_LINES[i])
        ax.axvline(
            CE_II_LINES[i],
            color=paultol.bright.colors[-1], 
            linestyle='--'
        )
        ax.set_xlim(CE_WINDOWS[i])
        ax.set_xlabel('Wavelength [Å]')
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.xaxis.set_minor_locator(MultipleLocator(0.2))
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
        ax.yaxis.set_minor_locator(MultipleLocator(0.02))
    ax0.set_ylabel('Relative Flux + Offset')

    # Flux offsets per spectrum for each panel
    offsets = [
        [0.175, 0.1, 0.05, -0.10, -0.15, -0.25],
        [0.2, 0.15, 0.1, -0.05, -0.1, -0.125],
        [0.2, 0.15, 0.1, 0.025, -0.025, -0.125]
    ]
    speclabels = ['(i)', '(ii)', '(iii)', '(iv)', '(v)', '(vi)']

    # Cycle through SDSS IDs
    for i, sdss_id in enumerate(sdss_id_list):
        # Determine telescope - APO or LCO
        hdu = {
            'apo25m': 3,
            'lco25m': 4
        }[mwm_sample.loc[sdss_id].telescope]

        # Get spectrum data
        access_kwargs = dict(v_astra='0.6.0', component='', sdss_id=sdss_id)
        mwmStar_filename = access.full('mwmStar', **access_kwargs)
        mwmStar = Table.read(mwmStar_filename, hdu=hdu)
        astraStar_filename = access.full('astraStarASPCAP', **access_kwargs)
        astraStar = Table.read(astraStar_filename, hdu=hdu)

        # Data conversions
        wl_arr = vac2air(mwmStar['wavelength'][0]) # wavuum - air wavelength conversion
        obs_flux = mwmStar['flux'][0] / astraStar['continuum'][0] # continuum normalization
        obs_flux_err = mwmStar['ivar'][0]**-0.5 / astraStar['continuum'][0] # flux error calculation
        model_ce = astraStar['model_flux_ce_h'][0] # model fit to Ce abundance
        model_global = astraStar['model_flux'][0] # global model fit

        # Plot mutliple spectral windows
        for j, ax in enumerate(spec_axs):
            offset = offsets[j][i]
            line = CE_II_LINES[j]
            wl_range = CE_WINDOWS[j]
            mask = (wl_arr >= wl_range[0]-0.2) & (wl_arr < wl_range[1]+0.2)
            ax.errorbar(
                wl_arr[mask], obs_flux[mask]+offset, 
                yerr=obs_flux_err[mask], 
                color=colors[i],
                marker=markers[i], ms=4, mfc='w', capsize=0, linestyle='none',
            )
            ax.plot(
                wl_arr[mask], model_ce[mask]+offset, 
                c=colors[i], 
                label=sdss_id
            )
            ax.plot(
                wl_arr[mask], model_global[mask]+offset,
                linestyle='--', linewidth=0.5, c=colors[i]
            )
            ax.ticklabel_format(style='plain', axis='x', useOffset=False)
            # Individual spectrum labels in right-hand panel
            if j==0:
                ax.text(
                    wl_arr[mask][2]+0.05, 
                    max(obs_flux[mask][2], model_ce[mask][2])+offset+0.015,
                    speclabels[i],
                    color=colors[i],
                    ha='center', va='bottom'
                )
    
    # Custom legend
    ax0.set_ylim((None, 1.38))
    handles = [
        Line2D([0], [0], ls='none', marker='o', ms=4, mec='k', mfc='w'),
        Line2D([0], [0], c='k'),
        Line2D([0], [0], c='k', ls='--', lw=0.5),
    ]
    labels = ['Observed', '[Ce/H] fit', 'Global fit']
    ax0.legend(handles, labels, loc='upper right', frameon=True)

    # Plot Kiel diagram
    gs2 = GridSpec(2, 1, left=0.78, right=0.98, top=1., hspace=0.25)
    kiel_ax = fig.add_subplot(gs2[0])
    norm = Normalize(vmin=1, vmax=600)
    hexbin_kwargs = dict(
        gridsize=50,
        cmap='binary_r', 
        norm=norm,
        linewidths=0.01,
        # reduce_C_function=logsum,
        mincnt=1
    )
    hb0 = kiel_ax.hexbin(
        mwm_sample['teff'], mwm_sample['logg'], 
        extent=[4000, 5400, LOGG_CUT[0], LOGG_CUT[1]],
        **hexbin_kwargs
    )
    # Indicate example stars on kiel diagram
    ms = 20
    facecolor = 'w'
    for i, sdss_id in enumerate(sdss_id_list):
        kiel_ax.scatter(
            mwm_sample.loc[sdss_id]['teff'], 
            mwm_sample.loc[sdss_id]['logg'],
            facecolors=facecolor,
            edgecolors=colors[i], 
            marker=markers[i],
            s=ms,
        )
    kiel_ax.set_xlim((5400, 4000))
    kiel_ax.set_ylim(LOGG_CUT)
    kiel_ax.yaxis.set_inverted(True)
    kiel_ax.xaxis.set_minor_locator(MultipleLocator(100))
    kiel_ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    kiel_ax.set_xlabel(r'$T_{\rm eff}$')
    kiel_ax.set_ylabel(r'$\log(g)$')
    # plt.colorbar(hb0, ax=kiel_ax, label='Number of stars', pad=0)

    # Plot abundance diagram
    cefe_ax = fig.add_subplot(gs2[1])
    xlim = (-1.5, 0.5)
    ylim = (-1.2, 1.2)
    hb1 = cefe_ax.hexbin(
        mwm_sample['fe_h'], mwm_sample['ce_fe'], 
        extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
        **hexbin_kwargs
    )
    # Indicate example stars on kiel diagram
    for i, sdss_id in enumerate(sdss_id_list):
        cefe_ax.scatter(
            mwm_sample.loc[sdss_id]['fe_h'], 
            mwm_sample.loc[sdss_id]['ce_fe'],
            facecolors=facecolor,
            edgecolors=colors[i], 
            marker=markers[i],
            s=ms,
        )
    cefe_ax.set_xlim(xlim)
    cefe_ax.set_ylim(ylim)
    cefe_ax.xaxis.set_major_locator(MultipleLocator(0.5))
    cefe_ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    cefe_ax.yaxis.set_major_locator(MultipleLocator(0.5))
    cefe_ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    cefe_ax.set_xlabel('[Fe/H]')
    cefe_ax.set_ylabel('[Ce/Fe]', labelpad=-4)
    plt.colorbar(hb1, ax=[kiel_ax, cefe_ax], label='Number of stars', pad=0.02, location='top', extend='max')

    if verbose:
        print(mwm_sample.loc[sdss_id_list][['obj', 'snr', 'logg', 'm_h_atm', 'alpha_m_atm', 'fe_h', 'ce_fe']])
    
    plt.savefig(paths.figures / 'ce_lines')

if __name__ == '__main__':
    main()
