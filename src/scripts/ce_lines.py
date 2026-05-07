"""
This script plots spectral windows around Ce II lines for a handful stars.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator
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
    15277.65, 
    15784.75, 
    15958.40, 
    15977.12, 
    16327.32, 
    16376.48, 
    16595.18, 
    16722.51
] # Angstroms

def main(style='paper', verbose=True, overwrite=False):
    mwm_sample = pd.read_csv(paths.data / 'MWM' / 'sample.csv', index_col='sdss_id')
    plt.rcParams['axes.prop_cycle'] = plt.cycler('color', paultol.bright.colors)
    colors = paultol.bright.colors
    markers = ['o', '^', 'd', 's', '*', 'p', 'P']

    # Get spectra
    sdss_id_list = [
        76020143, # logg~3
        77434984, # logg~2.5
        72928961, # logg~2
        57957167, # [Ce/Fe]~0, [Fe/H]~-1, logg~3
        61020837, # [Ce/Fe]~-0.3, [Fe/H]~-0.5, logg~3
        89108691, # [Ce/Fe]~0, [Fe/H]~+0.3, logg~3
        # 61604644, # [Ce/Fe]~0, [Fe/H]~-1, logg~2
        # 75933928, # [Ce/Fe]~-0.3, [Fe/H]~-0.5, logg~2
        # 82369483, # [Ce/Fe]~0, [Fe/H]~+0.3, logg~2
    ]
    # rows = [0, 0, 0, 1, 1, 1] # panel rows in which to plot the spectra
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
    gs1 = GridSpec(2, 2, figure=fig, right=0.72, wspace=0.15)

    # Plot spectral windows
    ax1 = fig.add_subplot(gs1[:,0])
    ax2 = fig.add_subplot(gs1[:,1])
    # ax3 = fig.add_subplot(gs1[1,0], sharex=ax1)
    # ax4 = fig.add_subplot(gs1[1,1], sharex=ax2)
    # axs = np.array([[ax1, ax2], [ax3, ax4]])
    ax1.xaxis.set_minor_locator(MultipleLocator(0.2))
    ax2.xaxis.set_minor_locator(MultipleLocator(0.2))
    ax1.set_xlabel('Wavelength [Å]')
    ax2.set_xlabel('Wavelength [Å]')
    ax1.set_ylabel('Relative Flux + Offset')
    # ax3.set_ylabel('Relative Flux + Offset')

    # Flux offsets per spectrum for each panel
    left_offsets = [0.05, 0.025, 0, -0.10, -0.15, -0.2]
    right_offsets = [0.1, 0.05, 0, -0.05, -0.1, -0.2]

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
        # row = axs[rows[i]] # select row to plot in
        for ax, line, offset_list in zip([ax1, ax2], [CE_II_LINES[1], CE_II_LINES[6]], [left_offsets, right_offsets]):
            offset = offset_list[i]
            ax.axvline(line, color=paultol.bright.colors[-1], linestyle='--')
            wl_range = (line-1.5, line+1.5)
            mask = (wl_arr >= wl_range[0]) & (wl_arr < wl_range[1])
            ax.errorbar(
                wl_arr[mask], obs_flux[mask]+offset, 
                yerr=obs_flux_err[mask], 
                color=colors[i],
                marker=markers[i], ms=4, mfc='w', capsize=0, linestyle='none',
                # label=round(logg, 2)
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

    # Plot Kiel diagram
    gs2 = GridSpec(2, 1, left=0.78, right=0.98, hspace=0.25)
    kiel_ax = fig.add_subplot(gs2[0])
    hexbin_kwargs = dict(
        gridsize=50,
        cmap='binary_r', 
        linewidths=0.01,
        # reduce_C_function=logsum,
        mincnt=1
    )
    kiel_ax.hexbin(
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

    # Plot abundance diagram
    cefe_ax = fig.add_subplot(gs2[1])
    xlim = (-1.5, 0.5)
    ylim = (-1.2, 1.2)
    cefe_ax.hexbin(
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
    cefe_ax.xaxis.set_minor_locator(MultipleLocator(0.2))
    cefe_ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    cefe_ax.set_xlabel('[Fe/H]')
    cefe_ax.set_ylabel('[Ce/Fe]', labelpad=-4)
    
    plt.savefig(paths.figures / 'ce_lines')

if __name__ == '__main__':
    main()
