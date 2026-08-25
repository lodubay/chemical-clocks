"""
This script plots spectral windows around Ce II lines for a handful stars.
"""
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator, ScalarFormatter
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize, LogNorm
from astropy.table import Table
from sdss_access import Access
access = Access(release='dr19')

import paths
from colormaps import paultol
from plotting import TWO_COLUMN_WIDTH
from sample import LOGG_CUT
from utils import vac2air, import_sample

# List of Ce II lines used to calculate abundance in ASPCAP
# from Cunha et al. (2017)
# Eight are within the APOGEE windows, but only three are used for abundance determination
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

def main(style='paper', verbose=False, overwrite=False):
    plt.style.use(paths.styles / f'{style}.mplstyle')
    savedir = {
        'paper': paths.figures,
        'presentation': paths.extra/'presentation'
    }[style]
    savedir.mkdir(exist_ok=True)
    if verbose: print('Importing MWM sample file...')
    mwm_sample = import_sample(good_ages=False)
    subset = mwm_sample[
        (mwm_sample['snr'] > 180) &
        (mwm_sample['snr'] < 220) &
        (mwm_sample['fe_h'] > -0.25) &
        (mwm_sample['fe_h'] < -0.15) &
        (mwm_sample['ce_fe'] > 0.25) &
        (mwm_sample['ce_fe'] < 0.35) &
        (mwm_sample['logg'] < 1.6) &
        (mwm_sample['logg'] > 1.4)
    ]
    print(subset[['snr', 'fe_h', 'ce_fe', 'logg']])
    # First figure: stars with similar log(g) and metallicity but different Ce
    if verbose: print('\nFigure 1: stellar siblings')
    sdss_id_list = [ # all have S/N~200
        75810381,
        59558349,
        86517081,
        54750610,
        58830163,
        54880428
    ]
    # Flux offsets per spectrum for each panel
    offsets = [
        [0.175, 0.1, 0.025, -0.05, -0.125, -0.25],
        [0.15, 0.1, 0.05, 0.0, -0.05, -0.125],
        [0.2, 0.15, 0.1, 0.025, -0.025, -0.1]
    ]
    plot_spectrum_comparison(
        sdss_id_list, 
        offsets,
        mwm_sample=mwm_sample,
        fname='ce_lines_1', 
        verbose=verbose, 
        overwrite=overwrite,
        savedir=savedir
    )
    # Second figure: exploring different log(g) and metallicity values
    if verbose: print('\nFigure 2: parameter space coverage')
    sdss_id_list = [ # All have S/N~200
        56493283, # logg~3
        116336280, # logg~2.5
        54639740, # logg~1.5
        70979365, # [Ce/Fe]~0, [Fe/H]~-1, logg~2
        62793899, # [Ce/Fe]~-0.3, [Fe/H]~-0.5, logg~2
        96579887, # [Ce/Fe]~0, [Fe/H]~+0.3, logg~2
    ]
    # Flux offsets per spectrum for each panel
    offsets = [
        [0.2, 0.125, 0.1, -0.10, -0.15, -0.25],
        [0.225, 0.175, 0.125, -0.05, -0.1, -0.125],
        [0.2, 0.15, 0.1, 0.025, -0.025, -0.125]
    ]
    plot_spectrum_comparison(
        sdss_id_list, 
        offsets,
        mwm_sample=mwm_sample,
        fname='ce_lines_2',
        verbose=verbose,
        overwrite=overwrite,
        savedir=savedir
    )


def plot_spectrum_comparison(
        sdss_id_list, 
        offsets,
        mwm_sample=None, 
        fname='ce_lines', 
        verbose=False, 
        overwrite=False,
        savedir=paths.figures
    ):
    """
    Plot a figure comparing Ce windows for multiple SDSS spectra.
    """
    download_sdss_spectra(sdss_id_list, verbose=verbose, overwrite=overwrite)

    fig = plt.figure(figsize=(TWO_COLUMN_WIDTH, 0.5*TWO_COLUMN_WIDTH))
    gs1 = GridSpec(1, 3, figure=fig, left=0.08, right=0.72, top=0.85, wspace=0.2)

    # Plot spectral windows
    ax0 = fig.add_subplot(gs1[0])
    ax1 = fig.add_subplot(gs1[1])
    ax2 = fig.add_subplot(gs1[2])
    spec_axs = np.array([ax0, ax1, ax2])
    for i, ax in enumerate(spec_axs):
        ax.set_title(r'$\lambda%s$ Å' % CE_II_LINES[i])
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

    # Cycle through SDSS IDs
    colors = paultol.bright.colors
    markers = ['o', '^', 'd', 's', '*', 'p', 'P']
    speclabels = ['(i)', '(ii)', '(iii)', '(iv)', '(v)', '(vi)']
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
    # ax0.set_ylim((None, 1.44))
    handles = [
        Line2D([0], [0], ls='none', marker='o', ms=4, mec='k', mfc='w'),
        Line2D([0], [0], c='k'),
        Line2D([0], [0], c='k', ls='--', lw=0.5),
    ]
    labels = ['Observed spectrum', '[Ce/H] best-fit model', 'Global best-fit model']
    ax1.legend(handles, labels, loc='lower center', ncols=3, bbox_to_anchor=(0.5, 1.05))

    # Plot Kiel diagram
    gs2 = GridSpec(2, 1, left=0.79, right=0.98, top=1., hspace=0.3)
    kiel_ax = fig.add_subplot(gs2[0])
    norm = LogNorm(vmin=1, vmax=200)
    hexbin_kwargs = dict(
        gridsize=100,
        cmap='binary_r', 
        norm=norm,
        linewidths=0.1,
        mincnt=1,
        rasterized=True
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

    # Plot abundance diagram
    cefe_ax = fig.add_subplot(gs2[1])
    xlim = (-1.5, 0.5)
    ylim = (-1., 1.)
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
    cbar = plt.colorbar(hb1, ax=[kiel_ax, cefe_ax], label='Number of stars', pad=0.02, location='top')#, extend='max')
    cbar.ax.xaxis.set_major_formatter(ScalarFormatter())

    if verbose:
        print(mwm_sample.loc[sdss_id_list][['obj', 'snr', 'logg', 'm_h_atm', 'alpha_m_atm', 'fe_h', 'ce_fe']])
    
    plt.savefig(savedir / fname)


def download_sdss_spectra(sdss_id_list, verbose=True, overwrite=False):
    """
    Download co-added APOGEE spectra and ASPCAP model spectra.
    
    Parameters
    ----------
    sdss_id_list : list of ints
        List of SDSS-V IDs to download. Must have been observed post-DR17.
    verbose : bool, optional
        If True, print verbose output. Default is True.
    overwrite: bool, optional
        If True, overwrite existing downloaded spectra. Default is False.

    Returns
    -------
    None
    """
    access.remote()
    # Check for spectrum files that already exist
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Plot spectral windows for selected MWM stars.'
    )
    parser.add_argument('--style',
        choices=('paper', 'presentation'),
        default='paper',
        help='Plot style to use (default: paper).'
    )
    parser.add_argument('-v', '--verbose',
        action='store_true',
        help='Print verbose output to terminal.'
    )
    parser.add_argument('-o', '--overwrite',
        action='store_true',
        help='Re-download all spectrum files (takes longer).'
    )
    args = parser.parse_args()
    main(**vars(args))
