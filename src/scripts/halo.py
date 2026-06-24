"""
This script plots Ce abundances in the Milky Way halo.
"""
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator
from astropy.io import ascii

import paths
from plotting import TWO_COLUMN_WIDTH, colored_text_legend
from colormaps import paultol
from utils import import_sample, get_bin_centers, fits_to_pandas
from contours import plot_kde2D_contours

DENSITY_COLORMAP = 'binary_r'


def main(style='paper'):
    # Get data
    data = import_sample(good_ages=False)
    # Kinematically-selected halo
    halo = data[data['E']/1e5 > halo_ELz_cut(data['Lz']/1e3)]
    # halo = data[(data['z_max'] > 3) | (data['vphi'] > -120)]
    # Kinematically-selected disk stars
    disk = data[data['E']/1e5 < halo_ELz_cut(data['Lz']/1e3)].copy()
    # disk = data[(data['z_max'] < 3) & (data['vphi'] < -120)]
    low_ia = disk[disk['high_alpha']]
    high_ia = disk[disk['low_alpha']]
    # Chemically-selected accreted stars
    accreted = halo[halo['mn_mg'] < halo_chem_cut(halo['al_fe'])]
    # Chemically-selected in-situ halo stars
    insitu = halo[halo['mn_mg'] > halo_chem_cut(halo['al_fe'])]
    
    # Set up figure
    plt.style.use(paths.styles / f'{style}.mplstyle')
    fig = plt.figure(figsize=(0.8*TWO_COLUMN_WIDTH, 0.8*TWO_COLUMN_WIDTH))
    gs0 = GridSpec(1, 2, figure=fig, top=0.98, bottom=0.64, wspace=0.35)
    ax0 = fig.add_subplot(gs0[0])
    ax1 = fig.add_subplot(gs0[1])
    hexbin_kwargs = dict(
        gridsize=(50, 33),
        cmap=DENSITY_COLORMAP, 
        linewidths=0.1,
        reduce_C_function=logsum,
        mincnt=1
    )
    scatter_kwargs = dict(
        s=2, rasterized=True, edgecolors='none',
    )
    # accreted_color = paultol.bright.colors[5]
    # insitu_color = paultol.bright.colors[2]
    # low_ia_color = paultol.bright.colors[1]
    # high_ia_color = paultol.bright.colors[0]
    accreted_color = paultol.vibrant.colors[1]
    insitu_color = paultol.vibrant.colors[2]
    low_ia_color = paultol.bright.colors[1]
    high_ia_color = paultol.bright.colors[0]
    # Italicize "in situ" if LaTeX is installed
    if plt.rcParams['text.usetex']:
        insitu_label = r'\textit{In situ}'
    else:
        insitu_label = 'In situ'

    # Kinematic cut in E-Lz plane
    ax0.set_xlabel(r'$L_z$ [$\times10^3$ kpc km s$^{-1}$]')
    ax0.set_ylabel(r'$E$ [$\times 10^5$ km$^2$ s$^{-2}$]')
    ax0.xaxis.set_major_locator(MultipleLocator(2))
    ax0.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax0.yaxis.set_major_locator(MultipleLocator(0.5))
    ax0.yaxis.set_minor_locator(MultipleLocator(0.1))
    xlim = (-6.5, 4.5)
    ylim = (-2.5, 0)
    ax0.set_xlim(xlim)
    ax0.set_ylim(ylim)
    # Plot disk stars
    pc = ax0.hexbin(
        disk['Lz']/1e3, disk['E']/1e5,
        C=np.ones(disk.shape[0]),
        extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
        **hexbin_kwargs
    )
    fig.colorbar(pc, ax=ax0, label=r'$\log N$ (disk)', pad=0., use_gridspec=True)
    # Plot accreted & in-situ
    ax0.scatter(
        insitu['Lz']/1e3, insitu['E']/1e5,
        c=insitu_color,
        **scatter_kwargs
    )
    ax0.scatter(
        accreted['Lz']/1e3, accreted['E']/1e5,
        c=accreted_color,
        **scatter_kwargs
    )
    # Indicate boundary
    Lz_arr = np.arange(-6.5, 5.1, 0.1)
    ax0.plot(Lz_arr, halo_ELz_cut(Lz_arr), '-', color='k')
    ax0.text(
        1.5, -0.5, 'Halo', 
        fontsize=plt.rcParams['axes.titlesize']
    )
    ax0.text(
        -4, -2, 'Disk', 
        fontsize=plt.rcParams['axes.titlesize']
    )

    # Chemical cut in Al/Fe - Mn/Mg plane
    ax1.set_xlabel('[Al/Fe]')
    ax1.set_ylabel('[Mn/Mg]')
    ax1.xaxis.set_major_locator(MultipleLocator(0.5))
    ax1.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax1.yaxis.set_major_locator(MultipleLocator(0.5))
    ax1.yaxis.set_minor_locator(MultipleLocator(0.1))
    xlim = (-0.9, 0.6)
    ylim = (-1.1, 0.6)
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    pc = ax1.hexbin(
        disk['al_fe'], disk['mn_mg'],
        C=np.ones(disk.shape[0]),
        extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
        **hexbin_kwargs
    )
    fig.colorbar(pc, ax=ax1, label=r'$\log N$ (disk)', pad=0., use_gridspec=True)
    # Plot accreted, in situ stars
    ax1.scatter(
        insitu['al_fe'], insitu['mn_mg'],
        c=insitu_color,
        **scatter_kwargs
    )
    ax1.scatter(
        accreted['al_fe'], accreted['mn_mg'],
        c=accreted_color,
        **scatter_kwargs
    )
    # Indicate boundary
    alfe_arr = np.arange(-0.9, 0.5, 0.1)
    ax1.plot(alfe_arr, halo_chem_cut(alfe_arr), '-', color='k')
    ax1.text(
        0.15, 0.4, 
        insitu_label, 
        color='k',
        fontsize=plt.rcParams['axes.titlesize'], 
        # style='italic'
    )
    ax1.text(
        -0.8, -1, 
        'Accreted', 
        fontsize=plt.rcParams['axes.titlesize'], 
        color='k',
        bbox={'color': 'w', 'pad': 1, 'alpha': 0.8}
    )

    # Set up second row
    gs1 = GridSpec(1, 2, figure=fig, width_ratios=[4, 1], bottom=0.08, top=0.56, right=0.88, wspace=0.)
    ax2 = fig.add_subplot(gs1[0])
    ax3 = fig.add_subplot(gs1[1], sharey=ax2)

    # [Ce/Mg] - [Mg/H] plane
    ax2.set_xlabel('[Mg/H]')
    ax2.set_ylabel('[Ce/Mg]')
    ax2.xaxis.set_major_locator(MultipleLocator(0.5))
    ax2.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax2.yaxis.set_major_locator(MultipleLocator(0.5))
    ax2.yaxis.set_minor_locator(MultipleLocator(0.1))
    xlim = (-1.8, 0.499)
    ylim = (-1., 1.)
    ax2.set_xlim(xlim)
    ax2.set_ylim(ylim)
    scatter_kwargs['s'] = 5
    median_linewidth = 2
    contour_linewidth = 0.5
    border_linewidth = 3
    # Plot chemically-selected accreted stars
    ax2.scatter(
        accreted['mg_h'], accreted['ce_mg'],
        c=accreted_color, marker='D', zorder=1,
        label='Accreted Halo',
        **scatter_kwargs
    )
    # Plot in-situ halo stars
    ax2.scatter(
        insitu['mg_h'], insitu['ce_mg'], 
        c=insitu_color, marker='o', zorder=2,
        label=insitu_label + ' Halo',
        **scatter_kwargs
    )
    # Plot rolling median of high-Ia stars
    sorted_high_ia = high_ia.sort_values('mg_h')[['mg_h', 'ce_mg']]
    rolling_high_ia = sorted_high_ia.rolling(
        2000, min_periods=1000, step=1000, on='mg_h', center=True
    )
    ax2.plot(
        rolling_high_ia['mg_h'].median(), rolling_high_ia['ce_mg'].median(), 
        'w-', linewidth=border_linewidth, zorder=3,
    )
    ax2.plot(
        rolling_high_ia['mg_h'].median(), rolling_high_ia['ce_mg'].median(), 
        '-', color=high_ia_color, linewidth=median_linewidth, zorder=4, 
        label='High-Ia Disk'
    )
    # Rolling median of low-Ia stars
    sorted_low_ia = low_ia.sort_values('mg_h')[['mg_h', 'ce_mg']]
    rolling_low_ia = sorted_low_ia.rolling(
        1000, min_periods=1000, step=300, on='mg_h', center=True
    )
    ax2.plot(
        rolling_low_ia['mg_h'].median(), rolling_low_ia['ce_mg'].median(), 
        'w-', linewidth=border_linewidth, zorder=3,
    )
    ax2.plot(
        rolling_low_ia['mg_h'].median(), rolling_low_ia['ce_mg'].median(), 
        '-', linewidth=median_linewidth, color=low_ia_color, zorder=4, 
        label='Low-Ia Disk'
    )
    # Plot contours for low- and high-Ia stars
    plot_kde2D_contours(
        ax2, high_ia, 'mg_h', 'ce_mg', c=high_ia_color, lw=contour_linewidth,
        path=paths.data / 'MWM' / 'kde' / 'mgh_cemg' / 'all_high_ia.dat'
    )
    plot_kde2D_contours(
        ax2, low_ia, 'mg_h', 'ce_mg', c=low_ia_color, lw=contour_linewidth,
        path=paths.data / 'MWM' / 'kde' / 'mgh_cemg' / 'all_low_ia.dat'
    )
    # Rolling median of in-situ stars
    sorted_insitu = insitu.sort_values('mg_h')[['mg_h', 'ce_mg']]
    rolling_insitu = sorted_insitu.rolling(
        200, min_periods=100, step=30, on='mg_h', center=True
    )
    ax2.plot(
        rolling_insitu['mg_h'].median(), 
        rolling_insitu['ce_mg'].median(), 
        'w-', linewidth=border_linewidth, zorder=3,
    )
    ax2.plot(
        rolling_insitu['mg_h'].median(), 
        rolling_insitu['ce_mg'].median(), 
        '-', color=insitu_color, linewidth=median_linewidth, zorder=4
    )
    # Rolling median of accreted stars
    sorted_accreted = accreted.sort_values('mg_h')[['mg_h', 'ce_mg']]
    rolling_accreted = sorted_accreted.rolling(
        200, min_periods=100, step=30, on='mg_h', center=True
    )
    ax2.plot(
        rolling_accreted['mg_h'].median(), 
        rolling_accreted['ce_mg'].median(), 
        'w-', linewidth=border_linewidth, zorder=3
    )
    ax2.plot(
        rolling_accreted['mg_h'].median(), 
        rolling_accreted['ce_mg'].median(), 
        '-', color=accreted_color, linewidth=median_linewidth, zorder=4,
    )
    # Compare Hasselquist et al. (2021) dwarf median trends
    dr17_dwarfs = get_hasselquist_dwarfs()
    textcoords = [
        (-1.6, -0.4),
        (-1.72, -0.28),
        (-1.48, -0.48)
    ]
    ls_list = ['-', '--', '-.']
    for i, sys in enumerate(['LMC', 'SMC', 'Sgr']):
        df = dr17_dwarfs[dr17_dwarfs['Sys'] == sys]
        sorted_df = df.sort_values('MG_H')
        rolling_df = sorted_df.rolling(
            30, min_periods=30, step=30, on='MG_H', center=True
        )
        ax2.plot(
            rolling_df['MG_H'].median(), 
            rolling_df['CE_MG'].median(),
            'k', ls=ls_list[i]
        )
        ax2.text(
            textcoords[i][0],
            textcoords[i][1],
            sys,
            bbox={'color': 'w', 'pad': 0.5, 'alpha': 1}
        )
    # Indicate grid edges
    mgh_arr = np.arange(-2.5, 1.25, 0.25)
    ax2.plot(mgh_arr, -2.1 - mgh_arr, 'k:') # edge of stars flagged bad
    ax2.plot(mgh_arr, -1.6 - mgh_arr, color='gray', ls=':') # indicates region of upper limits (manual)
    ax2.plot(mgh_arr, 0.9 - mgh_arr, 'k:') # edge of stars flagged bad
    # Indicate median abundance error
    ax2.errorbar(
        0.3, -0.8, 
        xerr=data['e_mg_h'].median(), 
        yerr=data['e_ce_mg'].median(),
        c='k', capsize=0
    )
    colored_text_legend(
        ax2, 
        loc='upper right', 
        ncols=2,
        columnspacing=1,
        fontsize=plt.rcParams['axes.titlesize'],
        frameon=True,
        framealpha=0.8,
    )


    # Marginal panel with histograms
    cemg_bins = np.arange(-1.1, 1.12, 0.05)
    colors = [high_ia_color, low_ia_color, insitu_color, accreted_color]
    labels = ['High-Ia Disk', 'Low-Ia Disk', insitu_label + ' Halo', 'Accreted Halo']
    for i, df in enumerate([high_ia, low_ia, insitu, accreted]):
        hist, bin_edges = np.histogram(df['ce_mg'], cemg_bins, density=True)
        if i < 2:
            lw = 1
        else:
            lw = 2
        ax3.plot(
            hist/hist.max(), get_bin_centers(bin_edges),
            c=colors[i], lw=lw, label=labels[i]
        )
    ax3.set_xlabel('Density')
    ax3.set_xlim((0, 1.2))
    ax3.tick_params(axis='y', labelleft=False, labelright=True)
    # colored_text_legend(ax3, loc='upper left')

    plt.subplots_adjust(bottom=0.08, top=0.96, left=0.08, right=0.92)
    plt.savefig(paths.figures / 'halo')
    plt.close()


def logsum(x):
    """
    Log of the sum of an array.
    """
    return np.log10(np.sum(x))


def halo_ELz_cut(Lz):
    """
    Arbitrary halo cut - find a better one in the literature
    """
    # # return -0.55 - np.exp(Lz / 2.5)
    return np.where(Lz<-3, 0, -0.7 - np.exp(Lz / 1.5))
    # return -0.6 - np.exp(Lz / 1.2)
    # return np.where(Lz<0, -0.7 - np.exp(Lz / 1.5), -3)


def halo_chem_cut(alfe):
    """
    Chemical cut in [Mn/Mg]-[Al/Fe] plane to select accreted stars.
    """
    return np.where(alfe > -0.2, -0.6-2*alfe, -0.2)


def get_hasselquist_dwarfs():
    """
    Select APOGEE DR17 targets in dwarf galaxies using the Hasselquist et al. 
    (2021) selection table.
    """
    select_table = ascii.read(
        paths.data / 'catalogs' / 'hasselquist2021_table2_mrt.txt'
    ).to_pandas().set_index('ID')
    dr17_full = fits_to_pandas(
        paths.data / 'catalogs' / 'allStarLite-dr17-synspec_rev1.fits', hdu=1
    )
    # Drop duplicate observations
    dr17_full = dr17_full[dr17_full['EXTRATARG'] != 16].set_index('APOGEE_ID').copy()
    # Make catalog of dwarf members
    dr17_dwarfs = dr17_full.join(select_table, how='right')
    # Drop flagged abundances, require S/N > 70
    dr17_dwarfs = dr17_dwarfs[
        (dr17_dwarfs['CE_FE_FLAG'] == 0) &
        (dr17_dwarfs['MG_FE_FLAG'] == 0) &
        (dr17_dwarfs['SNR'] > 70)
    ].copy()
    dr17_dwarfs['MG_H'] = dr17_dwarfs['MG_FE'] + dr17_dwarfs['FE_H']
    dr17_dwarfs['CE_MG'] = dr17_dwarfs['CE_FE'] - dr17_dwarfs['MG_FE']
    # dr17_dwarfs['CE_MG_ERR'] = np.sqrt(
    #     dr17_dwarfs['CE_FE_ERR']**2 + dr17_dwarfs['MG_FE_ERR']**2
    # )
    return dr17_dwarfs[['Sys', 'MG_H', 'CE_MG']]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Plot [Ce/Mg] vs [Mg/H] for halo stars.'
    )
    parser.add_argument('--style',
        choices=('paper', 'poster'),
        default='paper',
        help='Plot style to use (default: paper).'
    )
    args = parser.parse_args()
    main(**vars(args))
