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
from utils import import_sample, get_bin_centers, fits_to_pandas, box_smooth
from contours import plot_kde2D_contours

DENSITY_COLORMAP = 'binary_r'
HALO_LZ_CUT = 0.3 # cut in abs(Lz/Jtot) for halo stars
DISK_LZ_CUT = 0.6 # cut in abs(Lz/Jtot) for disk stars


def main(style='paper', verbose=False):
    # Get data
    fullsample = import_sample(good_ages=False)
    # Drop flagged abundances
    data = fullsample[
        (fullsample['al_h_flags'] == 0) &
        (fullsample['mn_h_flags'] == 0)
    ]
    dropped_stars = fullsample.shape[0] - data.shape[0]
    if verbose: print(f'Dropped {dropped_stars} flagged abundances.')

    # Halo and disk orbit selections via the "Action Diamond"
    ad_halo_mask = np.abs(data['Lz']/data['Jtot']) <= HALO_LZ_CUT
    ad_disk_mask = data['Lz']/data['Jtot'] >= DISK_LZ_CUT
    # Crude spatial bulge selection
    bulge_mask = data['galr'] <= 3
    # Feuillet et al. (2020) GSE selection
    feuillet_gse_mask = (
        (np.sqrt(data['Jr']) >= 30) &
        (np.sqrt(data['Jr']) <= 50) &
        (data['Lz'] >= -500) &
        (data['Lz'] <= 500)
    )

    # Chemical accreted & in-situ population selection
    buffer = 0.05 # dex, buffer between chemically-selected populations
    chem_accreted_mask = data['mn_mg'] <= halo_chem_cut(data['al_fe'])-buffer
    chem_insitu_mask = (
        (data['mn_mg'] >= halo_chem_cut(data['al_fe'])+buffer) & 
        (data['mn_mg'] <= -0.2)
    )

    # Apply orbit selections
    disk = data[ad_disk_mask]
    low_ia = disk[disk['high_alpha']]
    high_ia = disk[disk['low_alpha']]
    intermediate = data[(~ad_disk_mask) & (~ad_halo_mask)]
    bulge = data[ad_halo_mask & bulge_mask]
    # halo = data[ad_halo_mask & (~bulge_mask)]

    # Apply chemical selections
    accreted = data[ad_halo_mask & (~bulge_mask) & chem_accreted_mask & (~feuillet_gse_mask)]
    insitu = data[ad_halo_mask & (~bulge_mask) & chem_insitu_mask]
    other_chem = data[ad_halo_mask & (~bulge_mask) & (~chem_insitu_mask) & (~chem_accreted_mask)]
    gse = data[feuillet_gse_mask & chem_accreted_mask]

    # Print sub-sample sizes
    if verbose:
        print('Halo sub-sample sizes:')
        labels = ['gse', 'accreted halo', 'in situ halo', 'disk']
        subsamples = [gse, accreted, insitu, disk]
        for label, subsample in zip(labels, subsamples):
            print('\t%s: %s' % (label, subsample.shape[0]))

        print('Median [Ce/Mg] per sub-sample:')
        labels = ['gse', 'accreted halo', 'in situ halo']
        subsamples = [gse, accreted, insitu]
        for label, subsample in zip(labels, subsamples):
            print('\t%s: %s' % (label, subsample['ce_mg'].median()))
    
    # Set up figure
    plt.style.use(paths.styles / f'{style}.mplstyle')
    savedir = {
        'paper': paths.figures,
        'presentation': paths.extra/'presentation'
    }[style]
    savedir.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(0.8*TWO_COLUMN_WIDTH, 0.8*TWO_COLUMN_WIDTH))
    gs0 = GridSpec(1, 2, figure=fig, top=0.98, bottom=0.64, wspace=0.35)
    ax0 = fig.add_subplot(gs0[0]) # E-Lz plane
    ax1 = fig.add_subplot(gs0[1]) # [Mn/Mg] - [Al/Fe] plane
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
    accreted_color = paultol.vibrant.colors[1]
    insitu_color = paultol.vibrant.colors[2]
    gse_color = paultol.vibrant.colors[4]
    low_ia_color = paultol.bright.colors[1]
    high_ia_color = paultol.bright.colors[0]
    other_color = paultol.vibrant.colors[6]
    accreted_marker = 'D'
    insitu_marker = 'o'
    gse_marker = 's'
    other_marker = 'o'
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
    xlim = (-4.5, 6.5)
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
    # Scatter plot of sub-samples
    subsamples = [intermediate, bulge, insitu, accreted, gse]
    colors = [other_color, other_color, insitu_color, accreted_color, gse_color]
    markers = [other_marker, other_marker, insitu_marker, accreted_marker, gse_marker]
    for i, df in enumerate(subsamples):
        ax0.scatter(
            df['Lz']/1e3, df['E']/1e5, 
            c=colors[i], marker=markers[i], **scatter_kwargs
        )
    ax0.text(-2., -0.5, 'Halo')
    ax0.text(2.5, -1.5, 'Disk')
    ax0.text(-2.5, -2.2, 'Bulge')

    # Chemical cut in Al/Fe - Mn/Mg plane
    ax1.set_xlabel('[Al/Fe]')
    ax1.set_ylabel('[Mn/Mg]')
    ax1.xaxis.set_major_locator(MultipleLocator(0.5))
    ax1.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax1.yaxis.set_major_locator(MultipleLocator(0.5))
    ax1.yaxis.set_minor_locator(MultipleLocator(0.1))
    xlim = (-0.7, 0.7)
    ylim = (-1.2, 0.7)
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    pc = ax1.hexbin(
        disk['al_fe'], disk['mn_mg'],
        C=np.ones(disk.shape[0]),
        extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
        **hexbin_kwargs
    )
    fig.colorbar(pc, ax=ax1, label=r'$\log N$ (disk)', pad=0., use_gridspec=True)
    # Scatter plot of sup-samples
    subsamples = [other_chem, insitu, accreted, gse]
    colors = [other_color, insitu_color, accreted_color, gse_color]
    markers = [other_marker, insitu_marker, accreted_marker, gse_marker]
    for i, df in enumerate(subsamples):
        ax1.scatter(
            df['al_fe'], df['mn_mg'], 
            c=colors[i], marker=markers[i], **scatter_kwargs
        )
    # Indicate boundary
    alfe_arr = np.arange(-0.9, 0.71, 0.01)
    ax1.plot(alfe_arr, halo_chem_cut(alfe_arr), '-', color='k')
    ax1.plot(alfe_arr, insitu_chem_cut(alfe_arr), '-', color='k')
    ax1.text(
        0.6, -1.1, 
        'Low-Ia/\n%s' % insitu_label, 
        color='k',
        ha='right',
    )
    ax1.text(
        -0.6, -1.1, 
        'Accreted', 
        color='k',
    )
    ax1.text(
        0.6, 0.55,
        'High-Ia\nDisk',
        color='k',
        ha='right',
        va='top',
    )

    # Set up second row
    gs1 = GridSpec(1, 2, figure=fig, width_ratios=[4, 1], bottom=0.08, top=0.53, right=0.88, wspace=0.)
    ax2 = fig.add_subplot(gs1[0])
    ax3 = fig.add_subplot(gs1[1], sharey=ax2)

    # [Ce/Mg] - [Mg/H] plane
    ax2.set_xlabel('[Mg/H]')
    ax2.set_ylabel('[Ce/Mg]')
    ax2.xaxis.set_major_locator(MultipleLocator(0.5))
    ax2.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax2.yaxis.set_major_locator(MultipleLocator(0.5))
    ax2.yaxis.set_minor_locator(MultipleLocator(0.1))
    xlim = (-1.9, 0.499)
    ylim = (-1., 0.8)
    ax2.set_xlim(xlim)
    ax2.set_ylim(ylim)
    scatter_kwargs['s'] = 5
    median_linewidth = 2
    contour_linewidth = 0.5
    border_linewidth = 3
    # Scatter plot and rolling medians of halo sub-populations
    subsamples = [gse, accreted, insitu]
    colors = [gse_color, accreted_color, insitu_color]
    markers = [gse_marker, accreted_marker, insitu_marker]
    labels = ['GSE', 'Accreted Halo', insitu_label + ' Halo']
    for i, df in enumerate(subsamples):
        ax2.scatter(
            df['mg_h'], 
            df['ce_mg'], 
            c=colors[i], 
            marker=markers[i], 
            zorder=2,
            label=labels[i],
            **scatter_kwargs
        )
        # Rolling median
        sorted_mgh = df.sort_values('mg_h')[['mg_h', 'ce_mg']]
        rolling_mgh = sorted_mgh.rolling(
            100, min_periods=30, step=30, on='mg_h', center=True
        )
        ax2.plot(
            rolling_mgh['mg_h'].median(), 
            rolling_mgh['ce_mg'].median(), 
            'w-', 
            linewidth=border_linewidth, 
            zorder=3
        )
        ax2.plot(
            rolling_mgh['mg_h'].median(), 
            rolling_mgh['ce_mg'].median(), 
            '-', 
            color=colors[i], 
            linewidth=median_linewidth, 
            zorder=5,
        )
    # Rolling median and contours for low- and high-Ia disk stars
    subsamples = [low_ia, high_ia]
    colors = [low_ia_color, high_ia_color]
    labels = ['Low-Ia Disk', 'High-Ia Disk']
    fnames = ['all_low_ia.dat', 'all_high_ia.dat']
    for i, df in enumerate(subsamples):
        sorted_mgh = df.sort_values('mg_h')[['mg_h', 'ce_mg']]
        rolling_mgh = sorted_mgh.rolling(
            1000, min_periods=1000, step=1000, on='mg_h', center=True
        )
        ax2.plot(
            rolling_mgh['mg_h'].median(), 
            rolling_mgh['ce_mg'].median(), 
            'w-', 
            linewidth=border_linewidth, 
            zorder=3,
        )
        ax2.plot(
            rolling_mgh['mg_h'].median(), 
            rolling_mgh['ce_mg'].median(), 
            '-', 
            linewidth=median_linewidth, 
            color=colors[i], 
            zorder=4, 
            label=labels[i]
        )
        plot_kde2D_contours(
            ax2, df, 'mg_h', 'ce_mg', c=colors[i], lw=contour_linewidth,
            path=paths.data / 'MWM' / 'kde' / 'mgh_cemg' / fnames[i]
        )
    # Compare Hasselquist et al. (2021) dwarf median trends
    dr17_dwarfs = get_hasselquist_dwarfs()
    textcoords = [
        (-1.75, -0.18),
        (-1.8, 0.0),
        (-1.5, -0.2)
    ]
    ls_list = ['-', '--', '-.']
    for i, sys in enumerate(['LMC', 'SMC', 'Sgr']):
        df = dr17_dwarfs[dr17_dwarfs['Sys'] == sys]
        sorted_df = df.sort_values('mg_h')
        rolling_df = sorted_df.rolling(
            100, min_periods=30, step=30, on='mg_h', center=True
        )
        ax2.plot(
            rolling_df['mg_h'].median(), 
            rolling_df['ce_mg'].median(),
            'k', ls=ls_list[i]
        )
        ax2.text(
            textcoords[i][0],
            textcoords[i][1],
            sys,
            # bbox={'color': 'w', 'pad': 0.5, 'alpha': 1}
        )
    # Indicate grid edges
    mgh_arr = np.arange(-2.5, 1.25, 0.25)
    ax2.plot(mgh_arr, -2.1 - mgh_arr, 'k:') # edge of stars flagged bad
    ax2.plot(mgh_arr, -1.5 - mgh_arr, color='gray', ls=':') # indicates region of upper limits (approximate)
    ax2.plot(mgh_arr, 0.9 - mgh_arr, 'k:') # edge of stars flagged bad
    ax2.text(-1.22, -0.95, 'Grid edge', ha='right')
    ax2.text(-0.62, -0.95, 'Upper limits', ha='right')
    # Indicate median abundance error
    ax2.errorbar(
        0.3, -0.8, 
        xerr=data['e_mg_h'].median(), 
        yerr=data['e_ce_mg'].median(),
        c='k', capsize=0
    )
    colored_text_legend(
        ax2, 
        loc='lower left', 
        ncols=5,
        columnspacing=1,
        fontsize=plt.rcParams['axes.titlesize'],
        # frameon=True,
        # framealpha=0.8,
        bbox_to_anchor=(0.03, 0.98)
    )

    # Marginal panel with histograms
    cemg_bins = np.arange(-1.1, 1.12, 0.05)
    colors = [high_ia_color, low_ia_color, insitu_color, accreted_color, gse_color]
    # labels = ['High-Ia Disk', 'Low-Ia Disk', insitu_label + ' Halo', 'Accreted Halo']
    for i, df in enumerate([high_ia, low_ia, insitu, accreted, gse]):
        hist, bin_edges = np.histogram(df['ce_mg'], cemg_bins, density=True)
        hist_smooth = box_smooth(hist, bin_edges, 0.2)
        if i < 2:
            lw = 1
        else:
            lw = 2
        ax3.plot(
            hist_smooth/hist_smooth.max(), get_bin_centers(bin_edges),
            c=colors[i], lw=lw, #label=labels[i]
        )
    ax3.set_xlabel('Density')
    ax3.set_xlim((0, 1.2))
    ax3.tick_params(axis='y', labelleft=False, labelright=True)
    # colored_text_legend(ax3, loc='upper left')

    plt.subplots_adjust(bottom=0.08, top=0.96, left=0.08, right=0.92)
    plt.savefig(savedir / 'halo')
    plt.close()


def logsum(x):
    """
    Log of the sum of an array.
    """
    return np.log10(np.sum(x))


def halo_chem_cut(alfe):
    """
    Chemical cut in [Mn/Mg]-[Al/Fe] plane to select accreted stars from in-situ.
    """
    # return np.where(alfe > -0.2, -0.6-2*alfe, -0.2)
    return -0.6-2*alfe


def insitu_chem_cut(alfe):
    """
    Chemical cut in [Mn/Mg]-[Al/Fe] plane to separate in-situ stars from disk.
    """
    return np.where(alfe>=-0.2, -0.2, np.nan)


def get_hasselquist_dwarfs():
    """
    Select APOGEE DR17 targets in dwarf galaxies using the Hasselquist et al. 
    (2021) selection table.
    """
    select_table = ascii.read(
        paths.data / 'catalogs' / 'hasselquist2021_table2_mrt.txt'
    ).to_pandas().set_index('ID')
    mwm_full = fits_to_pandas(
        paths.data / 'catalogs/astraAllStarASPCAP-0.6.0.fits.gz', 
        hdu=2
    )
    # Limit to DR17 non-duplicate data to match Hasselquist et al. (2021)
    mwm_full = mwm_full[
        (mwm_full['sdss4_apogee_extra_target_flags'] != 16) &
        (mwm_full['release'] == 'dr17')
    ]
    # drop duplicate SDSS-V IDs with the lowest SNR
    mwm_full.sort_values(['sdss4_apogee_id', 'snr'], inplace=True, ascending=True)
    mwm_full.drop_duplicates(subset='sdss4_apogee_id', keep='last', inplace=True)
    mwm_full.set_index('sdss4_apogee_id', inplace=True)
    # Make catalog of dwarf members
    mwm_dwarfs = mwm_full.join(select_table, how='right')
    # Drop flagged abundances, require S/N > 70
    mwm_dwarfs = mwm_dwarfs[
        (mwm_dwarfs['ce_h_flags'] == 0) &
        (mwm_dwarfs['mg_h_flags'] == 0) &
        (mwm_dwarfs['snr'] > 70)
    ].copy()
    mwm_dwarfs['ce_mg'] = mwm_dwarfs['ce_h'] - mwm_dwarfs['mg_h']
    return mwm_dwarfs[['Sys', 'mg_h', 'ce_mg']]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Plot [Ce/Mg] vs [Mg/H] for halo stars.'
    )
    parser.add_argument('-s', '--style',
        choices=('paper', 'presentation'),
        default='paper',
        help='Plot style to use (default: paper).'
    )
    parser.add_argument('-v', '--verbose',
        action='store_true',
        help='Print verbose output to terminal.'
    )
    args = parser.parse_args()
    main(**vars(args))
