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

import paths
from plotting import TWO_COLUMN_WIDTH, colored_text_legend
from colormaps import paultol
from utils import binned_quantiles

DENSITY_COLORMAP = 'binary_r'


def main(style='paper'):
    # Get data
    data = pd.read_csv(paths.data / 'MWM' / 'sample.csv')
    # Kinematically-selected halo
    halo = data[data['E']/1e5 > halo_ELz_cut(data['Lz']/1e3)]
    # halo = data[(data['z_max'] > 3) | (data['vphi'] > -120)]
    # Kinematically-selected disk stars
    disk = data[data['E']/1e5 < halo_ELz_cut(data['Lz']/1e3)].copy()
    # disk = data[(data['z_max'] < 3) & (data['vphi'] < -120)]
    low_ia = disk[disk['high_alpha']]
    # Chemically-selected accreted stars
    accreted = halo[halo['mn_mg'] < halo_chem_cut(halo['al_fe'])]
    # Chemically-selected in-situ halo stars
    insitu = halo[halo['mn_mg'] > halo_chem_cut(halo['al_fe'])]
    
    # Set up figure
    plt.style.use(paths.styles / f'{style}.mplstyle')
    fig = plt.figure(figsize=(0.8*TWO_COLUMN_WIDTH, 0.8*TWO_COLUMN_WIDTH))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[2, 3], wspace=0.35)
    ax0 = fig.add_subplot(gs[0,0])
    ax1 = fig.add_subplot(gs[0,1])
    ax2 = fig.add_subplot(gs[1,:])
    hexbin_kwargs = dict(
        cmap=DENSITY_COLORMAP, 
        linewidths=0.2,
        reduce_C_function=logsum,
        mincnt=1
    )
    scatter_kwargs = dict(
        s=2, rasterized=True, edgecolors='none',
    )
    halo_color = 'k'
    disk_color = 'k'
    accreted_color = paultol.bright.colors[5]
    insitu_color = paultol.bright.colors[2]
    low_ia_color = paultol.bright.colors[1]
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
        gridsize=50,
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
        fontsize=plt.rcParams['axes.titlesize'], 
        color=halo_color
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
    xlim = (-0.8, 0.7)
    ylim = (-1.1, 0.6)
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    pc = ax1.hexbin(
        disk['al_fe'], disk['mn_mg'],
        C=np.ones(disk.shape[0]),
        gridsize=50,
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
    alfe_arr = np.arange(-0.8, 0.5, 0.1)
    ax1.plot(alfe_arr, halo_chem_cut(alfe_arr), '-', color='k')
    ax1.text(
        0.25, 0.3, insitu_label, 
        color=insitu_color,
        fontsize=plt.rcParams['axes.titlesize'], 
        # style='italic'
    )
    ax1.text(
        -0.7, -0.9, 'Accreted', 
        fontsize=plt.rcParams['axes.titlesize'], 
        color=accreted_color,
        bbox={'color': 'w', 'pad': 1}
    )

    # [Ce/Mg] - [Mg/H] plane
    ax2.set_xlabel('[Mg/H]')
    ax2.set_ylabel('[Ce/Mg]')
    ax2.xaxis.set_major_locator(MultipleLocator(0.5))
    ax2.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax2.yaxis.set_major_locator(MultipleLocator(0.5))
    ax2.yaxis.set_minor_locator(MultipleLocator(0.1))
    xlim = (-1.8, 0.6)
    ylim = (-1.1, 1.1)
    ax2.set_xlim(xlim)
    ax2.set_ylim(ylim)
    scatter_kwargs['s'] = 8
    pc = ax2.hexbin(
        disk['mg_h'], disk['ce_mg'],
        C=np.ones(disk.shape[0]),
        gridsize=(50, 18),
        zorder=1,
        extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
        **hexbin_kwargs
    )
    fig.colorbar(
        pc, ax=ax2, label=r'$\log N$ (disk)', 
        pad=0., fraction=0.07, aspect=30, 
        use_gridspec=True
    )
    # Plot chemically-selected accreted stars
    ax2.scatter(
        accreted['mg_h'], accreted['ce_mg'],
        c=accreted_color, marker='D', zorder=2,
        label='Accreted',
        **scatter_kwargs
    )
    # Plot in-situ halo stars
    ax2.scatter(
        insitu['mg_h'], insitu['ce_mg'], 
        c=insitu_color, marker='o', zorder=1,
        label=insitu_label,
        **scatter_kwargs
    )
    # Rolling median, 16th and 84th percentiles of low-Ia stars
    # mgh_bin_edges = np.arange(-1.55, 0.56, 0.1)
    # for q, ls in zip([0.16, 0.5, 0.84], ['--', '-', '--']):
    #     low_ia_trend = binned_quantiles(
    #         low_ia, 'ce_mg', 'mg_h',
    #         q=q, bin_edges=mgh_bin_edges, min_count=10
    #     )
    #     label = 'Low-Ia' if q == 0.5 else None
    #     ax2.plot(*low_ia_trend, 'w-', linewidth=2)
    #     ax2.plot(*low_ia_trend, ls, color=low_ia_color, label=label)
    #     insitu_trend = binned_quantiles(
    #         insitu, 'ce_mg', 'mg_h',
    #         q=q, bin_edges=mgh_bin_edges, min_count=10
    #     )
    #     ax2.plot(*insitu_trend, 'w-', linewidth=2)
    #     ax2.plot(*insitu_trend, ls, color=insitu_color)
    #     accreted_trend = binned_quantiles(
    #         accreted, 'ce_mg', 'mg_h',
    #         q=q, bin_edges=mgh_bin_edges, min_count=10
    #     )
    #     ax2.plot(*accreted_trend, 'w-', linewidth=2)
    #     ax2.plot(*accreted_trend, ls, color=accreted_color)
    sorted_low_ia = low_ia.sort_values('mg_h')[['mg_h', 'ce_mg']]
    rolling_low_ia = sorted_low_ia.rolling(
        1000, min_periods=1000, step=100, on='mg_h', center=True
    )
    ax2.plot(
        rolling_low_ia['mg_h'].median(), rolling_low_ia['ce_mg'].median(), 
        'w-', linewidth=2
    )
    ax2.plot(
        rolling_low_ia['mg_h'].median(), rolling_low_ia['ce_mg'].median(), 
        '--', color=low_ia_color, label='Low-Ia'
    )
    # Indicate dispersion
    # for q in [0.16, 0.84]:
    #     ax2.plot(
    #         rolling_low_ia['mg_h'].quantile(q), 
    #         rolling_low_ia['ce_mg'].quantile(q), 
    #         'w-', linewidth=2
    #     )
    #     ax2.plot(
    #         rolling_low_ia['mg_h'].quantile(q), 
    #         rolling_low_ia['ce_mg'].quantile(q), 
    #         ':', color=low_ia_color
    #     )
    # Rolling median of in-situ stars
    sorted_insitu = insitu.sort_values('mg_h')[['mg_h', 'ce_mg']]
    rolling_insitu = sorted_insitu.rolling(
        100, min_periods=100, step=10, on='mg_h', center=True
    )
    ax2.plot(
        rolling_insitu['mg_h'].median(), 
        rolling_insitu['ce_mg'].median(), 
        'w-', linewidth=2
    )
    ax2.plot(
        rolling_insitu['mg_h'].median(), 
        rolling_insitu['ce_mg'].median(), 
        '-', color=insitu_color
    )
    # Indicate dispersion
    # for q in [0.16, 0.84]:
    #     ax2.plot(
    #         rolling_insitu['mg_h'].quantile(q), 
    #         rolling_insitu['ce_mg'].quantile(q), 
    #         'w-', linewidth=2
    #     )
    #     ax2.plot(
    #         rolling_insitu['mg_h'].quantile(q), 
    #         rolling_insitu['ce_mg'].quantile(q), 
    #         ':', color=insitu_color
    #     )
    # Rolling median of accreted stars
    sorted_accreted = accreted.sort_values('mg_h')[['mg_h', 'ce_mg']]
    rolling_accreted = sorted_accreted.rolling(
        100, min_periods=100, step=10, on='mg_h', center=True
    )
    ax2.plot(
        rolling_accreted['mg_h'].median(), 
        rolling_accreted['ce_mg'].median(), 
        'w-', linewidth=2
    )
    ax2.plot(
        rolling_accreted['mg_h'].median(), 
        rolling_accreted['ce_mg'].median(), 
        '-', color=accreted_color
    )
    # Indicate dispersion
    # for q in [0.16, 0.84]:
    #     ax2.plot(
    #         rolling_accreted['mg_h'].quantile(q), 
    #         rolling_accreted['ce_mg'].quantile(q), 
    #         'w-', linewidth=2
    #     )
    #     ax2.plot(
    #         rolling_accreted['mg_h'].quantile(q), 
    #         rolling_accreted['ce_mg'].quantile(q), 
    #         ':', color=accreted_color
    #     )
    # Indicate grid edges
    mgh_arr = np.arange(-2.5, 1.25, 0.25)
    ax2.plot(mgh_arr, -2.1 - mgh_arr, 'k:') # edge of stars flagged bad
    ax2.plot(mgh_arr, -1.6 - mgh_arr, color='gray', ls=':') # indicates region of upper limits (manual)
    ax2.plot(mgh_arr, 0.9 - mgh_arr, 'k:') # edge of stars flagged bad
    # Indicate median abundance error
    ax2.errorbar(
        -1.7, -0.9, 
        xerr=data['e_mg_h'].median(), 
        yerr=data['e_ce_mg'].median(),
        c='k', capsize=0
    )
    # ax2.legend(loc='upper left')
    colored_text_legend(ax2, show_handles=True, loc='upper left', fontsize=plt.rcParams['axes.titlesize'], frameon=True)

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
