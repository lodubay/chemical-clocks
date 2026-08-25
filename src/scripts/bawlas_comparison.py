"""
This script compares the DR19 ASPCAP [Ce/H] abundances against those for
DR17 from BAWLAS (Hayes et al. 2022).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

import paths
from utils import import_sample, fits_to_pandas, binned_quantiles, sample_rows
from plotting import ONE_COLUMN_WIDTH

def main(style='paper'):
    sample = import_sample(good_ages=False, cut_limits=True)
    bawlas = fits_to_pandas(paths.data / 'catalogs' / 'dr17_nc_abund_v2_0.fits')
    # Merge DR19 and BAWLAS
    bawlas_dr19 = sample.join(bawlas.set_index(['APOGEE_ID', 'FIELD', 'TELESCOPE']), on=['sdss4_apogee_id', 'field', 'telescope'])
    bawlas_dr19['CE_H'] = bawlas_dr19['CE_FE'] + bawlas_dr19['FE_H']
    bawlas_dr19['CE_H_LIM'] = bawlas_dr19['CE_FE_LIM'] + bawlas_dr19['FE_H']
    bawlas_dr19.dropna(subset='CE_FE', inplace=True)
    # Limit number of points in scatter plot
    plot_sample = sample_rows(bawlas_dr19, n=10000)

    plt.style.use(paths.styles / f'{style}.mplstyle')
    fig, axs = plt.subplots(
        2, 1, 
        figsize=(ONE_COLUMN_WIDTH, 1.5*ONE_COLUMN_WIDTH), 
        sharex=True, sharey=True,
        gridspec_kw={'hspace': 0, 'wspace': 0}
    )

    kwargs = dict(
        rasterized=True, s=2, edgecolors='none', marker='.',
    )
    titles = [r'With $\log(g)$ corrections', r'Without $\log(g)$ corrections']
    xlim = (-1.5, 0.8)
    bin_edges = np.arange(xlim[0], xlim[1], 0.05)
    for i, sdss5_col in enumerate(['ce_h_corr', 'ce_h']):
        bawlas_dr19['ce_h_diff'] = bawlas_dr19[sdss5_col] - bawlas_dr19['CE_H']
        plot_sample['ce_h_diff'] = plot_sample[sdss5_col] - plot_sample['CE_H']
        axs[i].scatter(plot_sample['CE_H'], plot_sample['ce_h_diff'], c='gray', **kwargs)
        for q, ls in zip([0.16, 0.5, 0.84], ['--', '-', '--']):
            binned_quants = binned_quantiles(
                bawlas_dr19, 'ce_h_diff', 'CE_H', 
                q=q, bin_edges=bin_edges, min_count=10
            )
            axs[i].plot(*binned_quants, color='k', ls=ls)
        axs[i].set_title(titles[i], y=0.85)
        axs[i].plot(xlim, [0, 0], c='k', ls=':')
        axs[i].set_ylabel('DR19 - BAWLAS')
        axs[i].text(
            -1.4, -0.5,
            'Median: %.03f\nStd. Dev.: %.02f' % (
                bawlas_dr19['ce_h_diff'].median(), 
                bawlas_dr19['ce_h_diff'].std()
            )
        )
    axs[-1].set_xlabel('BAWLAS [Ce/H]')
    axs[0].set_xlim(xlim)
    axs[0].set_ylim((-0.6, 0.6))
    axs[0].xaxis.set_major_locator(MultipleLocator(0.5))
    axs[0].xaxis.set_minor_locator(MultipleLocator(0.1))
    axs[0].yaxis.set_major_locator(MultipleLocator(0.2))
    axs[0].yaxis.set_minor_locator(MultipleLocator(0.05))
    plt.savefig(paths.figures / 'bawlas_comparison')
    plt.close()


if __name__ == '__main__':
    main()
