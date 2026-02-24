"""
This script compares the age--[Ce/Mg] relations for MWM DR19 against the
OCCAM DR19 open cluster sample and against APOKASC-3.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, LogNorm
from matplotlib.ticker import MultipleLocator
from astropy.io import ascii

from mwm_sample import abundance_ratio
from utils import good_ages, fits_to_pandas
from plotting import truncate_colormap, ONE_COLUMN_WIDTH
import paths

def main(style='paper'):
    # Figure setup
    plt.style.use(paths.styles / f'{style}.mplstyle')
    fig, axs = plt.subplots(
        3, sharex=True, sharey=True,
        figsize=(ONE_COLUMN_WIDTH, 1.8 * ONE_COLUMN_WIDTH),
        gridspec_kw={'hspace': 0.},
        # constrained_layout=True
    )
    xlim = (0, 11)
    ylim = (-0.8, 0.8)

    # Plot full MWM DR19 sample
    mwm_rgb = pd.read_csv(paths.data / 'MWM' / 'sample.csv')
    mwm_ages = good_ages(mwm_rgb)
    pcm0 = axs[0].hexbin(
        mwm_ages['age'], mwm_ages['ce_mg'],
        C=np.ones(mwm_ages.shape[0]),
        reduce_C_function=np.sum,
        cmap='gist_heat_r',
        gridsize=30,
        linewidths=0.2,
        extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
        vmin=0
    )
    fig.colorbar(pcm0, ax=axs[0], shrink=0.9, label='Number of stars')
    # Rolling median
    sorted_ages = mwm_ages.sort_values('age')[['age', 'ce_mg']]
    rolling_medians = sorted_ages.rolling(
        1000, min_periods=1000, step=250, on='age', center=True
    ).median()
    axs[0].plot(
        rolling_medians['age'], rolling_medians['ce_mg'], 
        'w-', 
        linewidth=2
    )
    axs[0].plot(
        rolling_medians['age'], rolling_medians['ce_mg'], 
        'k-', 
        label='Rolling median'
    )
    # Plot median trend
    # age_bin_edges = np.arange(-0.5, 12.6, 1)
    # mwm_medians = binned_quantiles(
    #     mwm_ages, 'ce_mg', 'age',
    #     q=0.5, bin_edges=age_bin_edges, min_count=10
    # )
    # axs[0].plot(*mwm_medians, '-', color='w', linewidth=2)
    # axs[0].plot(
    #     *mwm_medians, '-', 
    #     color='k', 
    #     label='Median trend'
    # )
    axs[0].set_title('(a) StarFlow', y=0.82)

    # APOKASC-3 catalog
    apokasc_csv_path = paths.data / 'MWM' / 'APOKASC3_MWM.csv'
    if apokasc_csv_path.exists():
        apokasc3 = pd.read_csv(apokasc_csv_path)
    else:
        apokasc3 = join_apokasc_mwm()
        apokasc3.to_csv(apokasc_csv_path, index=True)
    scatter_kwargs = dict(s=1, edgecolors='none', marker='o')
    cmap = truncate_colormap('autumn', minval=0.1, maxval=0.7)
    norm = BoundaryNorm([0, 0.5, 1], cmap.N)
    pc = axs[1].scatter(
        apokasc3['AgeBest'], 
        apokasc3['MWM_CE_MG'],
        c=apokasc3['EvolState']=='RC', 
        cmap=cmap,
        norm=norm,
        # label='RC',
        **scatter_kwargs
    )
    cbar = fig.colorbar(pc, ax=axs[1], shrink=0.9)
    cbar.ax.set_yticks([0.25, 0.75], labels=['RGB', 'RC'])
    # DR19 median trend for comparison
    axs[1].plot(
        rolling_medians['age'], rolling_medians['ce_mg'], 
        'w-', 
        linewidth=2
    )
    axs[1].plot(
        rolling_medians['age'], rolling_medians['ce_mg'], 
        'k-', 
        label='Rolling median'
    )
    axs[1].set_title('(b) APOKASC-3', y=0.82)

    # OCCAM DR19 open clusters
    occam19 = pd.read_csv(paths.data / 'MWM' / 'occam_19cluster-rgb.csv')
    occam19['CG_Age'] = (10**(occam19['CG_logAge']))/1e9
    occam19['Ce_Mg'] = occam19['Ce_H'] - occam19['Mg_H']
    occam19['Ce_Mg_ERR'] = np.sqrt(occam19['Ce_H_ERR']**2 + occam19['Mg_H_ERR']**2)
    # Quality cuts
    occam19 = occam19[
        (occam19['OCCAM_Qual']>0) & 
        (occam19['Ce_H'] > -100) & 
        (occam19['Mg_H'] > -100) &
        (occam19['Ce_H_ERR'] < 0.2) & 
        (occam19['Mg_H_ERR'] < 0.2) & 
        (occam19['rgb_N_stars'] > 1) 
    ]
    im = axs[2].scatter(
        occam19['CG_Age'], occam19['Ce_Mg'], 
        s = 15, 
        c = np.log10(occam19['rgb_N_stars']), 
        cmap = 'viridis', 
        # vmin = 0, vmax = 18
        # norm = LogNorm()
        vmin=0
    )
    axs[2].errorbar(
        occam19['CG_Age'], occam19['Ce_Mg'], 
        yerr = occam19['Ce_Mg_ERR'], 
        c = 'tab:gray', fmt = '.', zorder = 0, capsize = 0
    )
    fig.colorbar(im, ax=axs[2], shrink=0.9, label='log number of stars')
    # DR19 median trend for comparison
    axs[2].plot(
        rolling_medians['age'], rolling_medians['ce_mg'], 
        'w-', 
        linewidth=2
    )
    axs[2].plot(
        rolling_medians['age'], rolling_medians['ce_mg'], 
        'k-', 
        label='Rolling median'
    )
    axs[2].set_title('(c) OCCAM', y=0.82)

    # Compare with Casali et al. (2025) best-fit relations
    # xarr = np.arange(0, 11.1, 0.1)
    # for feh in [-0.6, -0.3, 0, 0.3]:
    #     axs[3].plot(xarr, casali_relation(xarr, feh), label='[Fe/H]=%s' % feh)
    # axs[3].plot(
    #     rolling_medians['age'], rolling_medians['ce_mg'], 
    #     'k-', 
    #     label='Rolling median'
    # )
    # axs[3].set_title('Casali et al. (2025)')
    # axs[3].legend()

    # Axes adjustments
    axs[0].set_xlim(xlim)
    axs[0].set_ylim(ylim)
    axs[0].xaxis.set_major_locator(MultipleLocator(5))
    axs[0].xaxis.set_minor_locator(MultipleLocator(1))
    axs[0].yaxis.set_major_locator(MultipleLocator(0.5))
    axs[0].yaxis.set_minor_locator(MultipleLocator(0.1))
    axs[-1].set_xlabel('Age [Gyr]')
    for ax in axs:
        ax.set_ylabel('[Ce/Mg]')
    plt.savefig(paths.figures / 'dataset_comparison')
    plt.close()


def casali_relation(age, feh):
    """
    Global fit to the age-[Ce/Mg] relation from Casali et al. (2025).
    """
    return -0.032 * age + 0.194 * feh + 0.092


def join_apokasc_mwm():
    """Join APOKASC-3 catalogs with abundances and parameters from MWM DR19."""
    # APOKASC-3 catalog
    apokasc3 = ascii.read(paths.data / 'APOGEE' / 'apokasc3_rec_mrt.txt').to_pandas()
    apokasc3['APOGEE_ID'] = apokasc3['2MASS'].str.replace('2MASS J', '2M')
    # Select gold sample only
    apokasc3 = apokasc3[
        (apokasc3['CatTab'] == 'Gold') &
        ((apokasc3['EvolState'] == 'RGB') |
         (apokasc3['EvolState'] == 'RC'))
    ]
    # apokasc3['AgeBest'] = apokasc3['AgeRGB'].copy()
    apokasc3['AgeBest'] = apokasc3['AgeRGB'].where(
        apokasc3['EvolState'] == 'RGB',
        apokasc3['AgeRC']
    )
    # Import full DR19 catalog
    mwm_full = fits_to_pandas(
        paths.data / 'MWM' / 'astraAllStarASPCAP-0.6.0.fits.gz', 
        hdu=2
    )
    # drop duplicate SDSS-V IDs with the lowest SNR
    mwm_full.sort_values(['sdss_id', 'snr'], inplace=True, ascending=True)
    mwm_full.drop_duplicates(subset='sdss_id', keep='last', inplace=True)
    # Drop bad spectrum and abundance flags
    mwm_good = mwm_full[
        (mwm_full['flag_bad'] == 0) & 
        (mwm_full['spectrum_flags'] == 0) &
        (mwm_full['snr'] > 50) &
        (mwm_full['ce_h_flags'] == 0) &
        (mwm_full['mg_h_flags'] == 0) &
        (mwm_full['sdss_id'] > 0) &
        (mwm_full['logg'] < 3.5) &
        (mwm_full['teff'] > 4000)
    ].copy()
    mwm_good['ce_mg'], mwm_good['e_ce_mg'] = abundance_ratio(mwm_good, 'ce', 'mg')
    # Merge MWM Ce with APOKASC
    apokasc3 = apokasc3.set_index('APOGEE_ID').join(
        mwm_good[['sdss4_apogee_id', 'ce_mg', 'e_ce_mg']].set_index('sdss4_apogee_id')
    )
    apokasc3.rename(columns={'ce_mg': 'MWM_CE_MG', 'e_ce_mg': 'MWM_CE_MG_ERR'}, inplace=True)
    return apokasc3



if __name__ == '__main__':
    main()
