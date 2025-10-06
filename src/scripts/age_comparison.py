import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import ascii

import paths
from utils import fits_to_pandas
import _globals

def main():
    plt.style.use(paths.styles / 'paper.mplstyle')
    mwm_rgb = pd.read_csv(paths.data / 'MWM' / 'MWM_RGB.csv')
    mwm_rgb.rename(
        columns={'age': 'starflow_age', 'e_p_age': 'e_p_starflow_age', 'e_n_age': 'e_n_starflow_age'},
        inplace=True
    )
    mwm_cols = ['sdss_id', 'sdss4_apogee_id', 'starflow_age', 'e_p_starflow_age', 'e_n_starflow_age']
    # Join APOKASC-3 catalog
    apokasc = fits_to_pandas(paths.data / 'APOGEE' / 'APOKASC_cat_v7.4.0.fits')
    apokasc.replace(-9999, np.nan, inplace=True)
    apokasc3_cols = [
        '2MASS_ID', 'APOKASC3_EVSTATE', 'APOKASC3_CAT_TAB', 
        'APOKASC3_AGE_RGB', 'APOKASC3_AGE_RGB_PERR', 'APOKASC3_AGE_RGB_MERR', 
        'APOKASC3_AGE_RCAGB', 'APOKASC3_AGE_RCAGB_PERR', 'APOKASC3_AGE_RCAGB_MERR'
    ]
    apokasc_mwm = apokasc[apokasc3_cols].join(mwm_rgb[mwm_cols].set_index('sdss4_apogee_id'), on='2MASS_ID')
    apokasc_mwm.dropna(subset='sdss_id', inplace=True)
    # Join APO-K2 age catalogs
    rgb_table = ascii.read(paths.data / 'APOGEE' / 't1_apok2_rgb_ages.txt')
    apok2 = rgb_table.to_pandas()
    apok2.rename(
        columns={'Age': 'APO-K2_Age', 'e_Age': 'APO-K2_e_n_Age', 'E_Age': 'APO-K2_e_p_Age'},
        inplace=True
    )
    apok2_cols = ['APOGEE', 'APO-K2_Age', 'APO-K2_e_n_Age', 'APO-K2_e_p_Age']
    apok2_mwm = apok2[apok2_cols].join(mwm_rgb[mwm_cols].set_index('sdss4_apogee_id'), on='APOGEE')

    figwidth = _globals.TWO_COLUMN_WIDTH
    fig, axs = plt.subplots(
        1, 3, sharex=True, sharey=True, figsize=(figwidth, 0.33 * figwidth),
        gridspec_kw={'wspace': 0}
    )
    kwargs = {'s': 1, 'rasterized': True, 'edgecolor': 'none'}
    xlim = (0, 16)

    # one-to-one line
    axs[0].plot(xlim, xlim, c='r', linewidth=0.5)
    axs[1].plot(xlim, xlim, c='r', linewidth=0.5)
    axs[2].plot(xlim, xlim, c='r', linewidth=0.5)

    apokasc_mwm_rgb_gold = apokasc_mwm[
        (apokasc_mwm['APOKASC3_EVSTATE'] == 'RGB') &
        (apokasc_mwm['APOKASC3_CAT_TAB'] == 'Gold')
    ]
    axs[0].scatter(
        apokasc_mwm_rgb_gold['APOKASC3_AGE_RGB'], 
        apokasc_mwm_rgb_gold['starflow_age'],
        label='Gold', c='gold', **kwargs
    )
    apokasc_mwm_rgb_silver = apokasc_mwm[
        (apokasc_mwm['APOKASC3_EVSTATE'] == 'RGB') &
        (apokasc_mwm['APOKASC3_CAT_TAB'] == 'Silver')
    ]
    axs[0].scatter(
        apokasc_mwm_rgb_silver['APOKASC3_AGE_RGB'], 
        apokasc_mwm_rgb_silver['starflow_age'],
        label='Silver', c='silver', **kwargs
    )

    apokasc_mwm_rc = apokasc_mwm[apokasc_mwm['APOKASC3_EVSTATE'] == 'RC']
    axs[1].scatter(
        apokasc_mwm_rc['APOKASC3_AGE_RCAGB'], 
        apokasc_mwm_rc['starflow_age'],
        c='k', **kwargs
    )

    # Plot APO-K2 comparison
    axs[2].scatter(
        apok2_mwm['APO-K2_Age'], apok2_mwm['starflow_age'],
        c='k', **kwargs
    )

    axs[0].set_xlim(xlim)
    axs[0].set_ylim(xlim)
    axs[0].set_ylabel('StarFlow Age [Gyr]')
    axs[0].set_xlabel('APOKASC-3 RGB Age [Gyr]')
    axs[1].set_xlabel('APOKASC-3 RC Age [Gyr]')
    axs[2].set_xlabel('APO-K2 Age [Gyr]')
    axs[0].legend(loc='upper left', markerscale=2)
    plt.savefig(paths.figures / 'age_comparison')


if __name__ == '__main__':
    main()
