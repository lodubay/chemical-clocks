"""
Apply Roberts+ (2025) [C/N]-based stellar ages to MWM DR19.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import binned_quantiles, get_bin_centers
import paths
from plotting import TWO_COLUMN_WIDTH

# Coefficients for [C/N] age fit polynomial
CN_AGE_COEF = np.array([-1.721,  0.806,  -0.077,  0.276, -0.643, 10.048])

def main(style='paper'):
    mwm_rgb = pd.read_csv(paths.data / 'MWM' / 'sample.csv')
    mwm_rgb = mwm_rgb[mwm_rgb['good_age']].copy()
    mwm_rgb['log_age'] = np.log10(mwm_rgb['age'])
    mwm_rgb = generate_cn_ages(mwm_rgb)

    linear_lim = (0, 12)
    log_lim = (-0.5, 1.2)

    plt.style.use(paths.styles / f'{style}.mplstyle')
    fig, axs = plt.subplots(2, 2, height_ratios=[2, 1],
        figsize=(TWO_COLUMN_WIDTH, 0.67 * TWO_COLUMN_WIDTH),
        sharex='col', gridspec_kw={'hspace': 0}
    )
    kwargs = dict(c='k', marker='.', s=1, edgecolors='none', rasterized=True)
    axs[0,0].scatter(mwm_rgb['age'], mwm_rgb['cn_age'], **kwargs)
    axs[1,0].scatter(mwm_rgb['age'], mwm_rgb['cn_age'] - mwm_rgb['age'], **kwargs)
    axs[0,1].scatter(mwm_rgb['log_age'], mwm_rgb['cn_log_age'], **kwargs)
    axs[1,1].scatter(mwm_rgb['log_age'], mwm_rgb['cn_log_age'] - mwm_rgb['log_age'], **kwargs)
    # 1-1 lines
    axs[0,0].plot(linear_lim, linear_lim, 'r-')
    axs[1,0].plot(linear_lim, [0, 0], 'r-')
    axs[0,1].plot(log_lim, log_lim, 'r-')
    axs[1,1].plot(log_lim, [0, 0], 'r-')
    # Median trends
    linear_bin_edges = np.arange(0, 12.5, 0.5)
    linear_bin_centers = get_bin_centers(linear_bin_edges)
    _, linear_cn_medians = binned_quantiles(
        mwm_rgb, 'cn_age', 'age', bin_edges=linear_bin_edges, q=0.5
    )
    axs[0,0].plot(
        linear_bin_centers, linear_cn_medians, 
        color='gray', linestyle='-'
    )
    axs[1,0].plot(
        linear_bin_centers, linear_cn_medians - linear_bin_centers, 
        color='gray', linestyle='-'
    )
    log_bin_edges = np.arange(-0.5, 1.3, 0.1)
    log_bin_centers = get_bin_centers(log_bin_edges)
    _, log_cn_medians = binned_quantiles(
        mwm_rgb, 'cn_log_age', 'log_age', bin_edges=log_bin_edges, q=0.5
    )
    axs[0,1].plot(
        log_bin_centers, log_cn_medians, 
        color='gray', linestyle='-'
    )
    axs[1,1].plot(
        log_bin_centers, log_cn_medians - log_bin_centers, 
        color='gray', linestyle='-'
    )

    axs[0,0].set_xlim(linear_lim)
    axs[0,0].set_ylim(linear_lim)
    axs[1,0].set_ylim((-6, 6))
    axs[0,1].set_xlim(log_lim)
    axs[0,1].set_ylim(log_lim)
    axs[1,1].set_ylim((-1.1, 1.1))

    axs[0,0].set_ylabel('[C/N] Age [Gyr]')
    axs[1,0].set_ylabel('Residual')
    axs[1,0].set_xlabel('StarFlow Age [Gyr]')
    axs[0,1].set_ylabel('[C/N] log(age)')
    axs[1,1].set_ylabel('Residual')
    axs[1,1].set_xlabel('StarFlow log(age)')

    plt.savefig(paths.figures / 'cn_ages')
    plt.close()


def generate_cn_ages(data):
    """
    Calculate [C/N]-based ages for a subset of the APOGEE sample.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Full MWM dataset (post-cuts okay), must include [C/N] data.
    
    Returns
    -------
    pandas.DataFrame
        [C/N]-based ages and errors indexed on APOGEE IDs.
    
    Reference
    ---------
    Roberts, J. et al (in prep)
    """
    # Hard edge cuts
    goldregion = data[
        (data['logg'] >= 0.5) & (data['logg'] < 3.26) &
        (data['c_n'] >= -0.75) & (data['c_n'] < 1.0) &
        (data['e_c_n'] < 0.1)
    ].copy()
    # Get evolutionary state
    goldregion = evol_state(goldregion)
    goldregion['delta_teff'] = reference_temperature(
        goldregion['raw_fe_h'], goldregion['raw_logg']
    ) - goldregion['teff']
    LRGB = goldregion[
        (goldregion['evol_state'] == 1) & (goldregion['logg'] >= 2.5)
    ]
    URGB = goldregion[
        (goldregion['evol_state'] == 1) & (goldregion['logg'] < 2.5)
    ]
    RC = goldregion[goldregion['evol_state'] == 2]
    # Remove [Fe/H] < -0.4 for URGB and RC stars, then re-merge
    cn_age_region = pd.concat([
        LRGB[(LRGB['delta_teff'] >= -515) & (LRGB['delta_teff'] <= 340)], 
        URGB[
            (URGB['delta_teff'] >= -515) & (URGB['delta_teff'] <= 340) & 
            (URGB['fe_h'] >= -0.4)
        ], 
        RC[
            (RC['delta_teff'] >= -620) & (RC['delta_teff'] <= 100) & 
            (RC['fe_h'] >= -0.4)
        ]
    ]).copy()
    cn_log_age, cn_log_age_err = recover_age_quad(
        cn_age_region['c_n'].values, 
        cn_age_region['fe_h'].values, 
        CN_AGE_COEF,
    )
    # Convert years -> Gyr
    cn_age_region['cn_log_age'] = cn_log_age - 9.
    cn_age_region['e_cn_log_age'] = cn_log_age_err
    cn_age_region['cn_age'] = 10 ** cn_age_region['cn_log_age']
    cn_age_region['e_cn_age'] = cn_age_region['cn_age'] * np.log(10) * cn_age_region['e_cn_log_age']
    data = data.join(cn_age_region[
        ['cn_age', 'e_cn_age', 'cn_log_age', 'e_cn_log_age']
    ])
    return data


def recover_age_quad(cn_arr, feh_arr, params, cn_err=[], feh_err=[]):
    """
    Compute stellar ages via polynomial fit to [C/N] and [Fe/H].
    
    Parameters
    ----------
    cn_arr : array-like
        Array of stellar [C/N] abundances.
    feh_arr : array-like
        Array of stellar [Fe/H] abundances. Must be same length as cn_arr.
    params : array-like
        Polynomial fit coefficients. Must have length 6.
    
    Returns
    -------
    ages: array-like
        Array of log10(stellar ages in years).
    age_errors: array-like
        Array of error in log-age.
    
    Notes
    -----
    Thanks Jack.
    """
    assert len(cn_arr) == len(feh_arr)
    c2,c1,f2,f1,c1f1,b = params
    #check for stars with [C/N] past the parabola peak
    maxcn = (c1+c1f1*feh_arr)/(-2*c2)
    badcn = cn_arr > maxcn
    # Calculate ages
    ages = (c2*cn_arr**2) + (c1*cn_arr) + (f2*feh_arr**2) + (f1*feh_arr) + \
        (c1f1*feh_arr*cn_arr)+b 
    #for stars with c/n past the parabola peak, use the parabola peak instead
    ages[badcn] = (c2*maxcn[badcn]**2) + (c1*maxcn[badcn]) + \
        (f2*feh_arr[badcn]**2) + (f1*feh_arr[badcn]) + \
        (c1f1*feh_arr[badcn]*maxcn[badcn]) + b 
    # Generic age errors of 1.64 Gyr
    age_errors = 1.64e9 / (np.log(10) * 10 ** ages)
    # Propagate errors
    if len(cn_err) > 0 and len(feh_err) > 0:
        age_errors = np.sqrt(
            cn_err**2 * (2*c2*cn_arr + c1 + c1f1*feh_arr)**2 +
            feh_err**2 * (2*f2*feh_arr + f1 * c1f1*cn_arr)**2
        )
        # inflate by 40% per Jack
        age_errors *= 1.4
    return ages, age_errors


def evol_state(dataplot, verbose=False):
    """
    Using Warfield's APOK2 paper to separate RGB from RC stars
    1 is RGB, 2 is RC

    References
    ----------
    Warfield et al. (2024), ApJ 167:208
    """
    #Calculate Reference Temperature
    Tref = reference_temperature(dataplot['raw_fe_h'], dataplot['raw_logg'])
   
    #Calculate Equation A4 value
    a = 0.05915
    b = 0.003455
    c = 155.1
    criterion = a - (b*((c*dataplot['raw_fe_h']) + dataplot['raw_teff'] - Tref)) - (dataplot['raw_c_m_atm'] - dataplot['raw_n_m_atm'])
   
    #Apply Criterion
    loggrgb = dataplot['logg'] < 2.3 #These stars always RGB
    critrgb = criterion > 0 #A4 > 0 is RGB (Swapped from the paper)
    rgb = np.logical_or(loggrgb,critrgb) #Either case makes RGB
    critrc = criterion <= 0 #A4 < 0 is RC (Swapped from the paper)
    rc = np.logical_and(np.invert(loggrgb),critrc) #Must be A4<0 and not urgb
    if verbose: #double check the counts to make sure nothing got missed or double counted
        print(f"{sum(rc)} RC stars and {sum(rgb)} RGB stars")
        print(f"{dataplot.shape[0]} Total stars: Difference of {dataplot.shape[0] - (sum(rc) + sum(rgb))}")
    #Create output flags
    flagger = np.zeros(dataplot.shape[0])
    flagger[rgb] = flagger[rgb] + 1
    flagger[rc] = flagger[rc] + 2
    dataplot['evol_state'] = flagger
    return dataplot


def reference_temperature(feh, logg):
    """
    Calculate "reference" temperature as defined by
    Schonhut-Stasik et al. (2024).

    Parameters
    ----------
    feh : float or array-like
        Uncalibrated (spectroscopic) metallicity.
    logg : float or array-like
        Uncalibrated (spectroscopic) surface gravity.
    
    Returns
    -------
    Tref : float or array-like
        Reference temperature.
    """
    alp = 4427.18
    bet = -399.5
    gam = 553.17
    Tref = alp + (bet * feh) + (gam*(logg-2.5))
    return Tref


if __name__ == '__main__':
    main()
