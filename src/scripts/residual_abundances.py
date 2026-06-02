"""
Plot metallicity-corrected residual Ce abundances as a function of
position in the Galaxy.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from utils import binned_quantiles, sample_rows
from plotting import colored_text_legend, setup_hayden_plot, iterate_rz_bins
from colormaps import paultol
import paths

RBINS = [(3, 5), (5, 7), (7, 9), (9, 11), (11, 13)] # left to right
ZBINS = [(1, 2), (0.5, 1), (0, 0.5)] # top to bottom
ALPHA_BUFFER = 0.02 # dex, buffer around the [Mg/Fe] dividing line
# SAMPLE_FRACTION = 0.25 # fraction of stars to plot in each panel
SAMPLE_SIZE = 1000 # number of stars to plot in each panel, randomly selected

def main(style='paper'):
    plt.style.use(paths.styles / f'{style}.mplstyle')
    # Import MWM sample
    mwm_rgb = pd.read_csv(paths.data / 'sample.csv')
    mwm_res = residual_abundances(mwm_rgb)
    mwm_res_ages = mwm_res[mwm_res['good_age']].copy() # Good ages only

    age_bin_edges = np.arange(0.5, 11.6, 1)

    # Solar neighborhood sample
    local_low_alpha = mwm_res_ages[
        (mwm_res_ages['Rg'] >= 7) &
        (mwm_res_ages['Rg'] < 9) &
        (mwm_res_ages['z_max'] < 0.5) &
        (mwm_res_ages['low_alpha'])
    ].copy()
    local_high_alpha = mwm_res_ages[
        (mwm_res_ages['Rg'] >= 7) &
        (mwm_res_ages['Rg'] < 9) &
        (mwm_res_ages['z_max'] < 0.5) &
        (mwm_res_ages['high_alpha'])
    ].copy()
    # Calculate median trend with age (only stars with good ages)
    local_low_alpha_age_medians = binned_quantiles(
        local_low_alpha, 'delta_ce_h', 'age',
        q=0.5, bin_edges=age_bin_edges, min_count=10
    )
    local_high_alpha_age_medians = binned_quantiles(
        local_high_alpha, 'delta_ce_h', 'age',
        q=0.5, bin_edges=age_bin_edges, min_count=10
    )

    fig, axs = setup_hayden_plot(rbins=RBINS, zbins=ZBINS)
    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95)
    # scatterplot style arguments
    kwargs = dict(s=1, marker='.', rasterized=True, edgecolor='none')
    high_alpha_color = paultol.highcontrast.colors[2]
    low_alpha_color = paultol.highcontrast.colors[0]

    for i, j, zlim, rlim in iterate_rz_bins(rbins=RBINS, zbins=ZBINS):
        subset = mwm_res_ages[
            (mwm_res_ages['Rg'] >= rlim[0]) &
            (mwm_res_ages['Rg'] < rlim[1]) &
            (mwm_res_ages['z_max'] >= zlim[0]) &
            (mwm_res_ages['z_max'] < zlim[1])
        ]
        low_alpha = subset[subset['low_alpha']].copy()
        high_alpha = subset[subset['high_alpha']].copy()
        # Select random sample of stars for scatter plot
        sample = sample_rows(subset, SAMPLE_SIZE)
        low_alpha_sample = sample[sample['low_alpha']]
        high_alpha_sample = sample[sample['high_alpha']]
        if low_alpha.shape[0] >= 100:
            # Scatter plot random sample of points
            axs[i,j].scatter(
                low_alpha.loc[low_alpha_sample.index, 'age'], 
                low_alpha.loc[low_alpha_sample.index, 'delta_ce_h'],
                c=low_alpha_color, zorder=2, **kwargs
            )
            # Plot median trend with age
            age_medians = binned_quantiles(
                low_alpha, 'delta_ce_h', 'age',
                q=0.5, bin_edges=age_bin_edges, min_count=10
            )
            axs[i,j].plot(
                *age_medians, '.-', color=low_alpha_color, zorder=6,
                label='High-Ia'
            )
        if high_alpha.shape[0] >= 100:
            # Scatter plot random sample of points
            axs[i,j].scatter(
                high_alpha.loc[high_alpha_sample.index, 'age'], 
                high_alpha.loc[high_alpha_sample.index, 'delta_ce_h'],
                c=high_alpha_color, zorder=1, **kwargs
            )
            # Plot median trend with age
            age_medians = binned_quantiles(
                high_alpha, 'delta_ce_h', 'age',
                q=0.5, bin_edges=age_bin_edges, min_count=10
            )
            axs[i,j].plot(
                *age_medians, '.-', color=high_alpha_color, zorder=5,
                label='Low-Ia'
            )
        # Plot local low and high-alpha trends for comparison
        axs[i,j].plot(
            *local_low_alpha_age_medians, 
            linestyle='--', color=low_alpha_color, zorder=4,
        )
        axs[i,j].plot(
            *local_high_alpha_age_medians, 
            linestyle='--', color=high_alpha_color, zorder=3,
        )
        # Horizontal line for reference
        axs[i,j].plot([0, 12], [0, 0], linestyle=':', color='gray', zorder=0)
    # Indicate median abundance errors
    age_err_low = np.median(mwm_res_ages['age'] - mwm_res_ages['e_n_age'])
    age_err_high = np.median(mwm_res_ages['e_p_age'] - mwm_res_ages['age'])
    med_abund_err = mwm_res_ages['e_ce_h'].median()
    axs[0,0].errorbar(
        3, -0.4, 
        xerr=[[age_err_low], [age_err_high]], 
        yerr=med_abund_err, 
        c='gray', capsize=0, #elinewidth=0.5,
    )

    # Format axes
    axs[0,0].set_xlim((0, 12))
    axs[0,0].set_ylim((-0.6, 0.6))
    axs[0,0].xaxis.set_major_locator(MultipleLocator(5))
    axs[0,0].xaxis.set_minor_locator(MultipleLocator(1))
    axs[0,0].yaxis.set_major_locator(MultipleLocator(0.5))
    axs[0,0].yaxis.set_minor_locator(MultipleLocator(0.1))
    for ax in axs[-1,:]:
        ax.set_xlabel('Age [Gyr]')
    for ax in axs[:,0]:
        ax.set_ylabel(r'$\Delta$[Ce/H]')
    # Text-only lengend
    leg = colored_text_legend(axs[0,0], loc='upper left')

    plt.savefig(paths.figures / 'residual_abundances')


def residual_abundances(
        catalog, 
        col='ce_h_corr', 
        newcol='delta_ce_h', 
        rbins=RBINS,
        zbins=ZBINS
    ):
    """
    Calculate residual [Ce/H] abundances for high- and low-alpha populations
    separately in each Galactic region (defined by bins in Rg and z_max).
    
    Parameters
    ----------
    catalog : pandas.DataFrame
        Full MWM catalog.
    col : str, optional [default: 'ce_h_corr']
        Column with abundances to calculate the residuals for.
    newcol : str, optional [default: 'delta_ce_h']
        Name of new column with residual abundances
    rbins : list of tuples, optional
        List of bins in guiding radius (Rg) in which to calculate residuals.
    zbins : list of tuples, optional
        List of bins in z_max in which to calculate residuals.

    Returns
    -------
    catalog : pandas.DataFrame
        MWM catalog with residual abundance column appended.
    """
    mg_bin_edges = np.arange(-0.75, 0.76, 0.1)
    # Calculate residuals separately in each region for low-alpha
    res_abund = []
    for i, j, zlim, rlim in iterate_rz_bins(rbins=rbins, zbins=zbins):
        subset = catalog[
            (catalog['Rg'] >= rlim[0]) &
            (catalog['Rg'] < rlim[1]) &
            (catalog['z_max'] >= zlim[0]) &
            (catalog['z_max'] < zlim[1]) &
            (catalog['low_alpha'])
        ].copy()
        if subset.shape[0] >= 100:
            # Calculate median trend with [Mg/H]
            mgh_medians = binned_quantiles(
                subset, col, 'mg_h', 
                q=0.5, bin_edges=mg_bin_edges, min_count=10
            )
            # Calculate residual Ce abundance
            subset[newcol] = subset[col] - np.interp(
                subset['mg_h'], *mgh_medians
            )
            res_abund.append(subset[['sdss_id', newcol]].copy())
    # Calculate high-alpha residuals all together
    all_high_alpha = catalog[catalog['high_alpha']].copy()
    high_alpha_medians = binned_quantiles(
        all_high_alpha, col, 'mg_h', 
        q=0.5, bin_edges=mg_bin_edges, min_count=10
    )
    all_high_alpha[newcol] = all_high_alpha[col] - np.interp(
        all_high_alpha['mg_h'], *high_alpha_medians
    )
    res_abund.append(all_high_alpha[['sdss_id', newcol]].copy())
    # Join residual abundances to catalog DataFrame
    res_abund = pd.concat(res_abund)
    res_abund.set_index('sdss_id', inplace=True)
    catalog = catalog.join(res_abund, on='sdss_id')
    return catalog


if __name__ == '__main__':
    main()
