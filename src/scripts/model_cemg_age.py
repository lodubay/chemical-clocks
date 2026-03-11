"""
This script plots a Hayden-style [Ce/Mg]-age plot of a multizone model
compared to MWM data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MultipleLocator
import vice

from multizone_stars import MultizoneStars
from plotting import setup_hayden_plot, iterate_rz_bins, insert_colorbar_axes
from utils import plot_gas_abundance
from stats import weighted_quantile, kde2D
import paths
from multizone._globals import END_TIME

OUTPUT_NAME = 'agb-mscale/diskmodel'


def main(style='paper', cmap='viridis'):
    # Import MWM sample
    sample = pd.read_csv(paths.data / 'MWM' / 'sample.csv')
    # Set up figure
    plt.style.use(paths.styles / f'{style}.mplstyle')
    fig, axs = setup_hayden_plot()
    # Enlarge for colorbar
    figsize = fig.get_size_inches()
    fig.set_size_inches((figsize[0], figsize[1] * 1.25))
    cmap = plt.get_cmap(cmap)
    met_bins = np.arange(-0.8, 0.21, 0.2)
    cbar = fig.colorbar(
        ScalarMappable(
            BoundaryNorm(met_bins, cmap.N, extend='both'), 
            cmap
        ), 
        ax=axs,
        shrink=0.6, 
        aspect=30, 
        fraction=0.1, 
        pad=0.1,
        orientation='horizontal',
        label='[Mg/H]'
    )
    # Plot multizone output
    mzs = MultizoneStars.from_output(OUTPUT_NAME)
    mzs.model_uncertainty(sample, inplace=True)
    for i, j, zlim, rlim in iterate_rz_bins():
        ax = axs[i,j]
        mzs_subset = mzs.region(rlim, zlim)
        mzs_subset.scatter_plot(
            ax, 'age', '[ce/mg]', color='[mg/h]',
            cmap=cbar.cmap, norm=cbar.norm
        )
        plot_gas_abundance(
            ax, mzs_subset, 'lookback', '[ce/mg]', c='k', ls='--'
        )
        # Plot MWM median trends
        sample_subset = sample[
            (sample['Rg'] >= rlim[0]) &
            (sample['Rg'] < rlim[1]) &
            (sample['z_max'] >= zlim[0]) &
            (sample['z_max'] < zlim[1])
        ]
        running_median(ax, sample_subset, 'ce_mg_corr')
    
    # Format axes
    axs[0,0].set_xlim((0, END_TIME))
    axs[0,0].set_ylim((-0.8, 0.8))
    for ax in axs[-1]:
        ax.set_xlabel('Age [Gyr]')
    for ax in axs[:,0]:
        ax.set_ylabel('[Ce/Mg]')
    # Set x-axis ticks
    axs[0,0].xaxis.set_major_locator(MultipleLocator(5))
    axs[0,0].xaxis.set_minor_locator(MultipleLocator(1))
    # Set y-axis ticks
    axs[0,0].yaxis.set_major_locator(MultipleLocator(0.5))
    axs[0,0].yaxis.set_minor_locator(MultipleLocator(0.1))
    
    plt.savefig(paths.figures / 'model_cemg_age')
    plt.close()


def running_median(
        ax, 
        data, 
        col, 
        label=None, 
        color='r', 
        age_col='age', 
        window=1000,
        alpha=0.2, 
        linestyle='-', 
        marker='o', 
        **kwargs):
    """
    Plot running stellar abundance medians and 1-sigma range binned by age.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis on which to plot the medians.
    data : pandas.DataFrame
        MWM data sample.
    col : str
        Data column with abundance data.
    label : str, optional
        The main scatter plot / error bar label. The default is None.
    age_col : str, optional
        Name of column containing ages. The default is 'age'.
    window : int, optional
        Rolling window size. The default is 1000.
    alpha : float, optional
        Transparency of the 1-sigma range. The default is 0.2.
    **kwargs passed to matplotlib.pyplot.fill_between
    
    Returns
    -------
    spatch : matplotlib.patches.StepPatch
    pcol : matplotlib.collections.FillBetweenPolyCollection
    """
    # Sort by ascending age
    sorted_ages = data.sort_values(age_col)[[age_col, col]]
    # Calculate rolling median
    rolling_params = dict(
        min_periods=int(window/5), step=int(window/5), on=age_col, center=True
    )
    rolling_medians = sorted_ages.rolling(window, **rolling_params).median()
    # Rolling 16th and 84th percentiles
    rolling_low = sorted_ages.rolling(window, **rolling_params).quantile(0.16)
    rolling_high = sorted_ages.rolling(window, **rolling_params).quantile(0.84)
    pcol = ax.fill_between(
        rolling_medians[age_col],
        rolling_low[col],
        rolling_high[col],
        # step='post', 
        color=color, 
        alpha=alpha, 
        label=label, 
        edgecolor=color, 
        linestyle=linestyle,
        **kwargs
    )
    line2d = ax.plot(rolling_medians[age_col], rolling_medians[col], 
                     color=color, linestyle=linestyle, marker='none')
    return line2d[0], pcol


if __name__ == '__main__':
    main()
