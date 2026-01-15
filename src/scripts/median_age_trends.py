"""
Compare median trends in [Ce/Mg] and residual [Ce/H] as a function of age
and guiding-center radius.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import BoundaryNorm, Normalize

from utils import apply_alpha_cut, binned_quantiles, sample_rows, good_ages
from plotting import insert_colorbar_axes, colored_text_legend, truncate_colormap
from _globals import ONE_COLUMN_WIDTH
from colormaps import paultol
import paths

def main(style='paper', cmap='viridis_r'):
    plt.style.use(paths.styles / f'{style}.mplstyle')
    # Import MWM sample
    mwm_rgb = pd.read_csv(paths.data / 'MWM' / 'MWM_RGB.csv')
    radius_bin_edges = np.arange(3, 15.1, 2)
    age_bin_edges = np.arange(0.5, 11.6, 1)
    mg_bin_edges = np.arange(-0.75, 0.76, 0.1)

    # Select only low-alpha, near-midplane stars
    mwm_rgb = apply_alpha_cut(mwm_rgb)
    all_low_alpha = mwm_rgb[(mwm_rgb['z_max'] < 0.5) & (mwm_rgb['low_alpha'])].copy()

    # Calculate residual [Ce/H]
    all_low_alpha['delta_ce_h'] = np.nan * np.ones(all_low_alpha.shape[0])
    for i in range(len(radius_bin_edges)-1):
        radius_bin = radius_bin_edges[i:i+2]
        subset = all_low_alpha[
            (all_low_alpha['Rg'] >= radius_bin[0]) &
            (all_low_alpha['Rg'] < radius_bin[1])
        ]
        medians = binned_quantiles(
            subset, 'ce_h_corr', 'mg_h',
            q=0.5, bin_edges=mg_bin_edges, min_count=10
        )
        all_low_alpha.loc[subset.index, 'delta_ce_h'] = subset['ce_h_corr'] - np.interp(
            subset['mg_h'], *medians
        )
    
    # Set up figure
    fig, axs = plt.subplots(
        2, 1,
        figsize=(ONE_COLUMN_WIDTH, 1.5*ONE_COLUMN_WIDTH),
        sharex=True,
        gridspec_kw={'hspace': 0.}
    )
    cax = insert_colorbar_axes(fig, 'horizontal', pad=0.05)
    radial_cmap = plt.get_cmap(cmap)
    norm = BoundaryNorm(radius_bin_edges, radial_cmap.N)
    xlim = (0, 11)
    ylim = [(-0.3, 0.7), (-0.5, 0.5)]

    low_alpha_ages = good_ages(all_low_alpha)
    for i, col in enumerate(['ce_mg', 'delta_ce_h']):
        # Plot all stars
        pcm = axs[i].hexbin(
            low_alpha_ages['age'], low_alpha_ages[col],
            C=np.ones(low_alpha_ages.shape[0]),
            reduce_C_function=np.sum,
            gridsize=(30, 12),
            cmap='binary',
            norm=Normalize(vmin=0, vmax=260),
            linewidths=0.2,
            mincnt=1,
            extent=[xlim[0], xlim[1], ylim[i][0], ylim[i][1]]
        )
        # fig.colorbar(pcm, ax=axs[i])

        for j in range(len(radius_bin_edges)-1):
            radius_bin = radius_bin_edges[j:j+2]
            mean_radius = np.mean(radius_bin)
            subset = low_alpha_ages[
                (low_alpha_ages['Rg'] >= radius_bin[0]) &
                (low_alpha_ages['Rg'] < radius_bin[1])
            ]
            age_medians = binned_quantiles(
                subset, col, 'age',
                q=0.5, bin_edges=age_bin_edges, min_count=10
            )
            axs[i].plot(*age_medians, '-', color='w', linewidth=2)
            axs[i].plot(
                *age_medians, '-', 
                color=radial_cmap(norm(mean_radius)), 
                label=f'{int(mean_radius)} kpc'
            )
    fig.colorbar(pcm, cax=cax, orientation='horizontal', label='Number of stars')

    axs[0].set_xlim(xlim)
    axs[0].set_ylim(ylim[0])
    axs[1].set_ylim(ylim[1])

    axs[0].set_ylabel('[Ce/Mg]')
    axs[1].set_ylabel(r'$\Delta$[Ce/H]')
    axs[1].set_xlabel('Age [Gyr]')

    leg = colored_text_legend(axs[0], ncols=2, loc='upper right')

    plt.savefig(paths.figures / 'median_age_trends')
    plt.close()


if __name__ == '__main__':
    main()
