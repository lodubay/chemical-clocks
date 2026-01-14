"""
Plot the [Ce/H] and [Ce/Mg] radial gradient as a function of stellar age.
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

def main(style='paper', cmap='jet_r'):
    plt.style.use(paths.styles / f'{style}.mplstyle')
    # Import MWM sample
    mwm_rgb = pd.read_csv(paths.data / 'MWM' / 'MWM_RGB.csv')
    radius_bin_edges = np.arange(2.5, 15.6, 1)
    age_bin_edges = np.arange(0, 10.1, 2)

    # Select only low-alpha, near-midplane stars
    mwm_rgb = apply_alpha_cut(mwm_rgb)
    all_lowz = good_ages(mwm_rgb[(mwm_rgb['z_max'] < 0.5)].copy())
    all_low_alpha = good_ages(mwm_rgb[(mwm_rgb['z_max'] < 0.5) & (mwm_rgb['low_alpha'])].copy())
    
    # Set up figure
    fig, axs = plt.subplots(
        2, 1,
        figsize=(ONE_COLUMN_WIDTH, 1.5*ONE_COLUMN_WIDTH),
        sharex=True,
        gridspec_kw={'hspace': 0.}
    )
    cax = insert_colorbar_axes(fig, 'horizontal', pad=0.05)
    age_cmap = plt.get_cmap(cmap)
    norm = BoundaryNorm(age_bin_edges, age_cmap.N)
    xlim = (0, 16)
    ylim = [(-0.6, 0.6), (-0.5, 0.7)]

    for i, col in enumerate(['ce_h', 'ce_mg']):
        # Plot all stars
        pcm = axs[i].hexbin(
            all_lowz['Rg'], all_lowz[col],
            C=np.ones(all_lowz.shape[0]),
            reduce_C_function=np.sum,
            gridsize=(30, 12),
            cmap='binary',
            norm=Normalize(vmin=0, vmax=350),
            linewidths=0.2,
            # mincnt=1,
            extent=[xlim[0], xlim[1], ylim[i][0], ylim[i][1]]
        )
        # fig.colorbar(pcm, ax=axs[i])

        for j in range(len(age_bin_edges)-1):
            age_bin = age_bin_edges[j:j+2]
            mean_age = np.mean(age_bin)
            subset = all_lowz[
                (all_lowz['age'] >= age_bin[0]) &
                (all_lowz['age'] < age_bin[1])
            ]
            Rg_medians = binned_quantiles(
                subset, col, 'Rg',
                q=0.5, bin_edges=radius_bin_edges, min_count=10
            )
            axs[i].plot(*Rg_medians, '-', color='w', linewidth=2)
            axs[i].plot(
                *Rg_medians, '-', 
                color=age_cmap(norm(mean_age)), 
                label=f'{int(mean_age)} Gyr'
            )
    fig.colorbar(pcm, cax=cax, orientation='horizontal', label='Number of stars')

    axs[0].set_xlim(xlim)
    axs[0].set_ylim(ylim[0])
    axs[1].set_ylim(ylim[1])

    axs[0].yaxis.set_major_locator(MultipleLocator(0.5))
    axs[0].yaxis.set_minor_locator(MultipleLocator(0.1))
    axs[1].yaxis.set_major_locator(MultipleLocator(0.5))
    axs[1].yaxis.set_minor_locator(MultipleLocator(0.1))
    axs[1].xaxis.set_major_locator(MultipleLocator(4))
    axs[1].xaxis.set_minor_locator(MultipleLocator(1))

    axs[0].set_ylabel('[Ce/H]')
    axs[1].set_ylabel('[Ce/Mg]')
    axs[1].set_xlabel(r'$R_{\rm guide}$ [kpc]')

    leg = colored_text_legend(axs[0], loc='center left', title='Age')
    leg = colored_text_legend(axs[1], loc='center left', title='Age')

    plt.savefig(paths.figures / 'ce_gradient')
    plt.close()


if __name__ == '__main__':
    main()
