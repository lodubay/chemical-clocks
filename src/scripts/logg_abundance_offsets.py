"""
Generate LaTeX tables of abundance offsets as a function of log(g) for
[Fe/H] and [Ce/H].
"""
import numpy as np
import pandas as pd

import paths
from utils import get_bin_centers

def main():
    # Load calibration grids
    fe_offsets = np.load(paths.data / 'fe_offset_grid.npy')
    ce_offsets = np.load(paths.data / 'ce_offset_grid.npy')

    # Cut rows with log(g) < 1 and log(g) > 3
    fe_offsets = fe_offsets.T[2:-1]
    ce_offsets = ce_offsets.T[2:-1]

    fe_table = make_table(fe_offsets)
    with open(paths.output / 'fe_offset_table.tex', 'w') as f:
        f.write(fe_table)

    ce_table = make_table(ce_offsets)
    with open(paths.output / 'ce_offset_table.tex', 'w') as f:
        f.write(ce_table)

def make_table(offsets_grid):
    # Initialize grid of log(g), [Mg/H] values
    MgH_bin_edges = np.round(np.linspace(-0.75, 0.45, 13, endpoint=True), 2)
    MgH_bin_centers = get_bin_centers(MgH_bin_edges)
    logg_bin_edges = np.linspace(1.0, 3.0, 5, endpoint=True)
    logg_bin_centers = get_bin_centers(logg_bin_edges)

    # Generate LaTeX tables
    df = pd.DataFrame(
        offsets_grid, 
        index=pd.Series([f'{l:.2f}' for l in logg_bin_centers], name=r'$\log(g)$'),
        columns=pd.Series([f'{m:.1f}' for m in MgH_bin_centers], name='[Mg/H]')
    )
    latex_table = df.to_latex(
        column_format='r|' + 'c'*len(MgH_bin_centers),
        float_format='%.2g'
    )
    latex_table = latex_table.replace('-0 ', '0 ')
    # Replace \toprule, \midrule, \bottomrule with \hline
    latex_table = latex_table.replace('\\toprule', '\\hline\\hline')
    latex_table = latex_table.replace('\\midrule', '\\hline')
    latex_table = latex_table.replace('\\bottomrule', '\\hline')

    return latex_table


if __name__ == '__main__':
    main()
