# Chemical Clocks in Milky Way Mapper

Plotting scripts and LaTeX source code for Dubay et al. (in prep),
"A Broken Clock is Right Twice a Day: [Ce/Mg] is Not a Universal Chemical Clock".

To re-build the manuscript yourself, first ensure you have the right
packages by creating a Conda environment from the `environment.yml` file:
```
$ conda env create -f environment.yml
$ conda activate chemical-clocks
```

The required data files are too big to be stored in this repository. The
summary data file with all necessary information can be shared upon request.
Place the `sample.csv` file within the `src/data` directory.

To re-create all plots and tables in the manuscript, run the following:
```
$ bash make_figures.sh
```

## Repository Structure

```
.
├── src
│   ├── data                # Catalog files and model outputs (ignored by git)
│   ├── extra               # Non-paper figures and other outputs (ignored by git)
│   ├── scripts             # Python scripts for figures and tables
│   ├── tex                 # LaTeX manuscript files
│   ├── ├── figures         # Programatically-generated figures
│   ├── ├── output          # Programatically-generated tables and other outputs
├── make_figures.sh         # Script to produce all paper figures
├── environment.yml         # Package dependencies
├── LICENSE
└── README.md
```

## Software Dependencies

- numpy
- pandas
- matplotlib
- astropy
- scipy
- scikit-learn
- statsmodels
- sdss-access
