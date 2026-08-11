#!/bin/bash

# Move to src/scripts directory
cd ./src/scripts/

# Generate MWM sample data table
echo "Generating sample file..."
python sample.py

# Plots
echo "Plotting Figure 1..."
python logg_calibrations.py
echo "Plotting Figures 2 and 3..."
python ce_lines.py
echo "Plotting Figure 4..."
python sample_distributions.py
echo "Plotting Figure 5..."
python dataset_comparison.py
echo "Plotting Figure 6..."
python cemg_mgh_age.py
echo "Plotting Figure 7..."
python local_metallicity_fits.py
echo "Plotting Figure 8..."
python global_metallicity_fits.py
echo "Plotting Figure 9..."
python median_trends_grid.py
echo "Plotting Figure 10..."
python residual_abundances.py
echo "Plotting Figure 11..."
python residual_age_trends.py
echo "Plotting Figure 12..."
python gradients.py
echo "Plotting Figure 13..."
python halo.py
echo "Done!"
