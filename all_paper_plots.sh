#!/bin/bash

# Move to src/scripts directory
cd ./src/scripts/

# Generate MWM sample data table
python mwm_sample.py

# Plots
echo "Plotting Figure 1..."
python kiel_diagram.py
echo "Plotting Figure 2..."
python logg_calibrations.py
echo "Plotting Figure 3..."
python cemg_mgh_age.py
echo "Plotting Figure 4..."
python dataset_comparison.py
echo "Plotting Figure 5..."
python local_metallicity_trends.py
echo "Plotting Figure 6..."
python median_trends_grid.py
echo "Plotting Figure 7..."
python residual_explainer.py
echo "Plotting Figure 8..."
python median_age_trends.py
echo "Plotting Figure 9..."
python ce_gradient.py
echo "Plotting Figure 10..."
python halo.py
echo "Plotting Figure 11..."
python onezone_sfh.py
echo "Plotting Figure 12..."
python onezone_agb.py
echo "Done!"
