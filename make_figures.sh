#!/bin/bash
############################################################
# Help                                                     #
############################################################
Help()
{
   # Display Help
   echo "Generate all figures and tables for the manuscript."
   echo
   echo "Syntax: make_figures.sh [-h|o]"
   echo "options:"
   echo "h     Print this Help."
   echo "o     Overwrite main sample file (re-generate from scratch)."
   echo
}

############################################################
# Main program                                             #
############################################################
# Get the options
while getopts ":h" option; do
   case $option in
      h) # display Help
         Help
         exit;;
      o) # overwrite sample.csv
         overwrite=true;;
   esac
done

# Move to src/scripts directory
cd ./src/scripts/

# Generate MWM sample data table
if $overwrite; then
    echo "Generating sample file..."
    python sample.py
else
    if [ -f ../data/sample.csv ]; then
        echo "Found sample summary file!"
    else
        echo "Generating sample file..."
        python sample.py
    fi
fi

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

# Tables & other output
echo "Generating Tables 2 and 3..."
python logg_abundance_offsets.py
echo "Generating sample size files..."
python sample_size.py
