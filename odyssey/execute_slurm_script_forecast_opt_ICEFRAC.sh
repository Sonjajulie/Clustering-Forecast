#!/bin/bash

# documentation: http://www.cesm.ucar.edu/projects/community-projects/LENS/data-sets.html
# set last parameter to -1 if all states from predictand should be
# taken, otherwise set length of precursors
# 5th argument is for bootstrap method.

# variables in array:
# predictand=( prec_t TS )
predictand=( TS )

for pred in "${predictand[@]}"
do
    sbatch --output=output_opt_${pred}_ICEFRAC.out --error=error_opt_${pred}_ICEFRAC.err --job-name=cl_opt_$pred main_odyssey_forecast_opt_ICEFRAC.sh ${pred} 0 1980 standardized-opt
done
