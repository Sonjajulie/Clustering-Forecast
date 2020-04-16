#!/bin/bash

# documentation: http://www.cesm.ucar.edu/projects/community-projects/LENS/data-sets.html
# set last parameter to -1 if all states from predictand should be
# taken, otherwise set length of precursors
# 5th argument is for bootstrap method.

# variables in array:
predictand=( TS )

for pred in "${predictand[@]}"
do
    sbatch --output=output_$pred.out --error=error_$pred.err --job-name=cl_$pred main_odyssey_forecast.sh ${pred} 0 1980 standardized
done
