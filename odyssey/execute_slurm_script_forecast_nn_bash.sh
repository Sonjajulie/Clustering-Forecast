#!/bin/bash

# documentation: http://www.cesm.ucar.edu/projects/community-projects/LENS/data-sets.html
# set last parameter to -1 if all states from predictand should be
# taken, otherwise set length of precursors
# 5th argument is for bootstrap method.

# variables in array:
#predictand=( prec_t TS )
#
#for pred in "${predictand[@]}"
#do
#    sbatch --output=output_$pred.out --error=error_$pred.err --job-name=cl_$pred main_odyssey_forecast_nn.sh ${pred} 0 1980 standardized
#done
predictand=( TS )
# precursors=(FSNO-America FSNO-Eurasia  ICEFRAC Z500 SST PSL)

#~ precursors=(FSNO-America FSNO-Eurasia Z500 SST PSL)
precursors=(ICEFRAC)

for pred in "${predictand[@]}"
do
    for var in "${precursors[@]}"
    do
        sbatch --output=output_opt_${pred}_nn_${var}.out --error=error_opt_nn_${pred}_${var}.err --job-name=cl_nn_opt_${pred}_${var} main_odyssey_forecast_nn.sh ${pred} ${var} 0 1980 standardized-opt
    done
done
