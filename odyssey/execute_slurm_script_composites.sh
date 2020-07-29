#!/bin/bash

# documentation: http://www.cesm.ucar.edu/projects/community-projects/LENS/data-sets.html



# possible precursors
# precursors=(FSNO ICEFRAC Z500 SST PSL)
precursors=(FSNO )

# variables in array:
# predictand=( TS prec_t)
predictand=( TS )

for pred in "${predictand[@]}"
do
    for var in "${precursors[@]}"
    do
    # set last parameter to -1 if all states from predictand should be
    # taken, otherwise set length of precursors
    # 5th argument is for bootstrap method.
    # python3 main_composites.py -ini ../ini/composites_America_FSNO.ini --outputlabel not-standardized --predictand TS --logfile logs/log_FSNO_TS.log --numbers 1980 --percentage 1
    sbatch --output=output_${var}_${pred}.out --error=error_${var}_${pred}.err --job-name=cl_$var main_odyssey_composites.sh $var $pred 1980 1 not-standardized
    done
done

