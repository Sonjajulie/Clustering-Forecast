#!/bin/tcsh

# documentation: http://www.cesm.ucar.edu/projects/community-projects/LENS/data-sets.html


# variables in array:

set predictand = (prec_t)

# possible precursors
set arr = (FSNO ICEFRAC Z500 SST PSL SST-Pacific FSNO-America)
# set arr = (FSNO ICEFRAC)

foreach pred ($predictand)
    foreach var ($arr) 
    # set last parameter to -1 if all states from predictand should be
    # taken, otherwise set length of precursors
    # 5th argument is for bootstrap method.
    sbatch --output=output_cl_${var}_${pred}.out --error=error_cl_${var}_${pred}.err --job-name=cl_f_${var}_${pred} main_casper_forecast_opt.sh $var $pred standardized
    end
end

