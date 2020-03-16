#!/bin/tcsh

# documentation: http://www.cesm.ucar.edu/projects/community-projects/LENS/data-sets.html


# variables in array:

#~ set predictand = (prec_t TS)
set predictand = (prec_t)

# possible precursors
# set arr = (FSNO-Eurasia ICEFRAC Z500 SST-Pacific  PSL FSNO-America)
set arr = ('FSNO-Eurasia' 'SST-Pacific' 'FSNO-America' 'ICEFRAC' 'PSL' 'Z500' 'SST' 'Z500-NA')

foreach pred ($predictand)
    foreach var ($arr) 
    # set last parameter to -1 if all states from predictand should be
    # taken, otherwise set length of precursors
    # 5th argument is for bootstrap method.
    sbatch --output=output_${var}_${pred}.out --error=error_${var}_${pred}.err --job-name=cl_${var}_${pred} main_casper_forecast_opt.sh $var $pred standardized
    end
end

