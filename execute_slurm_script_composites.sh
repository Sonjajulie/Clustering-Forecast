#!/bin/tcsh

# documentation: http://www.cesm.ucar.edu/projects/community-projects/LENS/data-sets.html


# variables in array:

set predictand = (prec_t TS)

# possible precursors
#~ set arr = (FSNO ICEFRAC Z500 SST  PSL)
set arr = (FSNO ICEFRAC)

foreach pred ($predictand)
    foreach var ($arr) 
    # set last parameter to -1 if all states from predictand should be
    # taken, otherwise set length of precursors
    # 5th argument is for bootstrap method.
    sbatch --output=output_$var.out --error=error_$var.err --job-name=cl_$var main_cheyenne_composites.sh $var $pred 1980 1 not-standardized
    end
end

