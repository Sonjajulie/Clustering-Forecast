#!/bin/tcsh

# documentation: http://www.cesm.ucar.edu/projects/community-projects/LENS/data-sets.html


# variables in array:

#   precp - total precipitation
#   ts - surface temperautre
set arr=(prec_t TS)
foreach var ($arr)
    sbatch --output=output_$var.out --error=error_$var.err --job-name=cl_$var main_cheyenne_clusters.sh $var
end
