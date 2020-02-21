#!/bin/tcsh

########################################################################
# shift longitudinal range from -180 to 180 to 0 - 360                 #
# program usage: ./sst_shift_longitudes.sh                             #
########################################################################


set var = SST
# make variable name lower case
set var_small = ` echo $var | tr "[A-Z]" "[a-z]" `
# create file to shift longitudinal range from -180 to 180 to 0 - 360 
cdo -f nc sellonlatbox,-0,360,-89.5,89.5 -random,r360x180 cdoregridmap1x1.nc
# go through models
foreach i (`seq 1 1 35`)
    echo "Remap data for model $i to 0 - 360 degrees\n"
    # add padding, e.g. 1 --> 001
    set pad_i = `echo $i | sed -e :a -e 's/^.\{1,2\}$/0&/;ta'`
    set inputfile = /home/sonja/Documents/Clustering-Forecast/input_cluster/input_cluster_anom_sst/$var_small.final.$pad_i.192001-197912.nc
    set outputfile = /home/sonja/Documents/Clustering-Forecast/input_cluster/input_cluster_anom_sst-180/$var_small.final.$pad_i.192001-197912.nc
    # create directory if it does not exist
    mkdir -p /home/sonja/Documents/Clustering-Forecast/input_cluster/input_cluster_anom_sst-180/
    # remap $var from $inputfile according to cdoregridmap1x1.nc and save data in $outputfile
    cdo -f nc remapbil,cdoregridmap1x1.nc -selname,$var $inputfile $outputfile
end
rm cdoregridmap1x1.nc
