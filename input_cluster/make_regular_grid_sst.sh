#!/bin/tcsh


set var = SST
set var_small = ` echo $var | tr "[A-Z]" "[a-z]" `
foreach i (`seq 1 1 35`)
    echo "Remap data for model $i to regular grid\n"
    # add padding
    set pad_i = `echo $i | sed -e :a -e 's/^.\{1,2\}$/0&/;ta'`
    if ($i == 1) then
        set inputfile = /home/sonja/Documents/Composites/input_cluster/input_cluster_sst/SST.001.185001-210012.nc
        set outputfile = /home/sonja/Documents/Composites/input_cluster/input_cluster_anom_$var_small/$var_small.final.$pad_i.192001-198912.nc
        set outputfile_detrend = /home/sonja/Documents/Composites/input_cluster/input_cluster_detrend_$var_small/$var_small.final.$pad_i.192001-198012.nc
        set outputfile_NOTdetrend = /home/sonja/Documents/Composites/input_cluster/input_cluster_no_trend_$var_small/$var_small.final.$pad_i.192001-198012.nc
            
        cdo remapbil,global_1 -selvar,SST $inputfile remap.nc
        cdo detrend remap.nc detrend.nc
        cdo timmean remap.nc timmean.nc
        cdo add detrend.nc timmean.nc detrended_timmean.nc 
        #~ cdo delete,year=2100 detrended_timmean.nc $outputfile
        cdo selyear,1920/1989 detrended_timmean.nc $outputfile
        cdo selyear,1920/1989 detrend.nc $outputfile_detrend
        cdo selyear,1920/1989 $inputfile $outputfile_NOTdetrend
        rm detrend.nc timmean.nc detrended_timmean.nc remap.nc
    else
        set inputfile = /home/sonja/Documents/Composites/input_cluster/input_cluster_sst/SST.$pad_i.185001-210012.nc
        set outputfile = /home/sonja/Documents/Composites/input_cluster/input_cluster_anom_$var_small/$var_small.final.$pad_i.192001-198912.nc
        set outputfile_detrend = /home/sonja/Documents/Composites/input_cluster/input_cluster_detrend_$var_small/$var_small.final.$pad_i.192001-198012.nc
        set outputfile_NOTdetrend = /home/sonja/Documents/Composites/input_cluster/input_cluster_no_trend_$var_small/$var_small.final.$pad_i.192001-198012.nc
        
        cdo remapbil,global_1 -selvar,SST $inputfile remap.nc
        cdo detrend remap.nc detrend.nc
        cdo timmean remap.nc timmean.nc
        cdo add detrend.nc timmean.nc detrended_timmean.nc 
        cdo selyear,1920/1989 detrended_timmean.nc $outputfile
        cdo selyear,1920/1989 detrend.nc $outputfile_detrend
        cdo selyear,1920/1989 $inputfile $outputfile_NOTdetrend
        rm detrend.nc timmean.nc detrended_timmean.nc remap.nc
    endif
end
