#!/bin/tcsh


set var = SST
set var_small = ` echo $var | tr "[A-Z]" "[a-z]" `
foreach i (`seq 1 1 35`)
    echo "Remap data for model $i to regular grid\n"
    # add padding
    set pad_i = `echo $i | sed -e :a -e 's/^.\{1,2\}$/0&/;ta'`
    if ($i == 1) then
        set inputfile = /home/sonja/Documents/Composites/input_cluster/input_cluster_sst/SST.001.185001-210012.nc
        set outputfile = /home/sonja/Documents/Composites/input_cluster/input_cluster_anom_$var_small/$var_small.final.$pad_i.192001-197912.nc
        set outputfile_detrend = /home/sonja/Documents/Composites/input_cluster/input_cluster_detrend_$var_small/$var_small.final.$pad_i.192001-198012.nc
        set outputfile_NOTdetrend = /home/sonja/Documents/Composites/input_cluster/input_cluster_no_trend_$var_small/$var_small.final.$pad_i.192001-198012.nc
        mkdir -p /home/sonja/Documents/Composites/input_cluster/input_cluster_detrend_$var_small/
        mkdir -p /home/sonja/Documents/Composites/input_cluster/input_cluster_no_trend_$var_small/
        
        cdo remapbil,global_1 -selvar,SST $inputfile remap.nc
        cdo selyear,1920/1979 remap.nc remap_cut.nc
        cdo detrend remap_cut.nc $outputfile_detrend
        cdo timmean remap_cut.nc timmean.nc
        cdo add detrend.nc timmean.nc $outputfile
        cp  remap_cut.nc $outputfile_NOTdetrend
        
        rm detrend.nc timmean.nc detrended_timmean.nc remap.nc remap_cut.nc
    else
        set inputfile = /home/sonja/Documents/Composites/input_cluster/input_cluster_sst/SST.$pad_i.185001-210012.nc
        set outputfile = /home/sonja/Documents/Composites/input_cluster/input_cluster_anom_$var_small/$var_small.final.$pad_i.192001-197912.nc
        set outputfile_detrend = /home/sonja/Documents/Composites/input_cluster/input_cluster_detrend_$var_small/$var_small.final.$pad_i.192001-198012.nc
        set outputfile_NOTdetrend = /home/sonja/Documents/Composites/input_cluster/input_cluster_no_trend_$var_small/$var_small.final.$pad_i.192001-198012.nc
        mkdir -p /home/sonja/Documents/Composites/input_cluster/input_cluster_detrend_$var_small/
        mkdir -p /home/sonja/Documents/Composites/input_cluster/input_cluster_no_trend_$var_small/
        cdo remapbil,global_1 -selvar,SST $inputfile remap.nc

        cdo selyear,1920/1979 remap.nc remap_cut.nc
        cdo detrend remap_cut.nc $outputfile_detrend
        cdo timmean remap_cut.nc timmean.nc
        cdo add detrend.nc timmean.nc $outputfile
        cp  remap_cut.nc $outputfile_NOTdetrend
        rm detrend.nc timmean.nc detrended_timmean.nc remap.nc remap_cut.nc
    endif
end
