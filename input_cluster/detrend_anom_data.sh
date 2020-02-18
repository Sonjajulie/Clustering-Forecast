#!/bin/tcsh


#~ set arr=( TOTAL_PREC)
#~ foreach var ($arr) 
    #~ set var_small = ` echo $var | tr "[A-Z]" "[a-z]" `
    #~ foreach i (`seq 1 1 35`)
        #~ echo "Remap data for model $i to detrended anomaly data\n"
        #~ # add padding
        #~ set pad_i = `echo $i | sed -e :a -e 's/^.\{1,2\}$/0&/;ta'` # psl.001.185001-210012.nc  psl.006.192001-210012.nc
        #~ if ($i == 1) then
            #~ set inputfile = /home/sonja/Documents/Composites/input_cluster/input_cluster_$var_small/$var_small.final.001.185001-210012.nc
            #~ set outputfile = /home/sonja/Documents/Composites/input_cluster/input_cluster_anom_$var_small/$var_small.final.$pad_i.185001-210012.nc
            #~ cdo detrend $inputfile detrend.nc
            #~ cdo timmean $inputfile timmean.nc
            #~ cdo add detrend.nc timmean.nc $outputfile
            #~ rm detrend.nc timmean.nc
        #~ else
            #~ set inputfile =  /home/sonja/Documents/Composites/input_cluster/input_cluster_$var_small/$var_small.final.$pad_i.192001-210012.nc
            #~ set outputfile = /home/sonja/Documents/Composites/input_cluster/input_cluster_anom_$var_small/$var_small.final.$pad_i.192001-210012.nc
            #~ cdo detrend $inputfile detrend.nc
            #~ cdo timmean $inputfile timmean.nc
            #~ cdo add detrend.nc timmean.nc $outputfile
            #~ rm detrend.nc timmean.nc
        #~ endif
    #~ end
#~ end

#~ set arr=( TS)
#~ foreach var ($arr) 
    #~ set var_small = $var #` echo $var | tr "[A-Z]" "[a-z]" `
    #~ foreach i (`seq 1 1 35`)
        #~ echo "Remap data for model $i to detrended anomaly data\n"
        #~ # add padding
        #~ set pad_i = `echo $i | sed -e :a -e 's/^.\{1,2\}$/0&/;ta'` # psl.001.185001-210012.nc  psl.006.192001-210012.nc
        #~ if ($i == 1) then
            #~ set inputfile = /home/sonja/Documents/Composites/input_cluster/input_cluster_$var_small/$var_small.final.001.185001-210012.nc
            #~ set outputfile = /home/sonja/Documents/Composites/input_cluster/input_cluster_anom_$var_small/$var_small.final.$pad_i.185001-210012.nc
            #~ cdo detrend $inputfile detrend.nc
            #~ cdo timmean $inputfile timmean.nc
            #~ cdo add detrend.nc timmean.nc $outputfile
            #~ rm detrend.nc timmean.nc
        #~ else
            #~ set inputfile =  /home/sonja/Documents/Composites/input_cluster/input_cluster_$var_small/$var_small.final.$pad_i.192001-210012.nc
            #~ set outputfile = /home/sonja/Documents/Composites/input_cluster/input_cluster_anom_$var_small/$var_small.final.$pad_i.192001-210012.nc
            #~ cdo detrend $inputfile detrend.nc
            #~ cdo timmean $inputfile timmean.nc
            #~ cdo add detrend.nc timmean.nc $outputfile
            #~ rm detrend.nc timmean.nc
        #~ endif
    #~ end
#~ end



# PSL Z500
set arr=( FSNO ICEFRAC PSL)
foreach var ($arr) 
    set var_small = ` echo $var | tr "[A-Z]" "[a-z]" `
    foreach i (`seq 1 1 35`)
        echo "Remap data for model $i to detrended anomaly data\n"
        # add padding
        set pad_i = `echo $i | sed -e :a -e 's/^.\{1,2\}$/0&/;ta'` # psl.001.185001-210012.nc  psl.006.192001-210012.nc
        if ($i == 1) then
            set inputfile = /home/sonja/Documents/Composites/input_cluster/input_cluster_$var_small/$var_small.001.185001-210012.nc
            set outputfile = /home/sonja/Documents/Composites/input_cluster/input_cluster_anom_$var_small/$var_small.final.$pad_i.185001-210012.nc
            cdo detrend $inputfile detrend.nc
            cdo timmean $inputfile timmean.nc
            cdo add detrend.nc timmean.nc deterend_timmean.nc 
            cdo delete,year=2100 deterend_timmean.nc $outputfile
            rm detrend.nc timmean.nc deterend_timmean.nc 
        else
            set inputfile =  /home/sonja/Documents/Composites/input_cluster/input_cluster_$var_small/$var_small.$pad_i.192001-210012.nc
            set outputfile = /home/sonja/Documents/Composites/input_cluster/input_cluster_anom_$var_small/$var_small.final.$pad_i.192001-210012.nc
            cdo detrend $inputfile detrend.nc
            cdo timmean $inputfile timmean.nc
            cdo add detrend.nc timmean.nc deterend_timmean.nc 
            cdo delete,year=2100 deterend_timmean.nc $outputfile
            rm detrend.nc timmean.nc deterend_timmean.nc 
        endif
    end
end


set arr=(Z500)
foreach var ($arr) 
    set var_small = $var #` echo $var | tr "[A-Z]" "[a-z]" `
    foreach i (`seq 1 1 35`)
        echo "Remap data for model $i to detrended anomaly data\n"
        # add padding
        set pad_i = `echo $i | sed -e :a -e 's/^.\{1,2\}$/0&/;ta'` # psl.001.185001-210012.nc  psl.006.192001-210012.nc
        if ($i == 1) then
            set inputfile = /home/sonja/Documents/Composites/input_cluster/input_cluster_$var_small/$var_small.001.18500101-21001231.nc
            set outputfile = /home/sonja/Documents/Composites/input_cluster/input_cluster_anom_$var_small/$var_small.final.$pad_i.185001-210012.nc
            cdo detrend $inputfile detrend.nc
            cdo timmean $inputfile timmean.nc
            cdo add detrend.nc timmean.nc deterend_timmean.nc 
            cdo delete,year=2100 deterend_timmean.nc $outputfile
            rm detrend.nc timmean.nc deterend_timmean.nc 
        else
            set inputfile =  /home/sonja/Documents/Composites/input_cluster/input_cluster_$var_small/$var_small.$pad_i.19200101-21001231.nc
            set outputfile = /home/sonja/Documents/Composites/input_cluster/input_cluster_anom_$var_small/$var_small.final.$pad_i.192001-210012.nc
            cdo detrend $inputfile detrend.nc
            cdo timmean $inputfile timmean.nc
            cdo add detrend.nc timmean.nc deterend_timmean.nc 
            cdo delete,year=2100 deterend_timmean.nc $outputfile
            rm detrend.nc timmean.nc deterend_timmean.nc 
        endif
    end
end
