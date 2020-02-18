#!/bin/tcsh

########################################################################
# calculate detrended model data                                       #
# different for-loops, because variables have different input names    #
# program usage: ./detrend_anom_data_1920_1980.sh                      #
########################################################################


# variables to detrend
set arr=( TOTAL_PREC)
# go through array
foreach var ($arr)
    # make variable name lower case 
    set var_small = ` echo $var | tr "[A-Z]" "[a-z]" `
    # go through all models 1 - 35
    foreach i (`seq 1 1 35`)
        echo "Remap data for model $i to detrended anomaly data\n"
        # add padding, e.g. 1 --> 001
        set pad_i = `echo $i | sed -e :a -e 's/^.\{1,2\}$/0&/;ta'` # psl.001.185001-210012.nc  psl.006.192001-210012.nc
        # model 1 has a different input name than 2 - 35 model
        if ($i == 1) then
            set inputfile = /home/sonja/Documents/Composites/input_cluster/input_cluster_$var_small/$var_small.final.001.185001-210012.nc
            set outputfile = /home/sonja/Documents/Composites/input_cluster/input_cluster_anom_$var_small/$var_small.final.$pad_i.192001-198012.nc
            # select only model years 1921 - 1980
            cdo selyear,1921/1980 $inputfile cut.nc
            # detrend data, which also calculates anomalies
            cdo detrend cut.nc detrend.nc
            # calculate time mean
            cdo timmean cut.nc timmean.nc
            # add detrend to time mean to get the original variable values but detrended
            cdo add detrend.nc timmean.nc $outputfile
            # remove unnecessary files
            rm detrend.nc timmean.nc cut.nc
        else
            # calculate detrended data for models 2 - 35 in the same way as in model 1
            set inputfile =  /home/sonja/Documents/Composites/input_cluster/input_cluster_$var_small/$var_small.final.$pad_i.192001-210012.nc
            set outputfile = /home/sonja/Documents/Composites/input_cluster/input_cluster_anom_$var_small/$var_small.final.$pad_i.192001-198012.nc
            cdo selyear,1921/1980 $inputfile cut.nc
            cdo detrend cut.nc detrend.nc
            cdo timmean cut.nc timmean.nc
            cdo add detrend.nc timmean.nc $outputfile 
            rm detrend.nc timmean.nc cut.nc
        endif
    end
end

# variables to detrend
set arr=( TS)
# go through array
foreach var ($arr)
    # make variable name lower case  
    set var_small = $var #` echo $var | tr "[A-Z]" "[a-z]" `
    # go through all models 1 - 35
    foreach i (`seq 1 1 35`)
        echo "Remap data for model $i to detrended anomaly data\n"
        # add padding, e.g. 1 --> 001
        set pad_i = `echo $i | sed -e :a -e 's/^.\{1,2\}$/0&/;ta'` # psl.001.185001-210012.nc  psl.006.192001-210012.nc
        # model 1 has a different input name than 2 - 35 model
        if ($i == 1) then
            set inputfile = /home/sonja/Documents/Composites/input_cluster/input_cluster_$var_small/$var_small.final.001.185001-210012.nc
            set outputfile = /home/sonja/Documents/Composites/input_cluster/input_cluster_anom_$var_small/$var_small.final.$pad_i.192001-198012.nc
            # select only model years 1921 - 1980
            cdo selyear,1921/1980 $inputfile cut.nc
            # detrend data, which also calculates anomalies
            cdo detrend cut.nc detrend.nc
            # calculate time mean
            cdo timmean cut.nc timmean.nc
            # add detrend to time mean to get the original variable values but detrended
            cdo add detrend.nc timmean.nc $outputfile
            # remove unnecessary files 
            rm detrend.nc timmean.nc cut.nc
        else
            # calculate detrended data for models 2 - 35 in the same way as in model 1
            set inputfile =  /home/sonja/Documents/Composites/input_cluster/input_cluster_$var_small/$var_small.final.$pad_i.192001-210012.nc
            set outputfile = /home/sonja/Documents/Composites/input_cluster/input_cluster_anom_$var_small/$var_small.final.$pad_i.192001-198012.nc

            cdo selyear,1921/1980 $inputfile cut.nc
            cdo detrend cut.nc detrend.nc
            cdo timmean cut.nc timmean.nc
            cdo add detrend.nc timmean.nc $outputfile 
            rm detrend.nc timmean.nc cut.nc
        endif
    end
end

# variables to detrend
set arr=( FSNO ICEFRAC PSL)
# go through array
foreach var ($arr)
    # make variable name lower case   
    set var_small = ` echo $var | tr "[A-Z]" "[a-z]" `
    # go through all models 1 - 35
    foreach i (`seq 1 1 35`)
        echo "Remap data for model $i to detrended anomaly data\n"
        # add padding, e.g. 1 --> 001
        set pad_i = `echo $i | sed -e :a -e 's/^.\{1,2\}$/0&/;ta'` # psl.001.185001-210012.nc  psl.006.192001-210012.nc
        if ($i == 1) then
            set inputfile = /home/sonja/Documents/Composites/input_cluster/input_cluster_$var_small/$var_small.001.185001-210012.nc
            set outputfile = /home/sonja/Documents/Composites/input_cluster/input_cluster_anom_$var_small/$var_small.final.$pad_i.192001-197912.nc
            # select only model years 1920 - 1979
            cdo selyear,1920/1979 $inputfile cut.nc
            # detrend data, which also calculates anomalies
            cdo detrend cut.nc detrend.nc
            # calculate time mean
            cdo timmean cut.nc timmean.nc
            # add detrend to time mean to get the original variable values but detrended
            cdo add detrend.nc timmean.nc $outputfile 
            # remove unnecessary files 
            rm detrend.nc timmean.nc cut.nc
            
        else
            # calculate detrended data for models 2 - 35 in the same way as in model 1
            set inputfile =  /home/sonja/Documents/Composites/input_cluster/input_cluster_$var_small/$var_small.$pad_i.192001-210012.nc
            set outputfile = /home/sonja/Documents/Composites/input_cluster/input_cluster_anom_$var_small/$var_small.final.$pad_i.192001-197912.nc
                        
            cdo selyear,1920/1979 $inputfile cut.nc
            cdo detrend cut.nc detrend.nc
            cdo timmean cut.nc timmean.nc
            cdo add detrend.nc timmean.nc $outputfile 
            rm detrend.nc timmean.nc cut.nc
        endif
    end
end

# variables to detrend
set arr=(Z500)
# go through array
foreach var ($arr)
    # make variable name lower case   
    set var_small = $var #` echo $var | tr "[A-Z]" "[a-z]" `
    # go through all models 1 - 35
    foreach i (`seq 1 1 35`)
        echo "Remap data for model $i to detrended anomaly data\n"
        # add padding, e.g. 1 --> 001
        set pad_i = `echo $i | sed -e :a -e 's/^.\{1,2\}$/0&/;ta'` # psl.001.185001-210012.nc  psl.006.192001-210012.nc
        if ($i == 1) then
            set inputfile = /home/sonja/Documents/Composites/input_cluster/input_cluster_$var_small/$var_small.001.18500101-21001231.nc
            set outputfile = /home/sonja/Documents/Composites/input_cluster/input_cluster_anom_$var_small/$var_small.final.$pad_i.192001-197912.nc
            # select only model years 1920 - 1979
            cdo selyear,1920/1979 $inputfile cut.nc
            # detrend data, which also calculates anomalies
            cdo detrend cut.nc detrend.nc
            # calculate time mean
            cdo timmean cut.nc timmean.nc
            # add detrend to time mean to get the original variable values but detrended
            cdo add detrend.nc timmean.nc $outputfile
            # remove unnecessary files 
            rm detrend.nc timmean.nc cut.nc
        else
            # calculate detrended data for models 2 - 35 in the same way as in model 1
            set inputfile =  /home/sonja/Documents/Composites/input_cluster/input_cluster_$var_small/$var_small.$pad_i.19200101-21001231.nc
            set outputfile = /home/sonja/Documents/Composites/input_cluster/input_cluster_anom_$var_small/$var_small.final.$pad_i.192001-197912.nc
                        
            cdo selyear,1920/1979 $inputfile cut.nc
            cdo detrend cut.nc detrend.nc
            cdo timmean cut.nc timmean.nc
            cdo add detrend.nc timmean.nc $outputfile 
            rm detrend.nc timmean.nc cut.nc
        endif
    end
end
