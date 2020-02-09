#!/bin/tcsh


# predictands
# set predictand = (TS)
set predictand = (TS prec_t)
#~ set predictand = (prec_t)

# possible precursors
set arr = (Z500 FSNO SST ICEFRAC PSL)
#~ set arr = (PSL)
# set arr = (ICEFRAC)

foreach pred ($predictand)
    foreach var ($arr) 
    # set last parameter to -1 if all states from predictand should be
    # taken, otherwise set length of precursors
    # 5th argument is for bootstrap method.
        #~ python main.py ini/composites_America_$var.ini $var $pred 6043 3
        #~ python main.py ini/composites_America_$var.ini $var $pred 6043 0.0001
        python main_composites_model-1.py ini/composites_America_${var}1.ini $var $pred 1980 99.99
    end
end
# ini/composites_America_TS.ini SST TS 2100 99.99
