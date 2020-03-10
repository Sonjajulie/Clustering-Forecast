#!/bin/tcsh


# predictands
# set predictand = (TS)
set predictand = (prec_t TS)

# possible precursors
set arr = (FSNO ICEFRAC Z500 SST  PSL)
#~ set arr = (FSNO)
# set arr = (PSL)
# set arr = (ICEFRAC)

foreach pred ($predictand)
    foreach var ($arr) 
    # set last parameter to -1 if all states from predictand should be
    # taken, otherwise set length of precursors
    # 5th argument is for bootstrap method.
        #~ python main.py ini/composites_America_$var.ini $var $pred 6043 3
        #~ python main.py ini/composites_America_$var.ini $var $pred 6043 0.0001
        python main_composites.py ini/composites_America_$var.ini $var $pred 1980 20
    end
end
# ini/composites_America_TS.ini SST TS 2100 99.99
