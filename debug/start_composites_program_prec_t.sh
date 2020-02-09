#!/bin/tcsh


# predictands
# set predictand = (prec_t)
set predictand = (TS prec_t)

# possible precursors
#~ set arr = (Z500 FSNO SST ICEFRAC PSL)
set arr = (ICEFRAC)

foreach pred ($predictand)
    foreach var ($arr) 
    # set last parameter to -1 if all states from predictand should be
    # taken, otherwise set length of precursors
    # 5th argument is for bootstrap method.
        #~ python main.py ini/composites_America_$var.ini $var $pred 6043 3
        #~ python main.py ini/composites_America_$var.ini $var $pred 6043 0.0001
        python main.py ini/composites_America_$var.ini $var $pred 6009 99.9
    end
end
