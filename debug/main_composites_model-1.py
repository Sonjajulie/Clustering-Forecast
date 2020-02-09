# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 15:19:19 2017

@author: sonja
"""
# import libraries

import pickle
import numpy as np
from classes.Composites import Composites
from classes.Clusters import Clusters
import sys
# in composites must be
#   1.) files loaded
#   2.) numbers from certain cluster calculated ==> mean
#   3.) bootrap method for 90 and 95% ==> two different hastags?


def main():
    # variables
    inifile = sys.argv[1]
    logger.debug(f"inifile: {inifile}")
    method_name = "ward"
    for k in range(6,7):
        with open(rf"/home/sonja/Documents/Composites/output/{sys.argv[3]}/model-1/Cluster/"
                  rf"ward_Cluster_{k}/files/timeSeries_ward_{k}_f.txt", "rb") as input_file:
            f = np.array(pickle.load(input_file),dtype=int)
        f = np.array(f.reshape((2, int(f.shape[0]/2)))[1],dtype=int)
        if int(sys.argv[4]) > 0:
            flength = int(sys.argv[4])
            f = f[:flength]
            logger.debug(f"check size of f: {f.shape} == {flength}")
            print(f"f: {f.shape} == {flength}")
        composites = Composites(inifile)
        composites.get_composites_data_1d(f, k, method_name, sys.argv[3])
        composites.plot_composites(k, float(sys.argv[5]))
        composites.save_composites(k)
        start_year = 1920
        composites.plot_years(sys.argv[3], method_name, k, start_year, f)

    # # Load temperature




if __name__ == '__main__':
    import logging.config
    from config_dict import config

    logger = logging.getLogger(__name__)
    logging.config.dictConfig(config)
    logger.info("Start clustering program")
    main()
