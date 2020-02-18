# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 15:19:19 2017

@author: sonja
"""
# import libraries

import pickle
import numpy as np
from classes.Composites import Composites
# from classes.Clusters import Clusters
from classes.ClusteringParser import ClusteringParser
from classes.Config import Config
# in composites must be
#   1.) files loaded
#   2.) numbers from certain cluster calculated ==> mean
#   3.) bootrap method for 90 and 95% ==> two different hastags?
# /home/sonja/Documents/Composites/output/prec_t/Cluster/ward_Cluster_5/files/
# /home/sonja/Documents/Composites/output/prec_t/Cluster/ward_Cluster_5/files/timeSeries_ward_5_f.txt
# /home/sonja/Documents/Clustering-Forecast/output/TS/Cluster/ward_Cluster_5/files/timeSeries_ward_5_f.txt

def main(cl_parser: ClusteringParser, cl_config: dict):
    # variables
    predictand_var = cl_parser.arguments['predictand']
    output_label = cl_parser.arguments['outputlabel']
    inifile = cl_parser.arguments['inifile']
    len_numbers = cl_parser.arguments['numbers']
    percentage_boost = cl_parser.arguments['percentage']
    # pred_clusters = Clusters(inifile, output_label, cl_config)
    logger.debug(f"inifile: {inifile}")
    method_name = "ward"
    for k in [4,5,6,7,8]:
    # ~ for k in [3]:
        with open(rf"output-{output_label}/{predictand_var}/Cluster/ward_Cluster_{k}/files/timeSeries_ward_{k}_f.txt", "rb") as input_file:
            f = np.array(pickle.load(input_file), dtype=int)
        f = np.array(f.reshape((2, int(f.shape[0]/2)))[1], dtype=int)
        if int(len_numbers) > 0:
            flength = int(len_numbers)
            f = f[:flength]
            logger.debug(f"check size of f: {f.shape} == {flength}")
            print(f"f: {f.shape} == {flength}")
        composites = Composites(inifile,output_label, cl_config)
        composites.get_composites_data_1d(f, k, method_name, predictand_var)
        composites.plot_composites(k, float(percentage_boost))
        composites.save_composites(k)
        composites.time_plot(predictand_var, method_name, k, f)
        start_year = 1919
        # ~ if k == 5:
             # ~ composites.plot_years(predictand_var, method_name, k, f)
    # # Load temperature


if __name__ == '__main__':
    import logging.config
    parser = ClusteringParser()
    config = Config(parser.arguments['logfile'])
    logger = logging.getLogger(__name__)

    # read config log file from classes.Config
    logging.config.dictConfig(config.config_dict)
    logger.info("Start composites program")
    main(parser, config.config_dict)
