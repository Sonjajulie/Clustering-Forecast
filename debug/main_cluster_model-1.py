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
    var = sys.argv[1]
    pred_clusters = Clusters(f"ini/clusters_America_{var}1.ini")
    # pred_clusters = Clusters(f"ini/clusters_America_TS.ini")
    method_name = 'ward'


    for k in [3, 4, 5, 6, 7, 8, 9]:
        pred_clusters.calculate_clusters( method_name, k)
        pred_clusters.plot_clusters_and_time_series()
        pred_clusters.plot_elbow_plot()
        pred_clusters.plot_fancy_dendrogram()
        pred_clusters.save_clusters()
        pred_clusters.plot_years()
    # pred_clusters = Clusters(f"ini/clusters_America_prec_t.ini")
    # method_name = 'ward'
    # for k in [3, 4, 5, 6, 7, 8, 9]:
    #     pred_clusters.calculate_clusters( method_name, k)
    #     pred_clusters.plot_clusters_and_time_series()
    #     pred_clusters.plot_elbow_plot()
    #     pred_clusters.plot_fancy_dendrogram()
    #     pred_clusters.save_clusters()
    # for k in [6]:
    #     with open(rf"/home/sonja/Documents/Composites/output/{sys.argv[3]}/Cluster/"
    #               rf"ward_Cluster_{k}/files/timeSeries_ward_{k}_f.txt", "rb") as input_file:
    #         f = np.array(pickle.load(input_file),dtype=int)
    #     f = np.array(f.reshape((2, int(f.shape[0]/2)))[1],dtype=int)
    #     if int(sys.argv[4]) > 0:
    #         flength = int(sys.argv[4])
    #         f = f[:flength]
    #         logger.debug(f"check size of f: {f.shape} == {flength}")
    #         print(f"f: {f.shape} == {flength}")
    #     composites = Composites(inifile)
    #     composites.get_composites_data_1d(f, k, method_name, sys.argv[3])
    #     composites.plot_composites(k, float(sys.argv[5]))
    #     composites.save_composites(k)

    # # Load temperature




if __name__ == '__main__':
    import logging.config
    from config_dict import config

    logger = logging.getLogger(__name__)
    logging.config.dictConfig(config)
    logger.info("Start clustering program")
    main()
