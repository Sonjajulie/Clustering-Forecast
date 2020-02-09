# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 15:19:19 2017

@author: sonja
"""
# import libraries

# import pickle
# import numpy as np
# from classes.Composites import Composites
from classes.Clusters import Clusters
import sys


# in composites must be
#   1.) files loaded
#   2.) numbers from certain cluster calculated ==> mean
#   3.) bootrap method for 90 and 95% ==> two different hastags?


def main():
    # variables
    var = sys.argv[1]
    output_label = sys.arv[3]
    pred_clusters = Clusters(f"ini/clusters_America_{var}.ini", output_label)
    # pred_clusters = Clusters(f"ini/clusters_America_TS.ini")
    method_name = 'ward'

    for k in [3, 4, 5, 6, 7, 8, 9]:
        pred_clusters.calculate_clusters(method_name, k)
        pred_clusters.plot_clusters_and_time_series()
        pred_clusters.plot_elbow_plot()
        pred_clusters.plot_fancy_dendrogram()
        pred_clusters.save_clusters()
        # pred_clusters.plot_years()


if __name__ == '__main__':
    import logging.config
    from config_dict import config

    logger = logging.getLogger(__name__)
    logging.config.dictConfig(config)
    logger.info("Start clustering program")
    main()
