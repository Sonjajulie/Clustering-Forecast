#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 16:11:43 2019

@author: sonja
"""

import os
import matplotlib as mpl
# if I do not use this trick, no plots will be saved!
if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import logging
from scipy.cluster.hierarchy import dendrogram
from logging import config
from pathlib import Path
from classes.Clusters import Clusters
sns.set()


def _fancy_dendrogram(*args, **kwargs):
    """ actual plot for nicer dendrogram"""
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata


class Predictand(Clusters):
    """ Class to analyze Predictand """

    def __init__(self, inifile_in: str, output_path: str, output_label: str, cl_config: dict):
        """
        Initialize Clusters--> read file(s) using ini-file
        apply mask, if necessary
        extract data such as time and spatial data
        create 1d array
        :param inifile_in: file for initialization of variable
        :param output_path: path, where output should be saved
        :param output_label: label for substring of output directory
        :param cl_config: dictionary, where all information of logger is stored from classes/config
        """
        super().__init__(inifile_in, output_path, output_label, cl_config)
        logging.config.dictConfig(cl_config)
        self.logger = logging.getLogger(__name__)
        self.logger.info('Read ini-file')
        self.sigma_var = None
        self.varmean = None
        self.varAnom = None

    def _set_clusters_reshape(self):
        """ reshape 1d clusterst to 2d clusters"""
        self.clusters_reshape = np.zeros(
            (self.k, self.dict_predict[self.var].shape[1], self.dict_predict[self.var].shape[2]))
        for i in range(int(self.k)):
            self.clusters_reshape[i] = np.reshape(self.clusters[i],
                                                  (self.dict_predict[self.var].shape[1],
                                                   self.dict_predict[self.var].shape[2]))

    def calculate_clusters_from_test_data(self, train_data: dict, method_name: str, k: int):
        """
        calculate clusters for predictand variable
        :param train_data: cluster data which should be used to calculate clusters
        :param method_name: name of the method used for clustering
        :param k: number of clusters
        """
        self.logger.info('Calculate clusters')
        # do I really need this?
        # self.dict_pred_1D = train_data
        # for key in train_data.keys():
        #     self._calculate_standardized_predictand(key)

        self.dict_standardized_pred_1D = train_data
        self._set_method_name(method_name)
        self._set_k(k)
        self._set_linkage()
        self._set_f()
        self._cluster_frequency()
        self._set_clusters_1d()
        self._set_clusters_reshape()

        # calculate frequency
        # self._states_of_each_cluster()
        # set directories for plots and files
        self._set_directory_plots(f"{self.output_path}/output-{self.output_label}/{self.var}/Cluster/"
                                  f"{self.method_name}_Cluster_{self.k}/plots/")
        Path(self.directory_plots).mkdir(parents=True, exist_ok=True)
        self._set_directory_files(f"{self.output_path}/output-{self.output_label}/{self.var}/Cluster/"
                                  f"{self.method_name}_Cluster_{self.k}/files/")
        Path(self.directory_files).mkdir(parents=True, exist_ok=True)

    def time_plot(self):
        """
        Plot variable for each model and each time point as mean
        """
        self._set_directory_plots(
            f"output-{self.output_label}/{self.var}/Cluster/{self.method_name}_Cluster_{self.k}/plots/")
        Path(self.directory_plots).mkdir(parents=True, exist_ok=True)
        self._set_directory_files(
            f"output-{self.output_label}/{self.var}/Cluster/{self.method_name}_Cluster_{self.k}/files/")
        Path(self.directory_files).mkdir(parents=True, exist_ok=True)
        time1 = self.dict_predict[self.var].coords["time"].values
        time = [i for i in range(len(time1))]
        vals = np.zeros((len(self.dict_predict[self.var].coords["time"].values)))
        for year in range(len(self.dict_standardized_pred_1D[self.var])):
            vals[year] = np.mean(self.dict_pred_1D[self.var][year])

        df = pd.DataFrame(index=time, columns=[f"cluster {i}" for i in range(self.k)], dtype=float)
        df_all = pd.DataFrame(vals, index=time, columns=[""], dtype=float)
        for t, f_value, val in zip(time, self.f, vals):
            df.at[t, f"cluster {f_value}"] = np.float(val)
        # plt.plot(time, vals, color='k', linewidth=1)
        # sns_plot = sns.scatterplot(data=df)  # , x="timepoint", y="signal", hue="event", style="event",
        # markers=True, dashes=False
        sns.lineplot(data=df_all, palette=sns.color_palette("mako_r", 1), linewidth=0.5, alpha=0.7)
        sns_plot = sns.scatterplot(data=df)  # , x="timepoint", y="signal", hue="event", style="event",
        # markers=True, dashes=False

        plt.xlabel(" model year DJF ")
        # Set y-axis label
        plt.ylabel("mean temperature")
        fig_sns = sns_plot.get_figure()
        self.logger.debug(f"Save in {self.directory_plots}/{self.var}_time_plot.pdf")
        fig_sns.savefig(f"{self.directory_plots}/{self.var}_time_plot.pdf")
        fig_sns.savefig(f"{self.directory_plots}/{self.var}_time_plot.png")
        plt.close()

    def _create_dataset_from_clusters(self):
        """
        Create dataset for clusters as netcdf using xarray library
        """
        self.logger.info("create dataset with clusters as variables")
        self.data_vars = {}
        self.lons, self.lats = np.meshgrid(self.dict_predict[self.var].coords['lon'].values,
                                           self.dict_predict[self.var].coords['lat'].values)
        self.data_vars = {f"cluster_{self.var}": xr.DataArray(self.clusters_reshape,
                                                              coords={
                                                                  'lon': self.dict_predict[self.var].coords[
                                                                      'lon'].values,
                                                                  'lat': self.dict_predict[self.var].coords[
                                                                      'lat'].values},
                                                              attrs={'long_name': self.dict_predict[self.var].attrs[
                                                                  "long_name"],
                                                                     'units': self.dict_predict[self.var].attrs[
                                                                         "units"]},
                                                              dims=['c', 'lat', 'lon'])}

    def calculate_clusters_year(self, method_name: str, k: int, year: int):
        """
        calculate clusters for predictand variable
        :param method_name: name of method used to calculate clusters (e.g. ward)\
        :param k: cluster number
        :param year: year for which we would like to forecast_nn the predictand
        """
        self.logger.info('Calculate clusters')
        self.remove_year_and_calc_anomalies(year)
        self._set_method_name(method_name)
        self._set_k(k)
        self._set_linkage()
        self._set_f()
        self._cluster_frequency()
        self._set_clusters_1d()
        self._set_clusters_reshape()
        # set directories for plots and files
        self._set_directory_plots(f"output/{self.var}/Cluster/{self.method_name}_Cluster_{self.k}/plots/")
        Path(self.directory_plots).mkdir(parents=True, exist_ok=True)
        self._set_directory_files(f"output/{self.var}/Cluster/{self.method_name}_Cluster_{self.k}/files/")
        Path(self.directory_files).mkdir(parents=True, exist_ok=True)
        # calculate frequency
        # self._states_of_each_cluster()

    def remove_year_and_calc_anomalies(self, year: int):
        """
        remove selected year
        :param year: year for which we would like to forecast_nn the predictand and remove in this case from
        the training data set
        """
        self.dict_pred_1D[f"{self.var}_minus_year_1D"] = np.delete(self.dict_pred_1D[f"{self.var}"], year -
                                                                   int(self.time_start_file), axis=0)
        self.varmean = np.mean(self.dict_pred_1D[f"{self.var}_minus_year_1D"], axis=0)
        self.varAnom = self.dict_pred_1D[f"{self.var}_minus_year_1D"] - self.varmean
        # # divided by grid (1d-Array) and years - 1 (the year which we would like to forecast_nn)
        # # standardize
        self.sigma_var = np.sum(self.varAnom * self.varAnom) / (self.varAnom.shape[0] * self.varAnom.shape[1])
        self.dict_standardized_pred_1D[self.var] = self.varAnom  # / self.sigma_var
