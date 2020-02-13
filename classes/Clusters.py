#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 16:11:43 2019

@author: sonja
"""
import os
import matplotlib as mpl
if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xarray as xr
import configparser
import pandas as pd
import logging
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import os
import cftime
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from logging import config
from pathlib import Path
import pickle

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


class Clusters:
    """ Class to analyze Predictand """

    def __init__(self, inifile_in: str, output_label: str, cl_config: dict):
        """
        Initialize Clusters--> read file(s) using ini-file
        apply mask, if necessary
        extract data such as time and spatial data
        create 1d array
        :param inifile_in: file for initialization of variable
        :param output_label: label for substring of output directory
        :param cl_config: dictionary, where all information of logger is stored from classes/config
        """
        logging.config.dictConfig(cl_config)
        self.logger = logging.getLogger(__name__)
        self.logger.info('Read ini-file')
        self.inifile = inifile_in
        # https://docs.python.org/3/library/configparser.html
        self.config = configparser.ConfigParser()
        self.config.read(self.inifile)
        self.dict_standardized_pred_rmse = {}
        self.output_label = output_label

        self.logger.debug(f"Sections: {[prec for prec in self.config.sections() if 'PRED:' in prec]}")
        self.sec = [prec for prec in self.config.sections() if 'PRED:' in prec][0]
        self.var = self.config[self.sec]["var"]
        # all precursors in the ini-file should be assigned to the dictionary
        self._initialize_attributes()
        # check whether multiple files must be read
        if self.config.has_section(self.sec):
            if "nc" in self.config[self.sec]["filepath"]:
                self.dict_predict[self.var] = xr.open_dataset(self.config[self.sec]["filepath"])[self.var]
                self._set_extent_cluster(self.var)
                if self.config.has_option(self.sec, "mask"):
                    self._get_and_apply_mask(self.var)
                self._transform_to_1d_and_remove_nans(self.var)
                self._calculate_standardized_predictand(self.var)
            else:
                # since models have the same time and variable, an artificial time must
                # be created with time = model*time
                # assume that all files in directory have to be read
                self.path = f"{self.config[self.sec]['filepath']}/"
                self.list_of_files = [os.path.join(self.path, item) for item in os.listdir(self.path)
                                      if os.path.isfile(os.path.join(self.path, item))]
                # ,decode_times=False,combine='by_coords'   .load()
                self.list_of_files = sorted(self.list_of_files)
                length_files = len(self.list_of_files)
                for i in range(length_files):
                    self.logger.debug(f"file {i}: {self.list_of_files[i]}")
                    self.dict_predict[f"{self.var}_{i}"] = xr.open_dataset(self.list_of_files[i])[self.var]
                    self._set_extent_cluster(f"{self.var}_{i}")
                    if self.config.has_option(self.sec, "mask"):
                        self._get_and_apply_mask(f"{self.var}_{i}")
                    self._transform_to_1d_and_remove_nans(f"{self.var}_{i}")
                    self._calculate_standardized_predictand(f"{self.var}_{i}")
                # What dimenson has self.dict_precursors.values() ? dim = [model,time,lons,lats]?
                # has to be changed to dim = [time*models,lons,lats]!
                # 97 x 59# make dataset instead of array!!self.dict_precursors[f"{self.var}_{0}"].coords['time'].values
                # list_time_model = [f"{i + 1}: {j}" for i in range(len(self.list_of_files))
                #                    for j in self.dict_predict[f"{self.var}_{i}"].coords['time'].values]
                list_time_model = [f"model {i + 1}, date: {j.year}-{j.month}-{j.day}" for i in range(length_files)
                                   for j in self.dict_predict[f"{self.var}_{i}"].coords['time'].values]
                self.dict_predict = {self.var: xr.DataArray(np.concatenate(list(self.dict_predict.values())),
                                                            coords={'time': list_time_model,
                                                                    'lon': self.dict_predict[f"{self.var}_{0}"]
                                                            .coords['lon'].values, 'lat': self.
                                                            dict_predict[f"{self.var}_{0}"]
                                                            .coords['lat'].values}, dims=['time', 'lat', 'lon'])}
                self.dict_pred_1D = {self.var: np.concatenate(list(self.dict_pred_1D.values()))}
                self.dict_standardized_pred_1D = {self.var: np.concatenate(list(self.dict_standardized_pred_1D
                                                                                .values()))}
        else:
            self.logger.error(f"Option {self.var} and/or section {self.sec} not found in ini-file {self.inifile}")
            raise NameError()

    def _set_extent_cluster(self, label):
        """
         Get Longitudes and Latitudes, check whether latitudes go form -90 to 90 or from 90 to -90,
        if the letter, reverse order
        :param label: name of variable. If one uses a cluster the variable name is the same for different
        model initialization and therefore I renamed the variable name.
        """
        #  first read whether unit is lat or latitude or something else
        #  https://stackoverflow.com/questions/29135885/netcdf4-extract-for-subset-of-lat-lon
        self._get_dim_boundaries(label)
        if all(x in self.dict_predict[label].coords for x in ['latitude', 'longitude']):
            self.ll = self.dict_predict[label].coords['latitude'].values
            if self.ll[1] - self.ll[0] < 0:
                self.dict_predict[label] = self.dict_predict[label] \
                    .reindex(latitude=self.dict_predict[self.dict_predict].latitude[::-1])
            self.dict_predict[label] = self.dict_predict[label].sel(latitude=slice(*self.lat_bnds),
                                                                    longitude=slice(*self.lon_bnds))
        elif all(x in self.dict_predict[label].coords for x in ['lat', 'lon']):
            self.ll = self.dict_predict[label].coords['lat'].values
            if self.ll[1] - self.ll[0] < 0:
                self.dict_predict[label] = self.dict_predict[self.var] \
                    .reindex(lat=self.dict_predict[label].lat[::-1])
            self.dict_predict[label] = self.dict_predict[label].sel(lat=slice(*self.lat_bnds),
                                                                    lon=slice(*self.lon_bnds))
        # https://stackoverflow.com/questions/29135885/netcdf4-extract-for-subset-of-lat-lon
        self.dict_predict[label] = self.dict_predict[label].sel(lat=slice(*self.lat_bnds),
                                                                lon=slice(*self.lon_bnds))
        self.res = self.ll[1] - self.ll[0]

    def _initialize_attributes(self):
        """
        Initialize members of class with None or empty object
        """
        self.clusters = None
        self.dict_clusters_d = {}
        self.clustersnumber_save = None
        self.data_vars = {}
        self.dict_mask = {}
        self.dict_predict = {}
        self.dict_pred_1D = {}
        self.dict_standardized_pred_1D = {}
        self.dict_standardized_pred_rmse_reshape = {}
        self.f = None
        self.k = None
        self.method_name = None
        self.pin_arrays = None
        self.Z = None
        self.Z_dict = None

    def _get_dim_boundaries(self, label: str):
        """
        get dimensions of latitudes and longitudes from ini-file
        :param label: name of variable. If one uses a cluster the variable name is the same for different
        model initialization and therefore I renamed the variable name.
        """
        self.lat_min, self.lat_max, self.lon_min, self.lon_max = map(float,
                                                                     self.config[self.sec]["coords"]
                                                                     .split(','))
        self.lat_bnds, self.lon_bnds = [self.lat_min, self.lat_max], [self.lon_min, self.lon_max]
        # for later calculation save when time of predictand variable start
        self.time_start_file = self.dict_predict[label].time.values[0]
        if isinstance(self.time_start_file, cftime.DatetimeNoLeap):
            self.time_start_file = self.dict_predict[label].time.values[0].year
        else:
            self.time_start_file = pd.to_datetime(self.dict_predict[label].time.values[0]).year

    def _get_and_apply_mask(self, label: str):
        """
        apply mask to input-file
        :param label: name of variable. If one uses a cluster the variable name is the same for different
        model initialization and therefore I renamed the variable name.
        """
        self.dict_mask[self.config[self.sec]["var"]] = np.loadtxt(self.config[self.sec]["mask"])
        self.dict_predict[label] = self.dict_predict[label] \
            .where(self.dict_mask[self.config[self.sec]["var"]] == 0, 0)

    def _transform_to_1d_and_remove_nans(self, label: str):
        """
        transfrom array and set values 0, where no data is found as well reshape to 1D
        apply mask to input-file
        :param label: name of variable. If one uses a cluster the variable name is the same for different
        model initialization and therefore I renamed the variable name.
        """
        # self.dict_pred_1D[var] = 1
        self.logger.info('Reshape to 1D array and remove nans')
        self.dict_pred_1D[label] = np.reshape(np.array(self.dict_predict[label]),
                                              [np.array(self.dict_predict[label])
                                              .shape[0], -1])
        self.dict_pred_1D[label][self.dict_pred_1D[label] != self.dict_pred_1D[label]] = 0

    def _set_method_name(self, method_name: str):
        """
        set method
        method_name: method name for clustering
        """
        self.method_name = method_name

    def _set_k(self, k: int):
        """
        set k
        :param k: number of clusters
        """
        self.k = k

    def _set_linkage(self):
        """ set linkage according to method"""
        self.Z = linkage(self.dict_standardized_pred_1D[self.var], self.method_name)

    def _set_f(self):
        """ get f (flat clusters) from cluster method using linkage, k and maxclust criterium"""
        self.f = fcluster(self.Z, self.k, criterion='maxclust')
        # order according to frequency
        self.f_bins = np.bincount(self.f - 1)
        self.f_index = np.argsort(self.f_bins)[::-1]  # self.k - 1 -
        self.f_final = np.zeros(len(self.f_index))
        for index, i in enumerate(self.f_index):
            self.f_final[i] = index
        for nr, f_el in enumerate(self.f - 1):
            self.f[nr] = self.f_final[f_el]

    def _set_directory_plots(self, directory: str):
        """
        set directories for plots
        :param directory: directory for images
        """
        self.directory_plots = directory

    def _set_directory_files(self, directory):
        """
        set directory for files
        :param directory: directory for images
        """
        self.directory_files = directory

    def _set_clusters_1d(self):
        """ set 1d clusters from f"""
        self.clusters = np.zeros((self.k, self.dict_standardized_pred_1D[self.var].shape[1]))
        for cluster_number in range(self.k):
            self.clusters[cluster_number] = \
                np.mean(self.dict_standardized_pred_1D[self.var][self.f == cluster_number], axis=0)

    def _set_clusters_reshape(self):
        """ reshape 1d clusterst to 2d clusters"""
        self.clusters_reshape = np.zeros(
            (self.k, self.dict_predict[self.var].shape[1], self.dict_predict[self.var].shape[2]))
        for i in range(int(self.k)):
            self.clusters_reshape[i] = np.reshape(self.clusters[i],
                                                  (self.dict_predict[self.var].shape[1],
                                                   self.dict_predict[self.var].shape[2]))

    def _calculate_standardized_predictand(self, label: str):
        """
        calculate the standardized predictand
        :param label: name of variable. If one uses a cluster the variable name is the same for different
        model initialization and therefore I renamed the variable name.
        """
        self.varmean = np.mean(self.dict_pred_1D[label], axis=0)
        self.varAnom = self.dict_pred_1D[label] - self.varmean
        # divided by grid (1d-Array) and years - 1 (the year which we would like to forecast)
        # standardize
        if self.output_label == "standardized":
            self.sigma_var = np.sum(self.varAnom * self.varAnom) / (self.varAnom.shape[0] * self.varAnom.shape[1])
            self.dict_standardized_pred_1D[label] = self.varAnom / self.sigma_var
        else:
            self.dict_standardized_pred_1D[label] = self.varAnom

    def calculate_clusters(self, method_name, k):
        """
        calculate clusters for predictand variable
        :param k: cluster number
        :param method_name: name of method used to calculate clusters (e.g. ward)
        """
        self.logger.info('Calculate clusters')
        # self._calculate_standardized_predictand()
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
        self._set_directory_plots(f"output-{self.output_label}/{self.var}/Cluster/"
                                  f"{self.method_name}_Cluster_{self.k}/plots/")
        Path(self.directory_plots).mkdir(parents=True, exist_ok=True)
        self._set_directory_files(f"output-{self.output_label}/{self.var}/Cluster/"
                                  f"{self.method_name}_Cluster_{self.k}/files/")
        Path(self.directory_files).mkdir(parents=True, exist_ok=True)

    def _cluster_frequency(self):
        """
        Calculate cluster frequency from f
        """
        self.cluster_frequency = np.bincount(self.f)
        self.cluster_frequency = np.divide(self.cluster_frequency, float(self.dict_predict[self.var].shape[0]))*100
        for j in range(self.k):
            self.logger.info(f"Cluster{j}, {self.cluster_frequency[j]:.3f}")
        self.cluster_frequency_sort = np.argsort(np.argsort(self.cluster_frequency))
        self.cluster_bin = np.bincount(self.f)

    def _states_of_each_cluster(self):
        """
        Linkage only states from certain clusters -- Do I need this?
        """
        for step, i in enumerate(self.f):
            self.dict_clusters_d[f"clusters_{i}"] = self.dict_standardized_pred_1D[self.var][step]
        # Clustering
        i = 0
        for key, item in self.dict_clusters_d.items():
            self.Z_dict[key] = linkage(self.dict_clusters_d[key], self.method_name)
            self.logger.debug(f"self.Z_dict[{key}], {self.cluster_bin[i]}")
            self.logger.debug(self.Z_dict[key][-2:])
            i = i + 1

    def plot_years(self):
        """
        Plot each year of variable
        """
        for i in range(self.k):
            self._set_directory_plots(f"output/{self.var}/Cluster/{self.method_name}_Cluster_{self.k}/years/Cluster_{i}/plots/")
            Path(self.directory_plots).mkdir(parents=True, exist_ok=True)
            self._set_directory_files(f"output/{self.var}/Cluster/{self.method_name}_Cluster_{self.k}/years/Cluster_{self.f[year]}/files/")
            Path(self.directory_files).mkdir(parents=True, exist_ok=True)
        for year in range(len(self.dict_standardized_pred_1D[self.var])):
            var_reshape = np.reshape(self.dict_pred_1D[self.var][year], (self.dict_predict[self.var].shape[1],
                                                                         self.dict_predict[self.var].shape[2]))
            self.data_vars[f"{self.var}"] = xr.DataArray(var_reshape, dims=('lat', 'lon'))
            # (( 'lat','lon'), self.clusters_reshape[i])
            self.ds = xr.Dataset(self.data_vars, coords={'lon': self.dict_predict[self.var].coords["lon"].values,
                                                         'lat': self.dict_predict[self.var].coords["lat"].values})
            # n_cols = max(n, 1)
            map_proj = ccrs.PlateCarree()
            self.ds_arrays = self.ds[f"{self.var}"]
            ax = plt.axes(projection=map_proj)
            self.ds[f"{self.var}"].plot(
                ax=ax,
                transform=ccrs.PlateCarree(),  # the data's projection
                cmap=plt.cm.get_cmap('seismic', 31),
                cbar_kwargs={'shrink': 0.8},
            )

            ax.add_feature(cfeature.BORDERS, linewidth=0.1)
            ax.coastlines()
            ax.set_extent([self.lon_min, self.lon_max, self.lat_min, (2 * self.lat_max - 90)])
            # Without this aspect attributes the maps will look chaotic and the
            # "extent" attribute above will be ignored
            # ax.set_aspect("equal")
            ax.set_aspect(aspect=self.ds.dims["lon"] / self.ds.dims["lat"])
            ax.set_title(f"{self.var}, {self.dict_predict[self.var].time.values[year]}, cluster: {self.f[year]}",
                         fontsize=10)
            # ax.contourf(self.lons, self.lats, significance, levels=[0., 0.05, 0.5, 0.95, 1],
            #             hatches=["/////", ".....", None, None, None], colors='none', transform=ccrs.PlateCarree())
            # # hatches=["/////", ".....", ",,,,,", "/////", "....."], colors='none', transform=ccrs.PlateCarree())
            self.logger.debug(f"Save in {self.directory_plots}/{self.var}_{self.dict_predict[self.var].time.values[year]}"
                         f".pdf")
            plt.savefig(f"{self.directory_plots}/Cluster_{self.f[year]}/{year:03d}_{self.var}_{self.dict_predict[self.var].time.values[year]}"
                        f".pdf")
            plt.savefig(f"{self.directory_plots}/Cluster_{self.f[year]}/{year:03d}_{self.dict_predict[self.var].time.values[year]}.png")
            plt.close()

    def time_plot(self):
        """
        Plot variable for each model and each time point as mean
        """
        self._set_directory_plots(f"output/{self.var}/Cluster/{self.method_name}_Cluster_{self.k}/plots/")
        Path(self.directory_plots).mkdir(parents=True, exist_ok=True)
        self._set_directory_files(f"output/{self.var}/Cluster/{self.method_name}_Cluster_{self.k}/files/")
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

    def _save_clusters_plot(self):
        """ Save clusters into one plot using xarray library"""
        # self._set_clustersnumber_save()
        self._create_dataset_from_clusters()

        n_cols = min(self.k, 4)
        n_cols = np.ceil(self.k / n_cols)
        # n_cols = max(n, 1)
        map_proj = ccrs.PlateCarree()
        self.ds_arrays = self.ds.to_array()

        p = self.ds_arrays.plot(
            transform=ccrs.PlateCarree(),  # the data's projection
            col="variable",
            cmap=plt.cm.get_cmap('seismic', 31),
            size=2,
            col_wrap=int(n_cols),  # multiplot settings
            aspect=self.ds.dims["lon"] / self.ds.dims["lat"],  # for a sensible figsize
            subplot_kws={"projection": map_proj},  # the plot's projection
        )

        # We have to set the map's options on all four axes
        for ip, ax in enumerate(p.axes.flat):
            if ip < self.k:
                ax.add_feature(cfeature.BORDERS, linewidth=0.1)
                ax.coastlines()
                ax.set_extent([self.lon_min, self.lon_max, self.lat_min, (2 * self.lat_max - 90)])
                # Without this aspect attributes the maps will look chaotic and the
                # "extent" attribute above will be ignored
                # ax.set_aspect("equal")
                title = self.cluster_frequency[ip]
                ax.set_title(f"Cluster {ip} - {title:4.2f} %", fontsize=10)

        self.logger.debug(f"Save in {self.directory_plots}/clusters.pdf")
        plt.savefig(f"{self.directory_plots}/clusters.pdf")
        plt.close()

    def plot_clusters_and_time_series(self):
        """Plot clusters"""
        # self._save_separate_clusters()
        self._save_clusters_plot()
        self._save_time_series()
        self._save_timeseries_f()

    def _save_time_series(self):
        """ Plot and save time series as  pdf and txt file """
        # get time series
        time_data = [self.time_start_file + float(i)
                     for i in range(int(self.dict_standardized_pred_1D[self.var].shape[0]))]
        clust_data = [self.f[i]
                      for i in range(int(self.dict_standardized_pred_1D[self.var].shape[0]))]
        # plot time series
        fig5 = plt.figure()
        # plt.ylim([0,self.k])
        plt.xlabel('year')
        plt.ylabel('Cluster number')
        yint = range(0, self.k)
        plt.yticks(yint)
        sns.set_style()
        plt.plot(time_data, clust_data, 'o', linestyle='-')
        plt.close()
        fig5.savefig(f"{self.directory_plots}timeSeries_{self.method_name}_{self.k}.pdf")
        # use pickle since savetxt gives warning
        pickle.dump(np.hstack((time_data, clust_data)).astype(float),
                    open(f"{self.directory_files}/timeSeries_{self.method_name}_{self.k}.txt", "wb"))
        # np.savetxt(f"{self.directory_files}/timeSeries_{self.method_name}_{self.k}.txt", (time_data, clust_data),
        #            fmt='%4.2f', delimiter=".")  # x,y,z equal sized 1D arrays

    def _save_timeseries_f(self):
        """
        Save time series of f in pickle
        """
        time_data = [self.time_start_file + float(i)
                     for i in range(int(self.dict_standardized_pred_1D[self.var].shape[0]))]
        pickle.dump(np.hstack((time_data, self.f)).astype(float),
                    open(f"{self.directory_files}/timeSeries_{self.method_name}_{self.k}_f.txt", "wb"))

    def plot_elbow_plot(self):
        """
        Plot and save elbow plot as well as 2. derivative of elbow plot
        """
        fig3 = plt.figure()
        last = self.Z[-10:, 2]
        last_reverse = last[::-1]
        cluster_number_array = np.arange(1, len(last) + 1)
        plt.xlabel('Cluster number')
        plt.ylabel('Distance')
        plt.plot(cluster_number_array, last_reverse, marker='o', color='navy')
        # 2nd derivative of the distances
        acceleration = np.diff(last, 2)
        acceleration_rev = acceleration[::-1]
        plt.plot(cluster_number_array[:-2] + 1, acceleration_rev, marker='o')
        fig3.savefig(f"{self.directory_plots}/Elbow_method_{self.method_name}_Curvature.pdf")
        plt.close()
        pickle.dump(np.hstack((cluster_number_array, last)).astype(float),
                    open(f"{self.directory_files}/timeSeries_{self.method_name}_{self.k}.txt", "wb"))
        pickle.dump(np.hstack((cluster_number_array[:-2] + 1, acceleration_rev)).astype(float),
                    open(f"{self.directory_files}/timeSeries_{self.method_name}_{self.k}.txt", "wb"))

    def plot_fancy_dendrogram(self):
        """
        Plot a nicer dendrogram than the default dendrogram
        """
        fig2 = plt.figure()
        _fancy_dendrogram(
            self.Z,
            truncate_mode='lastp',
            p=12,
            leaf_rotation=90.,
            leaf_font_size=12.,
            show_contracted=True,
            annotate_above=10,  # useful in small plots so annotations don't overlap
            max_d=28,
        )
        fig2.savefig(f"{self.directory_plots}/Fancy_Dendrogram_{self.method_name}.png")
        plt.close()

    def plot_dendrogram(self):
        """
        Plot and save normal dendrogram
        """
        fig1 = plt.figure()
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('sample index')
        plt.ylabel('distance')
        dendrogram(
            self.Z,
            truncate_mode='lastp',  # show only the last p merged clusters
            p=12,  # show only the last p merged clusters
            show_leaf_counts=False,  # otherwise numbers in brackets are counts
            leaf_rotation=90.,
            leaf_font_size=12,
            show_contracted=True,  # to get a distribution impression in truncated branches
        )
        fig1.savefig(f"{self.directory_plots}/Dendrogram_{self.method_name}.pdf")
        plt.close()

    def _create_dataset_from_clusters(self):
        """
        Create dataset for clusters as netcdf using xarray library
        """
        self.logger.info("create dataset with clusters as variables")
        self.data_vars = {}
        for i in range(self.k):
            # cs = int(self.k) - int(self.clustersnumber_save[i]) - 1
            self.data_vars[f"cluster_{self.var}_{i}"] = xr.DataArray(self.clusters_reshape[i], dims=('lat', 'lon'))
            # (( 'lat','lon'), self.clusters_reshape[i])
        self.ds = xr.Dataset(self.data_vars, coords={'lon': self.dict_predict[self.var].coords["lon"].values,
                                                     'lat': self.dict_predict[self.var].coords["lat"].values})

    def save_clusters(self):
        """
        Save clusters using xarray
        """
        self.logger.info("Save clusters as netcdf")
        self._create_dataset_from_clusters()
        self.ds.to_netcdf(f"{self.directory_files}/clusters.nc")

    def calculate_rms(self):
        """
        Save root mean square error plot
        """
        self._calculate_standardized_predictand(self.var)
        self.dict_standardized_pred_rmse = np.zeros(self.dict_standardized_pred_1D[self.var].shape[1])
        for ci in range(self.dict_standardized_pred_1D[self.var].shape[1]):
            time_series = self.dict_standardized_pred_1D[self.var][:, ci]
            time_series_squared = time_series**2
            time_series_squared_mean = np.mean(time_series_squared, axis=0)
            self.dict_standardized_pred_rmse[ci] = np.sqrt(time_series_squared_mean)
        self.dict_standardized_pred_rmse_reshape[self.var] = np.reshape(self.dict_standardized_pred_rmse,
                                                                        (self.dict_predict[self.var].shape[1],
                                                                         self.dict_predict[self.var].shape[2]))
        self.data_vars["rms"] = xr.DataArray((self.dict_standardized_pred_rmse_reshape[self.var]), dims=('lat', 'lon'))
        # (( 'lat','lon'), self.clusters_reshape[i])
        self.ds = xr.Dataset(self.data_vars, coords={'lon': self.dict_predict[self.var].coords["lon"].values,
                                                     'lat': self.dict_predict[self.var].coords["lat"].values})
        self.ds_arrays = self.ds.to_array()
        p = self.ds_arrays.plot(
            cmap=plt.cm.get_cmap('seismic', 31),
            size=5,
            aspect=self.ds.dims["lon"] / self.ds.dims["lat"],  # for a sensible figsize
        )

        # We have to set the map's options on all four axes
        for ip, ax in enumerate(p.axes):
            p.axes.add_feature(cfeature.BORDERS, linewidth=0.1, alpha=0.5)
            ax.coastlines()
            ax.set_extent([self.lon_min, self.lon_max, self.lat_min, (2 * self.lat_max - 90)])
            # Without this aspect attributes the maps will look chaotic and the
            # "extent" attribute above will be ignored
            # ax.set_aspect("equal")
            ax.set_title(f"RMS", fontsize=10)
        # set directories for plots and files
        self._set_directory_plots(f"output/{self.var}/Cluster/")
        Path(self.directory_plots).mkdir(parents=True, exist_ok=True)
        self.logger.debug(f"Save in {self.directory_plots}/rms.pdf")
        plt.savefig(f"{self.directory_plots}/rms.pdf")
        plt.close()
