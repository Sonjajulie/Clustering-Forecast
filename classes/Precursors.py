#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 16:11:43 2019

@author: sonja
"""
# noinspection PyUnresolvedReferences
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xarray as xr
import configparser
import logging
import os
from logging import config
from pathlib import Path
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import pandas as pd

# seed random number generator
np.random.seed(0)
sns.set()


class Precursors:
    """Store and analyse possible precursors"""

    def __init__(self, inifile_in: str, output_label: str, cl_config: dict):
        """
        Store all parameters necessary for loading the netcdf file
        :param inifile_in: file for initialization of variable
        :param output_label: label for substring of output directory
        :param cl_config: dictionary, where all information of logger is stored from classes/config
        """
        logging.config.dictConfig(cl_config)
        self.logger = logging.getLogger(__name__)
        self.logger.info('Read ini-file')
        self.logger.info("Initialize class composites")
        self.ini = inifile_in
        self.config = configparser.ConfigParser()
        self.config.read(self.ini)
        self.percent_boot = None
        self.ds = None
        self.ds_arrays = None
        self.lons, self.lats = None, None
        self.lons1, self.lats1 = None, None
        self.end_n = None
        self.time_dim = None
        self.bootstrap_arrays = None
        self.output_label = output_label

        # all precursors in the ini-file should be assigned to the dictionary
        self._initialize_attributes()
        #  read precursors from ini-file
        self.precs_sections = [prec for prec in self.config.sections() if 'PREC:' in prec]
        for prec in self.precs_sections:
            # create dictionaries for each precursor again
            self.dict_prec_1D_var = {}
            self.dict_precursors_var = {}
            self.dict_standardized_precursors_var = {}
            if "nc" in self.config[prec]["filepath"]:
                # with xr.set_options(enable_cftimeindex=True):
                self.var = self.config[prec]["name"]
                self.dict_precursors_var[self.var] = xr.open_dataset(self.config[prec]["filepath"])[
                    self.config[prec]["var"]]
                self._set_area_composite(self.var, prec)
                if self.config.has_option(prec, "mask"):
                    self._get_and_apply_mask(self.config[prec]["name"], prec)
                self._transform_to_1d_and_remove_nans(self.config[prec]["name"])
                self._calculate_standardized_precursors(self.var)
                list_time_model = []
                for c_var in self.dict_precursors_var[self.config[prec]["name"]].coords['time'].values:
                    list_time_model.append(f"{c_var}")
                self.dict_precursors = {self.var: xr.DataArray(list(self.dict_precursors_var[self.var].values),
                                                               coords={'time': list_time_model,
                                                                       'lon': self.dict_precursors_var[f"{self.var}"]
                                                               .coords[self.label_lon].values,
                                                                       'lat': self.dict_precursors_var[f"{self.var}"]
                                                               .coords[self.label_lat].values},
                                                               dims=['time', self.label_lat, self.label_lon])}
                self.dict_standardized_precursors = {self.var: np.concatenate(list(self.dict_standardized_precursors_var
                                                                                   .values()))}
                self.dict_prec_1D = {self.var: np.concatenate(list(self.dict_prec_1D_var.values()))}
                if self.config.has_option(prec, "map_proj"):
                    self.map_proj_nr = int(self.config[prec]["map_proj"])
                self.fig_size = int(self.config[prec]["figsize"])
                self.aspect = int(self.config[prec]["aspect"])
                self.cross_size = float(self.config[prec]["hashsize"])
            else:
                # since models have the same time and variable, an artificial time must
                # be created with time = model*time
                # assume that all files in directory have to be read
                self.path = f"{self.config[prec]['filepath']}/"
                self.list_of_files = [os.path.join(self.path, item) for item in os.listdir(self.path)
                                      if os.path.isfile(os.path.join(self.path, item))]
                self.var = self.config[prec]["name"]
                # ,decode_times=False,combine='by_coords'   .load()
                self.list_of_files = sorted(self.list_of_files)
                length_files = len(self.list_of_files)
                for file in range(length_files):
                    self.logger.debug(f"file {file}: {self.list_of_files[file]}")
                    self.dict_precursors_var[f"{self.var}_{file}"] = \
                        xr.open_dataset(self.list_of_files[file])[self.config[prec]["var"]]
                    self._set_area_composite(f"{self.var}_{file}", prec)
                    if self.config.has_option(prec, "mask"):
                        self._get_and_apply_mask(f"{self.var}_{file}", prec)
                    self._transform_to_1d_and_remove_nans(f"{self.var}_{file}")
                    self._calculate_standardized_precursors(f"{self.var}_{file}")
                if self.config.has_option(prec, "map_proj"):
                    self.map_proj_nr = int(self.config[prec]["map_proj"])
                self.fig_size = int(self.config[prec]["figsize"])
                self.aspect = int(self.config[prec]["aspect"])
                self.cross_size = float(self.config[prec]["hashsize"])
                # change dimenson of precursor  to changed to dim = [time*models,lons,lats]!
                # list_time_model = [f"{i + 1}: {j}" for i in range(len(self.list_of_files))
                #                    for j in self.dict_predict[f"{self.var}_{i}"].coords['time'].values]
                list_time_model = [f"model {imodel + 1}, date: {jtime.year}-{jtime.month}-{jtime.day}" for imodel
                                   in range(length_files)
                                   for jtime in self.dict_precursors_var[f"{self.var}_{imodel}"].coords['time'].values]
                self.logger.debug(f"dims {self.label_lat}, {self.label_lon}")
                self.dict_precursors[self.var] = xr.DataArray(np.concatenate(list(self.dict_precursors_var.values())),
                                                              coords={'time': list_time_model,
                                                                      'lon': self.dict_precursors_var[
                                                                          f"{self.var}_{0}"].coords[self.label_lon]
                                                              .values,
                                                                      'lat': self.dict_precursors_var[
                                                                          f"{self.var}_{0}"].coords[self.label_lat]
                                                              .values},
                                                              dims=['time', self.label_lat, self.label_lon])
                self.dict_standardized_precursors[self.var] = np.concatenate(list(
                    self.dict_standardized_precursors_var.values()))
                self.dict_prec_1D[self.var] = np.concatenate(list(self.dict_prec_1D_var.values()))
                del self.dict_standardized_precursors_var
                del self.dict_precursors_var
                del self.dict_prec_1D_var

    def _initialize_attributes(self):
        """ initialize dictionaries for composites and plot properties"""
        self.dict_precursors = {}
        self.dict_mask = {}
        self.dict_prec_1D = {}
        self.dict_composites = {}
        self.dict_standardized_precursors = {}
        self.data_vars = {}
        self.composites_reshape = {}
        self.composites_significance_x = {}
        self.composites_significance_y = {}
        self.composites_significance = {}
        self.cluster_frequency = []
        self.cut_area = False
        self.aspect_ratio = 1
        self.map_proj_nr = 0

    def _set_area_composite(self, label: str, config_var: str):
        """
         Get Longitudes and Latitudes, check whether latitudes go form -90 to 90 or from 90 to -90,
        if the letter, reverse order
        :param label: name of variable. If one uses a cluster the variable name is the same for different
        model initialization and therefore I renamed the variable name.
        :param config_var: variable name of precursor section on config
        """
        self._get_dim_boundaries(config_var)
        # check name for latitude and longitude and cut area accordingly
        #  https://stackoverflow.com/questions/29135885/netcdf4-extract-for-subset-of-lat-lon
        if all(x in self.dict_precursors_var[label].coords for x in ['latitude', 'longitude']):
            self.label_lat, self.label_lon = 'latitude', 'longitude'
            # if latitude is sorted from positive to negative change order
            # otherwise sel-function will not work-> is there a better solution?
            self.ll = self.dict_precursors_var[label].coords['latitude'].values
            if self.ll[1] - self.ll[0] < 0:
                self.dict_precursors_var[label] = self.dict_precursors_var[label].reindex(
                    latitude=self.dict_precursors_var[label].latitude[::-1])
            if self.config.has_option(config_var, "coords"):
                self.cut_area = True
                self.dict_precursors_var[label] = self.dict_precursors_var[label].sel(latitude=slice(*self.lat_bnds),
                                                                                      longitude=slice(*self.lon_bnds))
        elif all(x in self.dict_precursors_var[label].coords for x in ['lat', 'lon']):
            self.label_lat, self.label_lon = 'lat', 'lon'
            # if latitude is sorted from positive to negative change order
            self.ll = self.dict_precursors_var[label].coords['lat'].values
            if self.ll[1] - self.ll[0] < 0:
                self.dict_precursors_var[label] = self.dict_precursors_var[label].reindex(
                    lat=self.dict_precursors_var[label].lat[::-1])
            if self.config.has_option(config_var, "coords"):
                self.dict_precursors_var[label] = self.dict_precursors_var[label].sel(lat=slice(*self.lat_bnds),
                                                                                      lon=slice(*self.lon_bnds))
        else:
            raise ValueError("Spatial attribute (e.g. latitude and longitude) not found!")

    def _transform_to_1d_and_remove_nans(self, label: str):
        """
        :param label: name of variable. If one uses a cluster the variable name is the same for different
        model initialization and therefore I renamed the variable name.
        """
        self.logger.info('Reshape to 1D array and remove nans')
        self.dict_prec_1D_var[label] = np.reshape(np.array(self.dict_precursors_var[label]),
                                                  [np.array(self.dict_precursors_var[label])
                                                  .shape[0], -1])
        # cluster algorithm does not work with nans, maybe drop it?
        self.dict_prec_1D_var[label][self.dict_prec_1D_var[label] != self.dict_prec_1D_var[label]] = 0

    def _get_dim_boundaries(self, config_var: str):
        """
        :param config_var: variable name of precursor section on config
        get boundaries of initialization file
        """
        if self.config.has_option(config_var, "coords"):
            self.lat_min, self.lat_max, self.lon_min, self.lon_max = \
                map(float, self.config[config_var]["coords"].split(','))
            self.lat_bnds, self.lon_bnds = [self.lat_min, self.lat_max], [self.lon_min, self.lon_max]

    def _get_and_apply_mask(self, label, config_var):
        """
        apply mask to input-file
        :param label: name of variable. If one uses a cluster the variable name is the same for different
        model initialization and therefore I renamed the variable name.
        :param config_var: variable name of precursor section on config
        """
        self.dict_mask[config_var] = np.loadtxt(self.config[config_var]["mask"])
        self.dict_precursors_var[label] = self.dict_precursors_var[label].where(self.dict_mask[config_var] == 0, 0)

    def reshape_precursors_to_1d(self):
        """ reshape precursors into 1D arrays"""
        for nb, prec in enumerate(self.dict_precursors.keys()):
            self._set_v_arr(prec, nb)
            self.dict_prec_1D[prec] = np.reshape(self.v_arr, (self.v_arr.shape[0], -1))

    def _set_cluster_frequency(self, f: np.ndarray):
        """
        calculate frequency of f
        :param f: np.ndarray containing the cluster number for each state
        """
        self.cluster_frequency = np.bincount(f)

    def get_composites_data_1d(self, f: np.ndarray, k: int, method_name: str, predictand: str):
        """
        calculate composites of standardized precursors
        :param f: np.ndarray containing the cluster number for each state
        :param k: cluster number
        :param method_name: name of method used to calculate clusters (e.g. ward)
        :param predictand: name of predicand
        """
        self._set_cluster_frequency(f)
        for prec in self.dict_precursors.keys():
            self._create_composites(prec, f, k, method_name, predictand)

    def _set_v_arr(self, prec: str, nb: int):
        """ get array from dictionary for certain key
        :param prec: key of dict_precursors dictionary
        :param nb: number of precursor: Is this really needed?
        """
        self.nb = nb
        self.v_arr = np.array(self.dict_precursors[prec]["var"].squeeze())

    def _calculate_standardized_precursors(self, label: str):
        """
        Calculate standardized composites by mean and standard deviation
        :param label: name of variable. If one uses a cluster the variable name is the same for different
        model initialization and therefore I renamed the variable name.
        """
        self.logger.info("Calculate Standardized values")
        self.varmean = np.mean(self.dict_prec_1D_var[label], axis=0)
        self.varAnom = self.dict_prec_1D_var[label] - self.varmean
        if self.output_label == "standardized":
            self.sigma_var = np.sum(self.varAnom * self.varAnom) / (self.varAnom.shape[0] * self.varAnom.shape[1])
            self.dict_standardized_precursors_var[label] = self.varAnom / self.sigma_var
        else:
            self.dict_standardized_precursors_var[label] = self.varAnom

    def _create_composites(self, key: str, f: np.ndarray, k: int, method_name: str, predictand: str):
        """
        create composites of 1D precursors
        :param key: key/variable name of composites
        :param f: np.ndarray containing the cluster number for each state
        :param k: cluster number
        :param method_name: name of method used to calculate clusters (e.g. ward)
        :param predictand: name of predictand
        """
        self.logger.info("Calculate composites")
        self.dict_standardized_precursors[key] = np.array(self.dict_standardized_precursors[key])
        self.dict_composites[key] = np.zeros((int(k), self.dict_standardized_precursors[key].shape[1]),
                                             dtype=np.float64)

        for i_cl, cluster_nr in enumerate(f):
            self.dict_composites[key][cluster_nr] += self.dict_standardized_precursors[key][i_cl]

        for i_cl in range(int(k)):
            self.dict_composites[key][i_cl] = np.divide(self.dict_composites[key][i_cl], (self.cluster_frequency[i_cl]))

        self._set_directory_plots(f"output-{self.output_label}//{predictand}/Composites/{key}/"
                                  f"{method_name}_Composite_{k}/plots/")
        Path(self.directory_plots).mkdir(parents=True, exist_ok=True)
        self._set_directory_files(f"output-{self.output_label}//{predictand}/Composites/"
                                  f"{key}/{method_name}_Composite_{k}/files/")
        Path(self.directory_files).mkdir(parents=True, exist_ok=True)

    def plot_composites(self, k: int):
        """
        Plot clusters
        :param k: cluster number
        """
        self._save_composites_plot(k)

    def _save_composites_plot(self, k: int):
        """
        save clusters into one plot using xarray library
        :param k: cluster number
        """
        self.logger.info("Plot composites")
        for prec in self.precs_sections:
            self._create_dataset_from_composites(self.config[prec]["name"], k)
            n_rows1 = min(k, 4)
            n_cols1 = np.ceil(k / n_rows1)
            if self.var == "ICEFRAC" or self.var == "FSNO":
                if k == 5 or k == 7:
                    n_cols1 = 1
            # n_cols1 = max(n, 1)
            map_project_array = [ccrs.PlateCarree(), ccrs.NorthPolarStereo(), ccrs.LambertConformal(),
                                 ccrs.Orthographic(0, 90)]
            map_project = map_project_array[self.map_proj_nr]
            self.ds_arrays = self.ds.to_array()
            p = self.ds_arrays.plot(transform=ccrs.PlateCarree(),
                                    col='variable',
                                    col_wrap=int(n_cols1),
                                    cmap=plt.cm.get_cmap('seismic', 31),
                                    subplot_kws={'projection': map_project},
                                    size=self.fig_size,  # 10, 3,  8
                                    add_colorbar=False,
                                    aspect=self.aspect,  # 2,  # 1.5
                                    # cbar_kwargs={'shrink': 0.8, 'pad':0.02},
                                    )

            p.fig.subplots_adjust(hspace=0.1, wspace=0.1)
            p.add_colorbar(orientation="vertical")
            for ip, ax in enumerate(p.axes.flat):
                if ip < k:
                    ax.add_feature(cfeature.BORDERS, linewidth=0.1)
                    ax.add_feature(cfeature.COASTLINE, linewidth=0.1)
                    ax.gridlines(color="Gray", linestyle="dotted", linewidth=0.5)
                    if self.cut_area:
                        ax.set_extent([self.lon_min, self.lon_max, self.lat_min, (2 * self.lat_max - 90)])

                    ax.set_title(f"Composite {ip}, p = {self.percent_boot:3f}", fontsize=10)
            plt.savefig(f"{self.directory_plots}/composites.pdf")
            plt.close()

    def plot_years(self, predictand: str, method_name: str, k: int, f: np.ndarray):
        """
        Plot composites for all years
        :param predictand: name of predictand/ cluster for saving in the correct folder
        :param f: list containing the cluster number for each state
        :param k: cluster number
        :param method_name: name of method used to calculate clusters (e.g. ward)
        """
        for prec in self.precs_sections:
            self._set_directory_plots(
                f"output//{predictand}/Composites/{self.var}/{method_name}_Composite_{k}/years/plots/")
            Path(self.directory_plots).mkdir(parents=True, exist_ok=True)
            self._set_directory_files(
                f"output//{predictand}/Composites/{self.var}//{method_name}_Composite_{k}/years/files/")
            Path(self.directory_files).mkdir(parents=True, exist_ok=True)
            for year in range(len(self.dict_precursors[self.var])):
                var_reshape = np.reshape(self.dict_standardized_precursors[self.config[prec]["name"]][year],
                                         (self.dict_precursors[self.config[prec]["name"]].shape[1],
                                          self.dict_precursors[self.config[prec]["name"]].shape[2]))
                self.data_vars[f"{self.config[prec]['name']}"] = xr.DataArray(var_reshape, dims=('lat', 'lon'))
                # (( 'lat','lon'), self.clusters_reshape[i])
                self.ds = xr.Dataset(self.data_vars, coords={
                    'lon': self.dict_precursors[self.config[prec]['name']].coords["lon"].values,
                    'lat': self.dict_precursors[self.config[prec]['name']].coords["lat"].values})
                # n_cols = max(n, 1)
                map_project_array = [ccrs.PlateCarree(), ccrs.NorthPolarStereo(), ccrs.LambertConformal(),
                                     ccrs.Orthographic(0, 90)]
                map_project = map_project_array[self.map_proj_nr]
                self.ds_arrays = self.ds[f"{self.config[prec]['name']}"]
                ax = plt.axes(projection=map_project)
                self.ds[f"{self.config[prec]['name']}"].plot(
                    ax=ax,
                    transform=ccrs.PlateCarree(),  # the data's projection
                    cmap=plt.cm.get_cmap('seismic', 31),
                    cbar_kwargs={'shrink': 0.8},
                )

                ax.add_feature(cfeature.BORDERS, linewidth=0.1)
                ax.coastlines()
                if self.cut_area:
                    ax.set_extent([self.lon_min, self.lon_max, self.lat_min, (2 * self.lat_max - 90)])
                # Without this aspect attributes the maps will look chaotic and the
                # "extent" attribute above will be ignored
                # ax.set_aspect("equal")
                # aspect = self.aspect,  # 2,  # 1.5
                ax.set_aspect(self.aspect)
                ax.set_title(f"{self.var}, {self.dict_precursors[self.var].time.values[year]}, cluster: "
                             f"{f[year]}", fontsize=10)
                self.logger.debug(
                    f"Save in {self.directory_plots}/{self.var}_{self.dict_precursors[self.var].time.values[year]}.pdf")
                plt.savefig(
                    f"{self.directory_plots}/{year:03d}_{self.var}_{self.dict_precursors[self.var].time.values[year]}"
                    f".pdf")
                plt.savefig(
                    f"{self.directory_plots}/{year:03d}_{self.var}_{self.dict_precursors[self.var].time.values[year]}"
                    f".png")
                plt.close()

    def time_plot(self, predictand: str, method_name: str, k: int, f: np.ndarray):
        """
        Plot mean var for each time point
        :param predictand: name of predictand/ cluster for saving in the correct folder
        :param f: list containing the cluster number for each state
        :param k: cluster number
        :param method_name: name of method used to calculate clusters (e.g. ward)
        """
        for prec in self.precs_sections:
            self.var = f"{self.config[prec]['name']}"
            self._set_directory_plots(
                f"output//{predictand}/Composites/{self.var}/{method_name}_Composite_{k}/plots/")
            Path(self.directory_plots).mkdir(parents=True, exist_ok=True)
            self._set_directory_files(
                f"output//{predictand}/Composites/{self.var}//{method_name}_Composite_{k}/files/")
            Path(self.directory_files).mkdir(parents=True, exist_ok=True)
            time1 = self.dict_precursors[self.var].coords["time"].values
            time = [t_i for t_i in range(len(time1))]
            vals = np.zeros((len(self.dict_precursors[self.var].coords["time"].values)))
            for year in range(len(self.dict_standardized_precursors[self.var])):
                vals[year] = np.mean(self.dict_prec_1D[self.var][year])

            df = pd.DataFrame(index=time, columns=[f"cluster {cl_i}" for cl_i in range(k)], dtype=float)
            df_all = pd.DataFrame(vals, index=time, columns=[""], dtype=float)
            for t, f_value, val in zip(time, f, vals):
                df.at[t, f"cluster {f_value}"] = np.float(val)
            # plt.plot(time, vals, color='k', linewidth=1)
            # sns_plot = sns.scatterplot(data=df)
            # , x="timepoint", y="signal",hue="event", style="event",markers=True, dashes=False
            sns.lineplot(data=df_all, palette=sns.color_palette("mako_r", 1), linewidth=0.5, alpha=0.7)
            sns_plot = sns.scatterplot(
                data=df)  # , x="timepoint", y="signal",hue="event", style="event",markers=True, dashes=False

            plt.xlabel(" model year SON ")
            # Set y-axis label
            plt.ylabel(f"mean {self.var}")
            fig_sns = sns_plot.get_figure()
            self.logger.debug(f"Save in {self.directory_plots}/{self.var}_time_plot.pdf")
            fig_sns.savefig(f"{self.directory_plots}/{self.var}_time_plot.pdf")
            fig_sns.savefig(f"{self.directory_plots}/{self.var}_time_plot.png")
            plt.close()

    def _create_dataset_from_composites(self, key: str, k: int):
        """
        create dataset for clusters as netcdf using xarray library
        :param key: name of precursor
        :param k: cluster number
        """
        self.logger.info("Create dataset with composites as variables")
        self._set_composites_reshape(key, k)
        self.data_vars = {}
        for ik in range(k):
            self.data_vars[f"composites_{key}_{ik}"] = \
                (xr.DataArray((self.composites_reshape[key][ik]), dims=('lat', 'lon')))
            # (( 'lat','lon'), self.clusters_reshape[ik])
        self.ds = xr.Dataset(self.data_vars, coords={'lon': self.dict_precursors[key].coords["lon"].values,
                                                     'lat': self.dict_precursors[key].coords["lat"].values})

    def _set_composites_reshape(self, key: str, k: int):
        """
        reshape 1d clusterst to 2d clusters
        :param key: name of precursor
        :param k: cluster number
        """
        self.composites_reshape[key] = np.zeros(
            (k, self.dict_precursors[key].shape[1],
             self.dict_precursors[key].shape[2]))
        for ik in range(int(k)):
            self.composites_reshape[key][ik] = \
                np.reshape(self.dict_composites[key][ik],
                           (self.dict_precursors[key].shape[1],
                            self.dict_precursors[key].shape[2]))

    def save_composites(self, k: int):
        """
        save clusters using xarray
        :param k: cluster number
        """
        self.logger.info("Save composites as netcdf")
        for prec in self.precs_sections:
            self._create_dataset_from_composites(self.config[prec]["name"], k)
            self.ds.to_netcdf(f"{self.directory_files}/composites_{self.config[prec]['name']}_{k}.nc")

    def _set_directory_plots(self, directory: str):
        """
        set directories for plots
        :param directory: path for plot directory
        """
        self.directory_plots = directory

    def _set_directory_files(self, directory: str):
        """
        set directories for plots
        :param directory: path for files directory
        """
        self.directory_files = directory

    def get_composites_data_1d_year(self, year: int, f: np.ndarray, k: int, method_name: str, predictand: str):
        """ calculate composites of standardized precursors
        :param year: year for which we would like to forecast predictand
        :param f: np.ndarray containing the cluster number for each state
        :param k: number of clusters
        :param method_name: method for clustering
        :param predictand: variable name of predictand
        """
        self._set_cluster_frequency(f)
        for prec in self.dict_precursors.keys():
            self._remove_year(prec, year)
            self._create_composites(prec, f, k, method_name, predictand)

    def get_composites_data_1d_train_test(self, train_data: list, f: np.ndarray, k: int, method_name: str,
                                          predictand: str):
        """ calculate composites of standardized precursors
        :param train_data: list with data which we use for forecasting
        :param f: np.ndarray containing the cluster number for each state
        :param k: number of clusters
        :param method_name: method for clustering
        :param predictand: variable name of predictand
        """
        self.dict_standardized_precursors = train_data
        self._set_cluster_frequency(f)
        for prec in self.dict_precursors.keys():
            self._create_composites(prec, f, k, method_name, predictand)

    def _remove_year(self, prec: str, year=-1):
        """
        Calculate standardized composites by mean and standard deviation
        :param year: year for which we would like to forecast variable
        :type prec: name of precursor
        """
        if year != -1:
            self.dict_prec_1D[f"{prec}_year"] = np.delete(self.dict_prec_1D[prec], year, axis=0)
        else:
            self.dict_prec_1D[f"{prec}_year"] = self.dict_prec_1D[prec]
        self.logger.info("Remove 1 year")
        self.dict_standardized_precursors[prec] = self.dict_prec_1D[f"{prec}_year"]
        del self.dict_prec_1D[f"{prec}_year"]
