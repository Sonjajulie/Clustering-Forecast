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
from config_dict import config
from pathlib import Path
import cartopy.feature as cfeature
import cartopy.crs as ccrs

# seed random number generator
np.random.seed(0)

sns.set()
logging.config.dictConfig(config)
logger = logging.getLogger(__name__)


class Precursors:
    """Store and analyse possible precursors"""

    def __init__(self, inifile_in):
        """
        Store all parameters necessary for loading the netcdf file
        :type inifile_in: basestring
        """
        logger.info("Initialize class composites")
        self.ini = inifile_in
        self.config = configparser.ConfigParser()
        self.config.read(self.ini)
        # all precursors in the ini-file should be assigned to the dictionary
        self._initialize_attributes()
        #  read precursors from ini-file
        self.precs_sections = [prec for prec in self.config.sections() if 'PREC:' in prec]
        self.dict_precursors = {}
        self.dict_mask = {}

        for prec in self.precs_sections:
            if "nc" in self.config[prec]["filepath"]:
                self.dict_precursors[self.config[prec]["name"]] = xr.open_dataset(self.config[prec]["filepath"])
                self._set_area_composite(self.config[prec]["name"], prec)
                # self.dict_precursors[self.config[prec]["name"]].info()
                if self.config.has_option(prec, "mask"):
                    self._get_and_apply_mask(self.config[prec]["name"], prec)
                self._transform_to_1d_and_remove_nans(self.config[prec]["name"])
                if self.config.has_option(prec, "map_proj"):
                    self.map_proj_nr = int(self.config[prec]["map_proj"])
                self.fig_size = int(self.config[prec]["figsize"])
                self.aspect = int(self.config[prec]["aspect"])

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
                for file in range(len(self.list_of_files)):
                    logger.debug(f"file {file}: {self.list_of_files[file]}")
                    self.dict_precursors[f"{self.var}_{file}"] = \
                        xr.open_dataset(self.list_of_files[file])[self.config[prec]["var"]]
                    self._set_area_composite(f"{self.var}_{file}", prec)
                    if self.config.has_option(prec, "mask"):
                        self._get_and_apply_mask(f"{self.var}_{file}", prec)
                    self._transform_to_1d_and_remove_nans(f"{self.var}_{file}")
                if self.config.has_option(prec, "map_proj"):
                    self.map_proj_nr = int(self.config[prec]["map_proj"])
                self.fig_size = int(self.config[prec]["figsize"])
                self.aspect = int(self.config[prec]["aspect"])
                # change dimenson of precursor  to changed to dim = [time*models,lons,lats]!
                list_time_model = [f"{c_file}: {c_var}" for c_file in range(len(self.list_of_files))
                                   for c_var in self.dict_precursors[f"{self.var}_{c_file}"].coords['time'].values]
                logger.debug(f"dims {self.label_lat}, {self.label_lon}")
                # concatenate all files to one precursor
                list_lons = self.dict_precursors[f"{self.var}_{0}"].coords[self.label_lon].values
                list_lats = self.dict_precursors[f"{self.var}_{0}"].coords[self.label_lat].values
                self.dict_precursors[self.var] = xr.DataArray(np.concatenate([self.dict_precursors
                                                                             .pop(f"{self.var}_{file}") for file in
                                                                              range(len(self.list_of_files))]),
                                                              coords={'time': list_time_model, 'lon': list_lons,
                                                                      'lat': list_lats},
                                                              dims=['time', self.label_lat, self.label_lon])
                self.dict_prec_1D[self.var] = np.concatenate([self.dict_prec_1D.pop(f"{self.var}_{file}")
                                                              for file in range(len(self.list_of_files))])

    def _initialize_attributes(self):
        """ initialize dictionaries for composites and plot properties"""

        self.dict_predict = {}
        self.dict_prec_1D = {}
        self.dict_composites = {}
        self.dict_standardized_precursors = {}
        self.composites_reshape = None
        self.cluster_frequency = None
        self.cut_area = False
        self.aspect_ratio = 1
        self.map_proj_nr = None

    def _set_area_composite(self, label, config_var):
        """
         Get Longitudes and Latitudes, check whether latitudes go form -90 to 90 or from 90 to -90,
        if the letter, reverse order
        :type label: basestring
        :type config_var: basestring
        """
        self._get_dim_boundaries(config_var)
        # check name for latitude and longitude and cut area accordingly
        #  https://stackoverflow.com/questions/29135885/netcdf4-extract-for-subset-of-lat-lon
        if all(x in self.dict_precursors[label].coords for x in ['latitude', 'longitude']):
            self.label_lat, self.label_lon = 'latitude', 'longitude'
            # if latitude is sorted from positive to negative change order
            # otherwise sel-function will not work-> is there a better solution?
            self.ll = self.dict_precursors[label].coords['latitude'].values
            if self.ll[1] - self.ll[0] < 0:
                self.dict_precursors[label] = self.dict_precursors[label].reindex(
                    latitude=self.dict_precursors[label].latitude[::-1])
            if self.config.has_option(config_var, "coords"):
                self.cut_area = True
                self.dict_precursors[label] = self.dict_precursors[label].sel(latitude=slice(*self.lat_bnds),
                                                                              longitude=slice(*self.lon_bnds))

        elif all(x in self.dict_precursors[label].coords for x in ['lat', 'lon']):
            self.label_lat, self.label_lon = 'lat', 'lon'
            # if latitude is sorted from positive to negative change order
            self.ll = self.dict_precursors[label].coords['lat'].values
            if self.ll[1] - self.ll[0] < 0:
                self.dict_precursors[label] = self.dict_precursors[label].reindex(
                    lat=self.dict_precursors[label].lat[::-1])
            if self.config.has_option(config_var, "coords"):
                self.dict_precursors[label] = self.dict_precursors[label].sel(lat=slice(*self.lat_bnds),
                                                                              lon=slice(*self.lon_bnds))
        else:
            raise NameError("Spatial attribute (e.g. latitude and longitude) not found!")

    def _transform_to_1d_and_remove_nans(self, label):
        """
        transfrom array and set values 0, where no data is found as well reshape to 1D
        :type label: basestring
        """
        logger.info('Reshape to 1D array and remove nans')
        self.dict_prec_1D[label] = np.reshape(np.array(self.dict_precursors[label]),
                                              [np.array(self.dict_precursors[label])
                                              .shape[0], -1])
        self.dict_prec_1D[label][self.dict_prec_1D[label] != self.dict_prec_1D[label]] = 0

    def _get_dim_boundaries(self, config_var):
        """
        get dimensions of latitudes and longitudes from ini-file
        :type config_var: basestring
        """
        if self.config.has_option(config_var, "coords"):
            self.lat_min, self.lat_max, self.lon_min, self.lon_max, self.end_year = \
                map(float, self.config[config_var]["coords"].split(','))
            self.lat_bnds, self.lon_bnds = [self.lat_min, self.lat_max], [self.lon_min, self.lon_max]

    def _get_and_apply_mask(self, label, config_var):
        """
        apply mask to input-file
        :type label: basestring
        :type config_var: basestring
        """
        self.dict_mask[config_var] = np.loadtxt(self.config[config_var]["mask"])
        self.dict_precursors[label] = self.dict_precursors[label].where(self.dict_mask[config_var] == 0, 0)

    def reshape_precursors_to_1d(self):
        """ reshape precursors into 1D arrays"""
        for nb, prec in enumerate(self.dict_precursors.keys()):
            self._set_v_arr(prec)
            self.dict_prec_1D[prec] = np.reshape(self.v_arr, (self.v_arr.shape[0], -1))

    def _set_cluster_frequency(self, f):
        """
        calculate frequency of f
        :type f: list [int]
        """
        self.cluster_frequency = np.bincount(f)

    def get_composites_data_1d(self, year, f, k, method_name, predictand):
        """ calculate composites of standardized precursors
        :type year: int
        :type f: list [int]
        :type k: int
        :type method_name: string
        :type predictand: string
        """
        self._set_cluster_frequency(f)
        for prec in self.dict_precursors.keys():
            self._calculate_standardized_precursors(prec, year)
            self._create_composites(prec, f, k, method_name, predictand)

    def _set_v_arr(self, label):
        """
        get array from dictionary for certain key
        :type label: basestring
        """
        self.v_arr = np.array(self.dict_precursors[label][self.config[self.precs_sections[i]]["var"]].squeeze())

    def _calculate_standardized_precursors(self, prec, year=-1):
        """
        Calculate standardized composites by mean and standard deviation
        :type year: int
        :type prec: basestring
        """
        if year != -1:
            self.dict_prec_1D[f"{prec}_year"] = np.delete(self.dict_prec_1D[prec], year, axis=0)
        else:
            self.dict_prec_1D[f"{prec}_year"] = self.dict_prec_1D[prec]
        logger.info("Calculate Standardized values")
        self.varmean = np.mean(self.dict_prec_1D[f"{prec}_year"], axis=0)
        self.varAnom = self.dict_prec_1D[f"{prec}_year"] - self.varmean
        self.sigma_var = np.sum(self.varAnom * self.varAnom) / (self.varAnom.shape[0] * self.varAnom.shape[1])
        self.dict_standardized_precursors[prec] = self.varAnom / self.sigma_var
        del self.dict_prec_1D[f"{prec}_year"]

    def _create_composites(self, key, f, k, method_name, predictand):
        """
        create composites of 1D precursors
        :type key: string
        :type f: list [int]
        :type k: int
        :type method_name: basestring
        :type predictand: basestring
        """
        logger.info("Calculate composites")
        self.dict_composites[key] = np.zeros((int(k), self.dict_standardized_precursors[key].shape[1]))

        for i_cl, cluster_nr in enumerate(f):
            self.dict_composites[key][cluster_nr] += self.dict_standardized_precursors[key][i_cl]

        for i_cl in range(int(k)):
            self.dict_composites[key][i_cl] = np.divide(self.dict_composites[key][i_cl], (self.cluster_frequency[i_cl]))

        self._set_directory_plots(f"output//{predictand}/Precursors/{key}/{method_name}_Composite_{k}/plots/")
        Path(self.directory_plots).mkdir(parents=True, exist_ok=True)
        self._set_directory_files(f"output//{predictand}/Precursors/{key}/{method_name}_Composite_{k}/files/")
        Path(self.directory_files).mkdir(parents=True, exist_ok=True)

    def plot_composites(self, k):
        """
        Plot clusters
        :type k: int
        """
        self._save_composites_plot(k)

    def _save_composites_plot(self, k):
        """
        save clusters into one plot using xarray library
        :type k: int
        """
        logger.info("Plot Precursors")
        for prec in self.precs_sections:
            self._create_dataset_from_composites(self.config[prec]["name"], k)
            n_rows1 = min(k, 4)
            n_cols1 = np.ceil(k / n_rows1)
            if self.var == "ICEFRAC":
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
                                    cmap=plt.cm.get_cmap('seismic', 51),
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
                    ax.set_title(f"Precursor composite {ip}", fontsize=10)
            plt.savefig(f"{self.directory_plots}/precursors_{self.var}.pdf")
            plt.close()

    def _create_dataset_from_composites(self, key, k):
        """ create dataset for clusters as netcdf using xarray library
        :type key: string
        :type k: int
        """
        logger.info("Create dataset with composites as variables")
        self._set_composites_reshape(key, k)
        self.data_vars = {}
        for ik in range(k):
            self.data_vars[f"composites_{key}_{ik}"] = \
                (xr.DataArray((self.composites_reshape[key][ik]), dims=('lat', 'lon')))
            # (( 'lat','lon'), self.clusters_reshape[ik])
        self.ds = xr.Dataset(self.data_vars, coords={'lon': self.dict_precursors[key].coords["lon"].values,
                                                     'lat': self.dict_precursors[key].coords["lat"].values})

    def _set_composites_reshape(self, key, k):
        """
        reshape 1d clusterst to 2d clusters
        :type key: string
        :type k: int
        """
        self.composites_reshape[key] = np.zeros(
            (k, self.dict_precursors[key].shape[1],
             self.dict_precursors[key].shape[2]))
        for ik in range(int(k)):
            self.composites_reshape[key][ik] = \
                np.reshape(self.dict_composites[key][ik],
                           (self.dict_precursors[key].shape[1],
                            self.dict_precursors[key].shape[2]))

    def save_composites(self, k):
        """
        save clusters using xarray
        :type k: int
        """
        logger.info("Save composites as netcdf")
        for prec in self.precs_sections:
            self._create_dataset_from_composites(self.config[prec]["name"], k)
            self.ds.to_netcdf(f"{self.directory_files}/composites_{self.config[prec]['name']}_{k}.nc")

    def _set_directory_plots(self, directory):
        """
        set directories for plots
        :type directory: string
        """
        self.directory_plots = directory

    def _set_directory_files(self, directory):
        """
        set directory for files
        :type directory: string
        """
        self.directory_files = directory


if __name__ == '__main__':

    # import libraries
    import matplotlib.pyplot as plt

    # Used variables
    inifile = "/home/sonja/Documents/Clustering-forecast/clustering_forecast.ini"
    # https://docs.python.org/3/library/configparser.html
    config = configparser.ConfigParser()
    config.read(inifile)

    precs = Precursors(inifile)
    number_precs = len(precs.dict_precursors)

    # plot data
    n_cols = 4
    n_rows = 2
    # fig, axes = plt.subplots(ncols=n_cols, figsize=(20,20))
    fig, axes = plt.subplots(ncols=n_cols, nrows=n_rows, figsize=(25, 20))
    precs_sections = [prec for prec in config.sections() if 'PREC' in prec]
    for i in range(n_rows):
        for j in range(n_cols):
            if (j + n_cols * i) < number_precs:
                (precs.dict_precursors[config[precs_sections[j + n_cols * i]]["name"]]
                 [config[precs_sections[j + n_cols * i]]["var"]].isel(time=0).plot(ax=axes[i, j]))
            # precs.dict_precursors[config[precs_sections[i]]["name"]].info()

    plt.savefig(f"output/precursors.pdf")
    precs.reshape_precursors_to_1d()
