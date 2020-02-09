#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 16:11:43 2019

@author: sonja
"""
# noinspection PyUnresolvedReferences
import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
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
from scipy import stats
import pandas as pd

# from xscale import signal
# seed the pseudorandom number generator


# seed random number generator
np.random.seed(0)

sns.set()
logging.config.dictConfig(config)
logger = logging.getLogger(__name__)


class Composites:
    """Store and analyse possible precursors"""

    def __init__(self, inifile_in, output_label):
        """ Store all parameters necessary for loading the netcdf file"""
        logger.info("Initialize class composites")
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
            if "nc" in self.config[prec]["filepath"]:
                # with xr.set_options(enable_cftimeindex=True):
                self.var = self.config[prec]["name"]
                self.dict_precursors[self.var] = xr.open_dataset(self.config[prec]["filepath"])[
                    self.config[prec]["var"]]
                self._set_area_composite(self.var, prec)
                if self.config.has_option(prec, "mask"):
                    self._get_and_apply_mask(self.config[prec]["name"], prec)
                self._transform_to_1d_and_remove_nans(self.config[prec]["name"])
                self._calculate_standardized_precursors(self.var)
                list_time_model = []
                for c_var in self.dict_precursors[self.config[prec]["name"]].coords['time'].values:
                    list_time_model.append(f"{c_var}")
                self.dict_precursors = {self.var: xr.DataArray(list(self.dict_precursors[self.var].values),
                                                               coords={'time': list_time_model,
                                                                       'lon': self.dict_precursors[f"{self.var}"]
                                                               .coords[self.label_lon].values,
                                                                       'lat': self.dict_precursors[f"{self.var}"]
                                                               .coords[self.label_lat].values},
                                                               dims=['time', self.label_lat, self.label_lon])}
                self.dict_standardized_precursors = {self.var: np.concatenate(list(self.dict_standardized_precursors
                                                                                   .values()))}
                self.dict_prec_1D = {self.var: np.concatenate(list(self.dict_prec_1D.values()))}
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
                    logger.debug(f"file {file}: {self.list_of_files[file]}")
                    self.dict_precursors[f"{self.var}_{file}"] = \
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
                                   for jtime in self.dict_precursors[f"{self.var}_{imodel}"].coords['time'].values]
                logger.debug(f"dims {self.label_lat}, {self.label_lon}")
                self.dict_precursors = {self.var: xr.DataArray(np.concatenate(list(self.dict_precursors.values())),
                                                               coords={'time': list_time_model,
                                                                       'lon': self.dict_precursors[f"{self.var}_{0}"]
                                                               .coords[self.label_lon].values,
                                                                       'lat': self.dict_precursors[f"{self.var}_{0}"]
                                                               .coords[self.label_lat].values},
                                                               dims=['time', self.label_lat, self.label_lon])}
                self.dict_standardized_precursors = {self.var: np.concatenate(list(
                    self.dict_standardized_precursors.values()))}
                self.dict_prec_1D = {self.var: np.concatenate(list(self.dict_prec_1D.values()))}

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

    def _set_area_composite(self, label, config_var):
        """
         Get Longitudes and Latitudes, check whether latitudes go form -90 to 90 or from 90 to -90,
        if the letter, reverse order
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
            raise ValueError("Spatial attribute (e.g. latitude and longitude) not found!")

    def _transform_to_1d_and_remove_nans(self, label):
        """ transfrom array and set values 0, where no data is found as well reshape to 1D"""
        logger.info('Reshape to 1D array and remove nans')
        self.dict_prec_1D[label] = np.reshape(np.array(self.dict_precursors[label]),
                                              [np.array(self.dict_precursors[label])
                                              .shape[0], -1])
        self.dict_prec_1D[label][self.dict_prec_1D[label] != self.dict_prec_1D[label]] = 0

    def _get_dim_boundaries(self, config_var):
        """ get dimensions of latitudes and longitudes from ini-file"""
        if self.config.has_option(config_var, "coords"):
            self.lat_min, self.lat_max, self.lon_min, self.lon_max = \
                map(float, self.config[config_var]["coords"].split(','))
            self.lat_bnds, self.lon_bnds = [self.lat_min, self.lat_max], [self.lon_min, self.lon_max]

    def _get_and_apply_mask(self, label, config_var):
        """apply mask to input-file"""
        self.dict_mask[config_var] = np.loadtxt(self.config[config_var]["mask"])
        self.dict_precursors[label] = self.dict_precursors[label].where(self.dict_mask[config_var] == 0, 0)

    def reshape_precursors_to_1d(self):
        """ reshape precursors into 1D arrays"""
        for nb, prec in enumerate(self.dict_precursors.keys()):
            self._set_v_arr(prec)
            self.dict_prec_1D[prec] = np.reshape(self.v_arr, (self.v_arr.shape[0], -1))

    def _set_cluster_frequency(self, f):
        """ calculate frequency of f"""
        self.cluster_frequency = np.bincount(f)

    def get_composites_data_1d(self, f, k, method_name, predictand):
        """ calculate composites of standardized precursors"""
        self._set_cluster_frequency(f)
        for prec in self.dict_precursors.keys():
            self._create_composites(prec, f, k, method_name, predictand)

    def _set_v_arr(self, label):
        """ get array from dictionary for certain key"""
        self.v_arr = np.array(self.dict_precursors[label][self.config[self.precs_sections[i]]["var"]].squeeze())

    def _calculate_standardized_precursors(self, label):
        """ Calculate standardized composites by mean and standard deviation """
        logger.info("Calculate Standardized values")
        self.varmean = np.mean(self.dict_prec_1D[label], axis=0)
        self.varAnom = self.dict_prec_1D[label] - self.varmean
        if self.output_label == "standardized":
            self.sigma_var = np.sum(self.varAnom * self.varAnom) / (self.varAnom.shape[0] * self.varAnom.shape[1])
            self.dict_standardized_precursors[label] = self.varAnom / self.sigma_var
        else:
            self.dict_standardized_precursors[label] = self.varAnom

    def _create_composites(self, key, f, k, method_name, predictand):
        """ create composites of 1D precursors"""
        logger.info("Calculate composites")
        self.dict_composites[key] = np.zeros((int(k), self.dict_standardized_precursors[key].shape[1]),
                                             dtype=np.float64)

        for i_cl, cluster_nr in enumerate(f):
            self.dict_composites[key][cluster_nr] += self.dict_standardized_precursors[key][i_cl]

        for i_cl in range(int(k)):
            self.dict_composites[key][i_cl] = np.divide(self.dict_composites[key][i_cl], (self.cluster_frequency[i_cl]))

        self._set_directory_plots(f"output-{self.output_label}//{predictand}/Composites/{key}/{method_name}_Composite_{k}/plots/")
        Path(self.directory_plots).mkdir(parents=True, exist_ok=True)
        self._set_directory_files(f"output-{self.output_label}//{predictand}/Composites/{key}/{method_name}_Composite_{k}/files/")
        Path(self.directory_files).mkdir(parents=True, exist_ok=True)

    def plot_composites(self, k, percent_boot):
        """Plot clusters"""
        self._save_composites_plot(k, percent_boot)

    def _save_composites_plot(self, k, percent_boot):
        """ save clusters into one plot using xarray library"""
        logger.info("Plot composites")
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


                    self._calculate_significance(ip, k, self.config[prec]["name"], percent_boot)
                    ax.set_title(f"Composite {ip}, p = {self.percent_boot:3f}", fontsize=10)
                    plt.rcParams['hatch.linewidth'] = 0.03  # previous pdf hatch linewidth
                    plt.rcParams['hatch.color'] = 'k'  # previous pdf hatch linewidth
                    ax.contourf(self.lons, self.lats,
                                np.reshape(self.composites_significance[self.config[prec]["name"]][ip],
                                           (self.dict_precursors[self.config[prec]["name"]].shape[1],
                                            self.dict_precursors[self.config[prec]["name"]].shape[2])),
                                # levels=[0, 100 - percent_boot, 50, percent_boot, 100],
                                levels=[0, self.percent_boot, 50, 100 - self.percent_boot, 100],
                                hatches=["/////", None, None, "/////", None], colors='none',
                                transform=ccrs.PlateCarree())  # alpha=0.0,
                    # levels = [0, 100 - percent_boot, 100 - percent_boot + 4, 50,
                    # percent_boot - 4, percent_boot, 100],
                    # hatches = ["/////", "...", None, None, "...", "/////", None], colors = 'none',
            plt.savefig(f"{self.directory_plots}/composites.pdf")
            plt.close()

    def plot_years(self, predictand, method_name, k, f):
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
                # ax.contourf(self.lons, self.lats, significance, levels=[0., 0.05, 0.5, 0.95, 1],
                #             hatches=["/////", ".....", None, None, None], colors='none', transform=ccrs.PlateCarree())
                # # hatches=["/////", ".....", ",,,,,", "/////", "....."], colors='none', transform=ccrs.PlateCarree())
                logger.debug(
                    f"Save in {self.directory_plots}/{self.var}_{self.dict_precursors[self.var].time.values[year]}.pdf")
                plt.savefig(
                    f"{self.directory_plots}/{year:03d}_{self.var}_{self.dict_precursors[self.var].time.values[year]}"
                    f".pdf")
                plt.savefig(
                    f"{self.directory_plots}/{year:03d}_{self.var}_{self.dict_precursors[self.var].time.values[year]}"
                    f".png")
                plt.close()

    def time_plot(self, predictand, method_name, k, f):
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
            logger.debug(f"Save in {self.directory_plots}/{self.var}_time_plot.pdf")
            fig_sns.savefig(f"{self.directory_plots}/{self.var}_time_plot.pdf")
            fig_sns.savefig(f"{self.directory_plots}/{self.var}_time_plot.png")
            plt.close()

    def _create_dataset_from_composites(self, key, k):
        """ create dataset for clusters as netcdf using xarray library"""
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
        """ reshape 1d clusterst to 2d clusters"""
        self.composites_reshape[key] = np.zeros(
            (k, self.dict_precursors[key].shape[1],
             self.dict_precursors[key].shape[2]))
        for ik in range(int(k)):
            self.composites_reshape[key][ik] = \
                np.reshape(self.dict_composites[key][ik],
                           (self.dict_precursors[key].shape[1],
                            self.dict_precursors[key].shape[2]))

    def save_composites(self, k):
        """ save clusters using xarray"""
        logger.info("Save composites as netcdf")
        for prec in self.precs_sections:
            self._create_dataset_from_composites(self.config[prec]["name"], k)
            self.ds.to_netcdf(f"{self.directory_files}/composites_{self.config[prec]['name']}_{k}.nc")

    def _set_directory_plots(self, directory):
        """ set directories for plots"""
        self.directory_plots = directory

    def _set_directory_files(self, directory):
        """ set directory for files"""
        self.directory_files = directory

    def _calculate_significance(self, ik, k, key, percent_boot):
        """calculate significance of composite using the bootstrap method
        Composite [key][ik]
        ik - index of the k-th composite
        k - total composites number of composites key
        key - dictionary key of composite
        percent_boot - significance of significance level percent_boot
        """
        # initialize variables
        self.initialize_variables_for_significance(key, ik, k, percent_boot)
        # call bootstrap method
        self._bootstrap_method(key, ik)

    def set_lats_and_lons(self, key, ik):
        """ get longittudes and latitudes for bootstrap method"""
        self.lons, self.lats = np.meshgrid(self.dict_precursors[key].coords['lon'].values,
                                           self.dict_precursors[key].coords['lat'].values)
        self.lats1 = np.reshape(self.lons, [self.dict_composites[key][ik].shape[0], -1])
        self.lons1 = np.reshape(self.lats, [self.dict_composites[key][ik].shape[0], -1])

    def initialize_variables_for_significance(self, key, ik, k, percent_boot):
        """ initialize variables for bootstrap method"""
        logger.info(f"Calculate Significance {k}, Significance level {percent_boot}")
        self.composites_significance[key] = np.zeros((k, self.dict_composites[key][ik].shape[0]))
        # Calculate end_n different randomly selected clusters to see whether our cluster is significant
        self.percent_boot = percent_boot
        self.end_n = 5000
        self.set_lats_and_lons(key, ik)
        # get time of our cluster for selecting different states
        self.time_dim = self.dict_standardized_precursors[key].shape[0]
        # initialize array for randomly selected clusters
        self.bootstrap_arrays = np.zeros((self.end_n, self.dict_standardized_precursors[key].shape[1]))

    def _bootstrap_method(self, key, ik):
        """ calculate significance according to bootstrap method"""
        for t in range(self.end_n):
            # choose random time states, but no duplicate states
            chosen_time_steps = np.random.choice(self.time_dim, self.cluster_frequency[ik], replace=False)
            self.bootstrap_arrays[t] = np.mean(self.dict_standardized_precursors[key][chosen_time_steps], axis=0)

        # sort p-values and compare with i/N * alpha_FDR ?
        alphas = []
        for ci, comp_val in enumerate(self.dict_composites[key][ik]):
            xyt_array = self.bootstrap_arrays[:, ci]
            self.composites_significance[key][ik][ci] = stats.percentileofscore(xyt_array, comp_val)
            if self.composites_significance[key][ik][ci] > 0:
                alphas.append(self.composites_significance[key][ik][ci])
        alphas = sorted(alphas)
        len_alpha = len(alphas)
        for i_sig in range(1, len_alpha + 1):
            if (alphas[len_alpha - 1 - i_sig] < 100 - (i_sig / len_alpha * self.percent_boot)) \
                    or (alphas[i_sig] > i_sig / len_alpha * self.percent_boot):
                self.percent_boot = i_sig / len_alpha * self.percent_boot
                break
        # self.percent_boot = percent_boot
        logger.debug(f"Winner is {self.percent_boot}")

if __name__ == '__main__':

    # import libraries
    import matplotlib.pyplot as plt

    # Used variables
    inifile = "/home/sonja/Documents/Clustering-forecast/clustering_forecast.ini"
    # https://docs.python.org/3/library/configparser.html
    config = configparser.ConfigParser()
    config.read(inifile)

    precs = Composites(inifile)
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
