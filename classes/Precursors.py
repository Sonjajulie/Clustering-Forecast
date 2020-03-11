#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 16:11:43 2019

@author: sonja
"""
# noinspection PyUnresolvedReferences
import os
import matplotlib as mpl

if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import logging
from logging import config
from pathlib import Path
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
# from xscale import signal
# seed the pseudorandom number generator
from classes.Composites import Composites

# seed random number generator
np.random.seed(0)
sns.set()


class Precursors(Composites):
    """Store and analyse possible precursors"""

    def __init__(self, inifile_in: str, output_path: str, output_label: str, cl_config: dict):
        """
        Store all parameters necessary for loading the netcdf file
        :param inifile_in: file for initialization of variable
        :param output_path: path, where output should be saved
        :param output_label: label for substring of output directory
        :param cl_config: dictionary, where all information of logger is stored from classes/config
        """
        super().__init__(inifile_in, output_path, output_label, cl_config)
        logging.config.dictConfig(cl_config)
        self.logger = logging.getLogger(__name__)

    def _calculate_standardized_precursors(self, label: str):
        """
        Calculate standardized composites by mean and standard deviation
        :param label: name of variable. If one uses a cluster the variable name is the same for different
        model initialization and therefore I renamed the variable name.
        """
        self.logger.info("Calculate Standardized values")
        self.varmean = np.mean(self.dict_prec_1D_var[label], axis=0)
        self.varAnom = self.dict_prec_1D_var[label] - self.varmean
        if self.output_label == "standardized" or self.output_label == "standardized-opt":
            self.sigma_var = np.sum(self.varAnom * self.varAnom) / (self.varAnom.shape[0] * self.varAnom.shape[1])
            self.dict_standardized_precursors_var[label] = self.varAnom / self.sigma_var
        else:
            self.dict_standardized_precursors_var[label] = self.varAnom

    def plot_composites_without_significance(self, k: int):
        """
        Plot clusters
        :param k: cluster number
        """
        self._save_composites_plot_without_significance(k)


    def _save_composites_plot_without_significance(self, k: int):
        """
        save clusters into one plot using xarray library
        :param k: cluster number
        """
        self.logger.info("Plot composites")
        for prec in self.precs_sections:
            self._create_dataset_from_composites(prec, k)
            n_rows1 = min(k, 4)
            n_cols1 = np.ceil(k / n_rows1)
            if "ICEFRAC" in self.var or "FSNO" in self.var:
                if k == 5 or k == 7:
                    n_cols1 = 1
            # n_cols1 = max(n, 1)
            map_project_array = [ccrs.PlateCarree(), ccrs.NorthPolarStereo(), ccrs.LambertConformal(),
                                 ccrs.Orthographic(0, 90), ccrs.PlateCarree(180)]
            map_project = map_project_array[self.map_proj_nr[self.var]]
            lsize = 14
            axislsize = 9
            plt.rc("legend", frameon=False, fontsize=lsize)
            plt.rc("axes", labelsize=lsize, titlesize=lsize)
            plt.rc("xtick", labelsize=lsize)
            plt.rc("ytick", labelsize=lsize)
            plt.rc("lines", linewidth=0.5)
            plt.rc("figure", dpi=100)
            p = self.data_vars[f"composite{self.var}"].plot(transform=ccrs.PlateCarree(), col='c',
                                                            col_wrap=int(n_cols1), cmap=plt.cm.get_cmap('seismic', 31),
                                                            subplot_kws={'projection': map_project},
                                                            size=self.fig_size[self.var],  # 10, 3,  8
                                                            add_colorbar=False, aspect=self.aspect[self.var],
                                                            # 2,  # 1.5
                                                            # cbar_kwargs={'shrink': 0.8, 'pad':0.02},
                                                            )

            p.fig.subplots_adjust(hspace=0.2, wspace=0.15)
            for ip, ax in enumerate(p.axes.flat):
                if ip < k:
                    ax.add_feature(cfeature.BORDERS, linewidth=0.1)
                    ax.add_feature(cfeature.COASTLINE, linewidth=0.1)
                    # ax.gridlines(color="Gray", linestyle="dotted", linewidth=0.5)

                    title = self.cluster_frequency[ip] / np.sum(self.cluster_frequency) * 100.
                    ax.set_title(f"Composite {ip} ({title:4.2f} %)", fontsize=lsize)
                    plt.rcParams['hatch.linewidth'] = 0.03  # hatch linewidth
                    plt.rcParams['hatch.color'] = 'k'  # hatch color --> black
                    if self.map_proj_nr[self.var] == 0 or self.map_proj_nr[self.var] == 4:
                        gl = ax.gridlines( draw_labels=True,
                                          linewidth=0.02, color='gray', linestyle='--')
                        gl.xlabels_top = False
                        gl.ylabels_right = False
                        if n_cols1 > 1 and ip % n_cols1:
                            gl.ylabels_left = False
                        gl.xformatter = LONGITUDE_FORMATTER
                        gl.yformatter = LATITUDE_FORMATTER
                        gl.xlabel_style = {'size': axislsize, 'color': 'black'}
                        gl.ylabel_style = {'size': axislsize, 'color': 'black'}
                        gl.xlocator = mticker.FixedLocator([i for i in range(-180, 360, 30)])
                        gl.ylocator = mticker.FixedLocator([i for i in range(-100, 100, 20)])
                    if self.cut_area[self.var] and self.var != "ICEFRAC":
                        # self.var = self.config[prec]["name"]
                        # self._get_dim_boundaries(prec)
                        # ax.set_extent([self.lon_min, self.lon_max, self.lat_min, (2 * self.lat_max - 90)])
                        ax.set_extent([self.lon_min[self.var], self.lon_max[self.var], self.lat_min[self.var], self.lat_max[self.var]])

                    # Without this aspect attributes the maps will look chaotic and the
                    # "extent" attribute above will be ignored
                    # ax.set_aspect("equal")

            p.add_colorbar(orientation="vertical",
                           label=f"{self.dict_precursors[self.var].attrs['long_name']}["
                                 f"{self.dict_precursors[self.var].attrs['units']}]",
                           shrink=0.8,
                           aspect=30, pad=0.1)
            # plt.subplots_adjust(left=0.03, right=0.82, top=0.90, bottom=0.05)
            plt.savefig(f"{self.directories_plots[self.var]}/composites-{self.var}.pdf")
            plt.close()

    def _set_directory_plots(self, directory: str):
        """
        set directories for plots
        :param directory: path for plot directory
        """
        self.directory_plots = f"{self.output_path}/{directory}"

    def _set_directory_files(self, directory: str):
        """
        set directories for plots
        :param directory: path for files directory
        """
        self.directory_files = f"{self.output_path}/{directory}"

    def get_composites_data_1d_year(self, year: int, f: np.ndarray, k: int, method_name: str, predictand: str):
        """ calculate composites of standardized precursors
        :param year: year for which we would like to forecast_nn predictand
        :param f: np.ndarray containing the cluster number for each state
        :param k: number of clusters
        :param method_name: method for clustering
        :param predictand: variable name of predictand
        """
        self._set_cluster_frequency(f)
        for prec in self.dict_precursors.keys():
            self._remove_year(prec, year)
            self._create_composites(prec, f, k, method_name, predictand)

    def get_composites_data_1d_train_test(self, train_data: dict, f: np.ndarray, k: int, method_name: str,
                                          predictand: str):
        """ calculate composites of standardized precursors
        :param train_data: list with data which we use for forecasting
        :param f: np.ndarray containing the cluster number for each state
        :param k: number of clusters
        :param method_name: method for clustering
        :param predictand: variable name of predictand

        """
        # do I really need this?
        # self.dict_prec_1D_var = train_data
        self._set_cluster_frequency(f)
        self.dict_standardized_precursors = train_data
        # go through all forecast_nn variables
        for prec in self.dict_standardized_precursors.keys():
            self.var = prec


            # self._calculate_standardized_precursors(self.var)
            # {self.output_path}/
            self.directories_plots[self.var] = f"output-{self.output_label}/" \
                                               f"/{predictand}/Composites/{self.var}/" \
                                               f"{method_name}_Composite_{k}/plots/"
            Path(self.directories_plots[self.var]).mkdir(parents=True, exist_ok=True)
            self.directories_files[self.var] = f"output-{self.output_label}/{predictand}" \
                                               f"/Composites/{self.var}/{method_name}_Composite_{k}/files/"
            Path(self.directories_files[self.var]).mkdir(parents=True, exist_ok=True)
            self._create_composites(prec, f, k, method_name, predictand)

    def _remove_year(self, prec: str, year=-1):
        """
        Calculate standardized composites by mean and standard deviation
        :param year: year for which we would like to forecast_nn variable
        :type prec: name of precursor
        """
        if year != -1:
            self.dict_prec_1D[f"{prec}_year"] = np.delete(self.dict_prec_1D[prec], year, axis=0)
        else:
            self.dict_prec_1D[f"{prec}_year"] = self.dict_prec_1D[prec]
        self.logger.info("Remove 1 year")
        self.dict_standardized_precursors[prec] = self.dict_prec_1D[f"{prec}_year"]
        del self.dict_prec_1D[f"{prec}_year"]
