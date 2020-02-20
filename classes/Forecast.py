#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 16:11:43 2019

@author: sonja
"""
from logging import config
import logging
import numpy as np
import ast
import configparser
from scipy import stats
from itertools import combinations



class Forecast:
    def __init__(self, inifile_in: str, cl_config: dict, k=8, method_name="ward"):
        """
        Initialize Forecast--> read forecast parameters using ini-file
        :param inifile_in: file for initialization of variable
        :param cl_config: dictionary, where all information of logger is stored from classes/config
        :param k: number of clusters
        :param method_name: method for clustering data
        """
        logging.config.dictConfig(cl_config)
        self.logger = logging.getLogger(__name__)
        self.logger.info('Read ini-file')
        self.inifile = inifile_in
        # initialize data
        self.alpha_all = None
        self.alpha_matrix = None
        self.forecast_var = None
        self.selected_composites_1D, self.selected_data_1d = None, None
        self.t_corr_arr = None
        self.t_corr_signif_arr = None
        self.list_precursors_all = []

        # initialize list of possible precursors according to k and method_name
        self._get_forecast_parameters(k, method_name)

    def _get_forecast_parameters(self, k: int, method_name: str):
        """
        load all forecast parameters and save composites which should be saved
        :param k: number of clusters
        :param method_name: method for clustering data
        """
        self.method_name = method_name
        self.k = k
        self.config = configparser.ConfigParser()
        self.config.read(self.inifile)
        # get begin and end time for forecast from ini file
        self.beg_year = int(self.config["Forecast-Parameters"]["begin"])
        # assert self.beg_year >= 1967, \
        #     logger.error(f"Start year of forecast must be 1967 or higher! It is {self.beg_year}")
        self.end_year = int(self.config["Forecast-Parameters"]["end"])
        self.diff = self.end_year - self.beg_year
        assert self.diff > 0, \
            self.logger.error(f"end year of forecast must be greater than start year! It is {self.end_year}")
        # Select from saveArray which precursors should be used to calculate forecast
        # self.list_precursors = ast.literal_eval(config.get("Forecast-Parameters", "forecastprecs"))
        self.list_precursors_all = ast.literal_eval(self.config.get("Forecast-Parameters", "forecastprecs"))
        self.plot = self.config["Forecast-Parameters"]["plot"]
        self.all_combinations = self.config["Forecast-Parameters"]["all_combinations"]
        self.list_precursors_combinations = []
        if self.all_combinations:
            for r in range(len(self.list_precursors_all)):
                [self.list_precursors_combinations
                     .append(list(x)) for x in combinations(self.list_precursors_all, r + 1)]
        else:
            self.list_precursors = self.list_precursors_all

    def prediction_train_test(self, clusters_1d: dict, composites_1d: dict, X_test: np.ndarray, year):
        """make forecast
        :param clusters_1d: dict of all k-clusters
        :param composites_1d: dict of composites with time and one-dimensional array
        :param X_test: np.ndarray with all the train data of precursors
        :param year: year which should be forecasted
        """
        # Merge only those precursors which were selected
        self.selected_composites_1D, self.selected_data_1d = self._merge_selected_precursors(composites_1d,
                                                                                             X_test, year)

        # Calculate projection coefficients
        self.alpha_matrix = np.zeros((self.k, self.k))
        self.alpha_all = self._projection_coefficients()

        self.forecast_var = np.zeros(len(clusters_1d[0]))
        for i in range(int(self.k)):
            self.forecast_var += self.alpha_all[i] * clusters_1d[int(i)]

        return self.forecast_var

    def prediction(self, clusters_1d: dict, composites_1d: dict, data_year_1d: dict, year: int):
        """make forecast
        :param clusters_1d: dict of all k-clusters
        :param composites_1d: dict of composites with time and one-dimensional array
        :param data_year_1d: np.ndarray with all  data of precursors
        :param year: year which should be forecasted
        """
        # Merge only those precursors which were selected
        self.selected_composites_1D, self.selected_data_1d = self._merge_selected_precursors(composites_1d,
                                                                                             data_year_1d, year)
        # Calculate projection coefficients
        self.alpha_matrix = np.zeros((self.k, self.k))
        self.alpha_all = self._projection_coefficients()

        self.forecast_var = np.zeros(len(clusters_1d[0]))
        for i in range(int(self.k)):
            self.forecast_var += self.alpha_all[i] * clusters_1d[int(i)]
        return self.forecast_var

    def _merge_selected_precursors(self, composites_1d: dict, data_1d: dict, year: int):
        """  Merge only those precursors which were selected
        :param composites_1d: dictionary of all k-composites
        :param data_1d: dictionary of precursor data for specified year
        :param year: year for which predictand should be forecasted
        """
        return (np.hstack([composites_1d[i] for i in self.list_precursors]),
                np.hstack([data_1d[i][year] for i in self.list_precursors]))

    def _projection_coefficients(self):
        """ calculate projection coefficients"""
        # composite_i*composite_j

        for i in range(int(self.k)):
            for j in range(int(self.k)):
                # scalarproduct--> inner product
                self.alpha_matrix[i, j] = sum(self.selected_composites_1D[i, :] * self.selected_composites_1D[j, :])

        # # composite_i*y(t)
        rhs = np.zeros(self.k)
        for i in range(int(self.k)):
            rhs[i] = sum(self.selected_composites_1D[i, :] * self.selected_data_1d[:])
        #
        # # Singular values smaller (in modulus) than 0.01 * largest_singular_value (again, in modulus) are set to zero
        self.pinv_matrix = np.linalg.pinv(self.alpha_matrix, 0.01)

        return np.dot(self.pinv_matrix, rhs)

    def calculate_time_correlation(self, data_1d: np.ndarray, forecast_data: np.ndarray, time_start_file: int):
        """calculate time correlation for given forecast data
        :param forecast_data: list of forecasted data
        :param data_1d: list of obervational data which shoud be forecasted
        :type time_start_file: int
        """
        self.t_corr_arr = np.zeros((forecast_data.shape[1]))
        self.t_corr_signif_arr = np.zeros((forecast_data.shape[1]))
        start_obs = int(self.beg_year) - time_start_file + 1
        end_end = int(self.end_year) - time_start_file + 1
        for i in range(forecast_data.shape[1]):
            # check whether value in data range does not change, but has same value as observational data
            # because then it is still correct even though there is no curve --> only for debug
            # normally these are areas which are not considered in the analys as continents for sst
            if all(data_1d[start_obs:end_end, i] == forecast_data[0:self.diff, i]) and \
                    data_1d[start_obs:end_end, i][0] == 0:
                self.t_corr_arr[i] =1
                self.t_corr_signif_arr[i] = 0
            else:
                t_corr = stats.pearsonr(forecast_data[0:self.diff, i], data_1d[start_obs:end_end, i])
                self.t_corr_arr[i] = t_corr[0]
                self.t_corr_signif_arr[i] = t_corr[1]
                # if t_corr[1] > 0.05:
                #     self.t_corr_signif_arr[i] = self.t_corr_arr[i]
                # else:
                #     self.t_corr_signif_arr[i] = 1

        return self.t_corr_arr, self.t_corr_signif_arr

    def calculate_time_correlation_all_times(self, data_1d: np.ndarray, forecast_data: np.ndarray, time_start_file: int):
        """calculate time correlation for given forecast data for all times --> easier function for model data
        :param forecast_data: list of forecasted data
        :param data_1d: list of obervational data which shoud be forecasted
        :type time_start_file: int
        """
        self.t_corr_arr = np.zeros((forecast_data.shape[1]))
        self.t_corr_signif_arr = np.zeros((forecast_data.shape[1]))

        for i in range(forecast_data.shape[1]):
            # if value does not change, it leads to np.nan
            t_corr = stats.pearsonr(forecast_data[:, i], data_1d[:, i])
            self.t_corr_arr[i] = t_corr[0]
            self.t_corr_signif_arr[i] = t_corr[1]

        return self.t_corr_arr, self.t_corr_signif_arr