# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 15:19:19 2017

@author: sonja
"""
# import libraries
import numpy as np
from classes.Precursors import Precursors
from classes.Predictand import Predictand
from classes.Forecast import Forecast
from classes.ExportVarPlot import ExportVarPlot
from classes.ClusteringParser import ClusteringParser
from classes.Config import Config
from scipy.optimize import shgo
import json

def lat_range(x):
    if (x[1] < 0 and x[0] > 0):
        return -1
    return x[1] - x[0] - 10  # >=0


def lon_range(x):
    if (x[1] < 0 and x[0] > 0):
        return -1
    return x[1] - x[0] - 10  # >=0



def train_test_split_pred(predictand, precursors, data_range, forecast_predictands: list, train_size=0.66, random_state=2019,):
    """
    calculate number of train and test points
    : param predictand: obect of the class Predictand
    : param precursors: obect of the class Precursors
    : param data_range: how long should data go, since some precursors have not total datarange, e.g. (0, 1980)
    : param train_size: how many data points should be used for training, is given in percentage
    : param random_state: which seed should be used
    """
    np.random.seed(random_state)
    len_predicts = data_range[1] - data_range[0]
    len_train_data = int(len_predicts * train_size)
    selected_time_steps = np.random.choice(len_predicts, len_train_data, replace=False)
    y_train = {}
    # noinspection PyPep8Naming
    X_train = {}
    y_test = {}
    # noinspection PyPep8Naming
    X_test = {}

    for i in range(len_predicts):
        if i in selected_time_steps:
            y_train.setdefault(predictand.var, []).append(predictand.dict_standardized_pred_1D[predictand.var][i])
            for prec in forecast_predictands:
                X_train.setdefault(prec, []).append(precursors.dict_standardized_precursors[prec][i])
        else:
            y_test.setdefault(predictand.var, []).append(predictand.dict_standardized_pred_1D[predictand.var][i])
            for prec in forecast_predictands:
                X_test.setdefault(prec, []).append(precursors.dict_standardized_precursors[prec][i])
    return y_train, X_train, y_test, X_test



def train_test_split_prec(precursors, data_range, forecast_predictands: list, train_size=0.66, random_state=2019,):
    """
    calculate number of train and test points
    : param predictand: obect of the class Predictand
    : param precursors: obect of the class Precursors
    : param data_range: how long should data go, since some precursors have not total datarange, e.g. (0, 1980)
    : param train_size: how many data points should be used for training, is given in percentage
    : param random_state: which seed should be used
    """
    np.random.seed(random_state)
    len_predicts = data_range[1] - data_range[0]
    len_train_data = int(len_predicts * train_size)
    selected_time_steps = np.random.choice(len_predicts, len_train_data, replace=False)
    # noinspection PyPep8Naming
    X_train = {}
    # noinspection PyPep8Naming
    X_test = {}

    for i in range(len_predicts):
        if i in selected_time_steps:
            for prec in forecast_predictands:
                X_train.setdefault(prec, []).append(precursors.dict_standardized_precursors[prec][i])
        else:
            for prec in forecast_predictands:
                X_test.setdefault(prec, []).append(precursors.dict_standardized_precursors[prec][i])
    return X_train, X_test


def main(cl_parser: ClusteringParser, cl_config: dict):
    logger.info("Start forecast model opt")

    # load inifile according to variable
    # var = cl_parser.arguments['predictand'] # not needed anymore, because total inifile is given
    inifile = cl_parser.arguments['inifile']
    output_label = cl_parser.arguments['outputlabel']
    output_path = cl_parser.arguments['outputpath']
    data_range = cl_parser.arguments['datarange']
    predictand = Predictand(inifile, output_path, output_label, cl_config)
    dict_skills_pattern = {}

    # load forecast_nn-parameters
    method_name = 'ward'
    k = 5
    forecast = Forecast(inifile, cl_config, k, method_name)
    logger.info("Clusters: " + str(forecast.k))

    # load precursors
    precursors = Precursors(inifile, output_path, output_label, cl_config)
    forecast_precursors = cl_parser.arguments['forecast_precursors']

    y_train, X_train, y_test, X_test = train_test_split_pred(predictand, precursors, data_range, forecast_precursors)
    # Calculate clusters of precursors for var, by removing one year
    predictand.calculate_clusters_from_test_data(y_train, forecast.method_name, forecast.k)

    def skill(x, info):
        cut_area_opt = x
        precursors.set_area_composite_opt(forecast_precursors[0], cut_area_opt)
        # Create train and test dataset with an 66:33 split
        # noinspection PyPep8Naming
        X_train, X_test = train_test_split_prec(precursors, data_range, forecast_precursors)
        # Calculate clusters of precursors for var, by removing one year
        # Calculate composites
        precursors.get_composites_data_1d_train_test(X_train, predictand.f, forecast.k, forecast.method_name,
                                                     predictand.var)

        # Calculate forecast_nn for all years
        forecast.list_precursors = forecast_precursors
        pattern_corr_values = []
        # Prediction
        forecast_data = np.zeros((len(y_test[f"{predictand.var}"]),
                                  predictand.dict_pred_1D[f"{predictand.var}"].shape[1]))


        logger.info(forecast_precursors)

        for year in range(len(y_test[predictand.var])):  # len(y_test[predictand.var])):
            # print(year)
            forecast_temp = forecast.prediction(predictand.clusters, precursors.dict_composites, X_test, year)
            # Assign forecast_nn data to array
            forecast_data[year] = forecast_temp


        # Calculate time correlation for each point
        time_correlation, significance = forecast.calculate_time_correlation_all_times(
            np.array(y_test[f"{predictand.var}"]), forecast_data)
        # display information
        # display information
        if info['Nfeval'] % 2 == 0:
            logger.info(f"{x[0]:4f} {x[1]:4f} {x[2]:4f} {x[3]:4f} {time_correlation}")
        info['Nfeval'] += 1

        return  -time_correlation

    cons = ({'type': 'ineq', 'fun': lat_range},
            {'type': 'ineq', 'fun': lon_range},)
    var = forecast_precursors[0]
    bounds = [(precursors.lat_min[var],precursors.lat_max[var] - 10),
              (precursors.lat_min[var] + 10, precursors.lat_max[var]),
              (precursors.lon_min[var], precursors.lon_max[var] - 10),
              (precursors.lon_min[var] + 10,precursors.lon_max[var])]
    res = shgo(skill, bounds, args=({'Nfeval':0},), iters=3, constraints=cons)
    print(res)


if __name__ == '__main__':
    import logging.config
    parser = ClusteringParser()
    # logs / log_
    # {sys.argv[2]}.log
    config = Config(parser.arguments['logfile'])
    logger = logging.getLogger(__name__)

    # read config log file from classes.Config
    logging.config.dictConfig(config.config_dict)
    logger.info("Start clustering program")
    main(parser,  config.config_dict)
