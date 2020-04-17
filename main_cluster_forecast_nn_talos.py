# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 15:19:19 2017

@author: sonja
"""
# import libraries
import numpy as np
from classes.Precursors import Precursors
from classes.Predictand import Predictand

from classes.ExportVarPlot import ExportVarPlot
from classes.ClusteringParser import ClusteringParser
from classes.Config import Config
from scipy import stats
import pandas as pd
#
#



def train_test_split_pred(predictand, precursors, data_range, train_size=0.66, random_state=2019):
    """
    calculate number of train and test points
    : param predictand: obect of the class Predictand
    : param precursors: obect of the class Precursors
    : param data_range: how long should data go, since some precursors have not total datarange, e.g. (0, 1980)
    : param train_size: how many data points should be used for training, is given in percentage
    : param random_state: which seed should be used
    """
    np.random.seed(random_state)
    len_predicts = len(predictand.dict_standardized_pred_1D[predictand.var][data_range[0]:data_range[1]])
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
            for prec in precursors.dict_precursors.keys():
                X_train.setdefault(prec, []).append(precursors.dict_standardized_precursors[prec][i])
        else:
            y_test.setdefault(predictand.var, []).append(predictand.dict_standardized_pred_1D[predictand.var][i])
            for prec in precursors.dict_precursors.keys():
                X_test.setdefault(prec, []).append(precursors.dict_standardized_precursors[prec][i])
    return y_train, X_train, y_test, X_test


def main(cl_parser: ClusteringParser, cl_config: dict):
    logger.info("Start forecast_nn model")

    # load inifile according to variable
    # var = cl_parser.arguments['predictand'] # not needed anymore, because total inifile is given
    inifile = cl_parser.arguments['inifile']
    output_label = cl_parser.arguments['outputlabel']
    output_path = cl_parser.arguments['outputpath']
    data_range = cl_parser.arguments['datarange']
    predictand = Predictand(inifile, output_path, output_label, cl_config)
    dict_skills_pattern = {}

    # load precursors
    precursors = Precursors(inifile, output_path, output_label, cl_config)

    # load forecast_nn-parameters
    method_name = 'ward'
    k = 5

    # unfortunately, I can not load the library as the beginning, because netcdf load function for xarray
    # does not work then
    from classes.ForecastNN import ForecastNN
    forecast_nn = ForecastNN(inifile, output_path, output_label, cl_config, predictand.var, k, method_name)
    logger.info("Clusters: " + str(forecast_nn.k))


    # Create train and test dataset with an 66:33 split
    # noinspection PyPep8Naming
    y_train, X_train, y_test, X_test = train_test_split_pred(predictand, precursors, data_range)

    # Calculate clusters of precursors for var, by removing one year
    predictand.calculate_clusters_from_test_data(y_train, forecast_nn.method_name, forecast_nn.k)
    # Calculate composites
    precursors.get_composites_data_1d_train_test(X_train, predictand.f, forecast_nn.k, forecast_nn.method_name,
                                                 predictand.var)
    # precursors.plot_composites(k, 1)
    # subtract train mean also for test data
    # for prec in forecast_nn.list_precursors_all:
    #     X_test[prec] -= precursors.varmean
    # y_test[predictand.var] -= predictand.varmean
    df_parameters_opt = pd.DataFrame(columns=["precursor", "nr_neurons", "opt_method", "nr_epochs", "nr_layers", "lr_rate",
                                              "nr_batch_size", "time_correlation", "pattern_correlation"])

#     nr_epochs = 500
    #for forecast_predictands in forecast_nn.list_precursors_combinations:
    # Calculate forecast_nn for all years
    # ~ forecast_nn.list_precursors = forecast_predictands
    forecast_nn.list_precursors = ["Z500"]
    list_methods = ["SGD","Adam"]
    forecast_predictands = forecast_nn.list_precursors

    dict_calc_X_y = {
        'composites_1d': precursors.dict_composites,
        'forecast_predictands': forecast_nn.list_precursors,
        'clusters_1d': predictand.clusters,
    }
    alphas_train, alphas_val, y_train_pseudo, y_val_pseudo = forecast_nn.calc_alphas_for_talos(X_train, y_train[predictand.var], dict_calc_X_y)
    len_alpha = len(alphas_train)
    # set the parameter space boundary
    p = {
        # ~ 'lr': [0.1, 0.01, 0.001, 0.0001],
        'lr': [0.1, 0.01, 0.001],
        # ~ 'activation': ['relu', 'elu'],
        'activation': ['relu'],
        'kernel_initializer': ['random_uniform'],
         # ~ 'optimizer': ['Nadam','Adam','SGD'],
         'optimizer': ['Adam','SGD'],
         'losses': ['logcosh'],
         'shapes': ['brick'],
         # ~ 'first_neuron': [5, 16, 32, 64, 128],
         'first_neuron': [8, 16],
        'forecast_predictands': forecast_nn.list_precursors,
        'len_alpha':[len_alpha],
         # ~ 'hidden_layers': [0, 1, 2, 3, 4, 5],
         'hidden_layers': [1, 2, 3, 4, 5],
         # ~ 'dropout': [.2, .3, .4],
         'dropout': [.2, .3, .4, .5],
         # ~ 'batch_size': [5, 8, 16, 32, 64],
         'batch_size': [8, 16, 32],
         'epochs': [5],
         'last_activation': ['linear'],
         'y_train': [y_train[predictand.var]],
         'x_test': [X_test],
         'y_test': [y_test[predictand.var]],
        'composites_1d': [precursors.dict_composites],
         'pattern_corr': [1],
         'time_corr': [1],
    }

    index_df = 0
    import talos as ta
    t = ta.Scan(x=alphas_train,
                y=y_train_pseudo,
                x_val=alphas_val,
                y_val=y_val_pseudo,
                model= forecast_nn.train_nn_talos,
                params=p,
                experiment_name='opt-nn-clustering')
    # print('\nAdvanced usage:')
    # r = ta.Reporting("/home/sonja/Documents/Clustering-Forecast/desktop/opt-nn-clustering/040120220751.csv")
    # r = ta.Reporting(t)
    #
    # # returns the results dataframe
    # print(r.data)
    #
    # # returns the highest value for 'val_fmeasure'
    # # print(r.high('val_fmeasure'))
    #
    # # returns the number of rounds it took to find best model
    # r.rounds2high()
    #
    # # draws a histogram for 'val_acc'
    # print(r.plot_hist())











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
    main(parser, config.config_dict)

