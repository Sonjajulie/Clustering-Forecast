# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 15:19:19 2017

@author: sonja
"""
# import libraries
import numpy as np
from classes.Precursors import Precursors
from classes.Predictand import Predictand
from classes.ForecastNN import ForecastNN
from classes.ExportVarPlot import ExportVarPlot
from classes.ClusteringParser import ClusteringParser
from classes.Config import Config
from scipy import stats
import pandas as pd
import os  # needed for loop over files
# from pulp import *  # integer optimization
from ortools.linear_solver import pywraplp

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

    # load forecast_nn-parameters
    method_name = 'ward'
    k = 5
    forecast_nn = ForecastNN(inifile, output_path, output_label, cl_config, predictand.var, k, method_name)
    logger.info("Clusters: " + str(forecast_nn.k))

    # load precursors
    precursors = Precursors(inifile, output_path, output_label, cl_config)

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
    # df_parameters_opt = pd.DataFrame(columns=["precursor", "nr_neurons", "opt_method", "nr_epochs", "nr_layers", "lr_rate",
    #                                           "nr_batch_size", "time_correlation", "pattern_correlation"])

    nr_epochs = 500
    #for forecast_predictands in forecast_nn.list_precursors_combinations:
    # Calculate forecast_nn for all years
    # ~ forecast_nn.list_precursors = forecast_predictands
    forecast_nn.list_precursors = ["Z500"]
    list_methods = ["SGD","Adam"]
    forecast_predictands = forecast_nn.list_precursors


    index_df = 0
    def objective(opt_m, nr_batch_size, lr_rate, nr_layers, nr_neurons):
        opt_method = "Adam"
        # train small NN
        forecast_nn.train_nn_opt(forecast_nn.list_precursors, predictand.clusters,
                                 precursors.dict_composites, X_train,
                                 y_train[f"{predictand.var}"], nr_neurons, opt_method,
                                 nr_epochs, nr_layers,
                                 lr_rate, nr_batch_size)

        # Calculate forecast_nn for all years
        pattern_corr_values = []

        # Prediction
        forecast_data = np.zeros((len(y_test[f"{predictand.var}"]),
                                  predictand.dict_pred_1D[f"{predictand.var}"].shape[1]))
        logger.info(forecast_predictands)

        for year in range(len(y_test[predictand.var])):  # len(y_test[predictand.var])):
            print(year)
            forecast_temp = forecast_nn.prediction_nn(forecast_nn.list_precursors_all,
                                                      predictand.clusters,
                                                      precursors.dict_composites, X_test, year)
            # Assign forecast_nn data to array
            forecast_data[year] = forecast_temp

            # Calculate pattern correlation
            pattern_corr_values.append(
                stats.pearsonr(forecast_temp, y_test[f"{predictand.var}"][year])[0])

        # Calculate time correlation for each point
        time_correlation, significance = forecast_nn.calculate_time_correlation_all_times(
            np.array(y_test[f"{predictand.var}"]), forecast_data)

        # Reshape correlation maps
        pred_t_corr_reshape = np.reshape(time_correlation,
                                         (predictand.dict_predict[predictand.var].shape[1],
                                          predictand.dict_predict[predictand.var].shape[2]))
        significance_corr_reshape = np.reshape(significance, (
            predictand.dict_predict[predictand.var].shape[1],
            predictand.dict_predict[predictand.var].shape[2]))

        logger.info(f'time correlation: {np.nanmean(pred_t_corr_reshape)}')
        logger.info(f'pattern correlation: {np.nanmean(pattern_corr_values)}')

        logger.info("Plot and save variables")
        ex = ExportVarPlot(output_label, cl_config)

        ex.save_plot_and_time_correlationNN(forecast_nn.list_precursors, predictand,
                                          pred_t_corr_reshape,
                                          significance_corr_reshape,
                                          forecast_nn.list_precursors_all,
                                          np.nanmean(pred_t_corr_reshape), nr_neurons,
        opt_method, nr_epochs, nr_layers, lr_rate, nr_batch_size)
        df_parameters_opt = pd.DataFrame({"precursor": ex.predictor_names, "nr_neurons": nr_neurons,
                                  "opt_method": opt_method, "nr_epochs": nr_epochs,
                                  "nr_layers": nr_layers, "lr_rate": lr_rate,
                                  "nr_batch_size": nr_batch_size,
                                  "time_correlation": np.nanmean(pred_t_corr_reshape),
                                  "pattern_correlation": np.nanmean(pattern_corr_values),}, index=[index_df])
        filename = f'output-{output_label}/skill_correlation-{predictand.var}-opt-sim.csv'
        with open(filename, 'a') as f:
            df_parameters_opt.to_csv(f, header=f.tell() == 0)
            index_df +=1
        return time_correlation
        
    # prob = LpProblem("forecast_Problem",LpMaximize)
    # opt_m = LpVariable.dicts("Optimization_Method",indexs=["om"], lowBound=0, upBound=1, cat='Continuous')
    # nr_batch_size = LpVariable.dicts("Batch_size",indexs=["bs"], lowBound=1, upBound=64, cat='Integer')
    # lr_rate = LpVariable.dicts("Learning_rate",indexs=["lr"], lowBound=0.000001, upBound=0.1, cat='Continuous')
    # nr_layers = LpVariable.dicts("Layers",indexs=["l"], lowBound=1, upBound=16, cat='Integer')
    # nr_neurons = LpVariable.dicts("Neurons",indexs=["n"], lowBound=5, upBound=16, cat='Integer')
    # prob += objective(opt_m, nr_batch_size, lr_rate, nr_layers, nr_neurons)
    # prob.solve(timeLimit=86400)

    # # Create the mip solver with the CBC backend.
    # solver = pywraplp.Solver('simple_mip_program',
    #                          pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
    # infinity = solver.infinity()
    # # x and y are integer non-negative variables.
    # opt_m = solver.IntVar(0, 1, 'opt_m')
    # nr_batch_size = solver.IntVar(1, 64, 'nr_batch_size')
    # lr_rate = solver.NumVar(0.000001, 0.1, 'lr_rate')
    # nr_layers = solver.IntVar(1, 16, 'nr_layers')
    # nr_neurons = solver.IntVar(5, 16, 'nr_neurons')
    # print('Number of variables =', solver.NumVariables())
    #
    # solver.Maximize(objective(opt_m, nr_batch_size, lr_rate, nr_layers, nr_neurons))
    #
    # status = solver.Solve()
    #
    # if status == pywraplp.Solver.OPTIMAL:
    #     print('Solution:')
    #     print('Objective value =', solver.Objective().Value())
    #     print('opt_m =', opt_m.solution_value())
    #     print('nr_batch_size =', nr_batch_size.solution_value())
    #     print('lr_rate =', lr_rate.solution_value())
    #     print('nr_layers =', nr_layers.solution_value())
    #     print('nr_neurons =', nr_neurons.solution_value())
    # else:
    #     print('The problem does not have an optimal solution.')
    #
    # print('\nAdvanced usage:')
    # print('Problem solved in %f milliseconds' % solver.wall_time())
    # print('Problem solved in %d iterations' % solver.iterations())
    # print('Problem solved in %d branch-and-bound nodes' % solver.nodes())









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
