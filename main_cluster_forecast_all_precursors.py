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
from scipy import stats
import json


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
    forecast = Forecast(inifile, cl_config, k, method_name)
    logger.info("Clusters: " + str(forecast.k))

    # load precursors
    precursors = Precursors(inifile, output_path, output_label, cl_config)
    # Create train and test dataset with an 66:33 split
    # noinspection PyPep8Naming
    y_train, X_train, y_test, X_test = train_test_split_pred(predictand, precursors, data_range)
    # Calculate clusters of precursors for var, by removing one year
    predictand.calculate_clusters_from_test_data(y_train, forecast.method_name, forecast.k)

    # Calculate composites
    precursors.get_composites_data_1d_train_test(X_train, predictand.f, forecast.k, forecast.method_name,
                                                 predictand.var)
    # precursors.plot_composites(k, 1)
    # subtract train mean also for test data
    # for prec in forecast_nn.list_precursors_all:
    #     X_test[prec] -= precursors.varmean
    # y_test[predictand.var] -= predictand.varmean

    for forecast_predictands in forecast.list_precursors_combinations:
        # Calculate forecast_nn for all years
        forecast.list_precursors = forecast_predictands
        pattern_corr_values = []
        # Prediction
        forecast_data = np.zeros((len(y_test[f"{predictand.var}"]),
                                  predictand.dict_pred_1D[f"{predictand.var}"].shape[1]))
        logger.info(forecast_predictands)

        for year in range(len(y_test[predictand.var])):  # len(y_test[predictand.var])):
            # print(year)
            forecast_temp = forecast.prediction(predictand.clusters, precursors.dict_composites, X_test, year)

            # Assign forecast_nn data to array
            forecast_data[year] = forecast_temp

            # Calculate pattern correlation
            # remove zeros from array
            # forecast_temp = forecast_temp[forecast_temp != 0]
            # obs_temp = y_test[f"{predictand.var}"][year][y_test[f"{predictand.var}"][year] != 0]
            pattern_corr_values.append(stats.pearsonr(forecast_temp, y_test[f"{predictand.var}"][year])[0])

        # Calculate time correlation for each point
        time_correlation, significance = forecast.calculate_time_correlation_all_times(
            np.array(y_test[f"{predictand.var}"]), forecast_data)

        # Reshape correlation maps
        pred_t_corr_reshape = np.reshape(time_correlation, (predictand.dict_predict[predictand.var].shape[1],
                                                            predictand.dict_predict[predictand.var].shape[2]))
        significance_corr_reshape = np.reshape(significance, (predictand.dict_predict[predictand.var].shape[1],
                                                              predictand.dict_predict[predictand.var].shape[2]))

        logger.info(f'time correlation: {np.nanmean(pred_t_corr_reshape)}')
        logger.info(f'pattern correlation: {np.nanmean(pattern_corr_values)}')

        # Plot correlation map, if specified in ini-file
        if forecast.plot:
            logger.info("Plot and save variables")
            ex = ExportVarPlot(output_label, cl_config)
            ex.save_plot_and_time_correlation(forecast.list_precursors, predictand, pred_t_corr_reshape,
                                              significance_corr_reshape, forecast.list_precursors_all, np.nanmean(pred_t_corr_reshape))
            dict_skills_pattern[ex.predictor_names] = {'time correlation':  np.nanmean(pred_t_corr_reshape),
                                                       'pattern correlation': np.nanmean(pattern_corr_values)}
    if forecast.plot:
        with open(f'{output_path}/output-{output_label}/skill_correlation-{predictand.var}.json', 'w') as fp:
            json.dump(dict_skills_pattern, fp)


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
