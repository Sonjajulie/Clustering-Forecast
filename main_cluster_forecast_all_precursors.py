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


def train_test_split_pred(predictand, precursors, test_size=0.66, random_state=2019):
    np.random.seed(random_state)
    len_predicts = len(predictand.dict_pred_1D[predictand.var])
    len_test_data = int(len_predicts * test_size)
    selected_time_steps = np.random.choice(len_predicts, len_test_data, replace=False)
    y_train = {}
    X_train = {}
    y_test = {}
    X_test = {}

    for i in range(len_test_data):
        if i in selected_time_steps:
            y_train.setdefault(predictand.var, []).append(predictand.dict_pred_1D[predictand.var][i])
            for prec in precursors.dict_precursors.keys():
                X_train.setdefault(prec, []).append(precursors.dict_prec_1D[prec][i])
        else:
            y_test.setdefault(predictand.var, []).append(predictand.dict_pred_1D[predictand.var][i])
            for prec in precursors.dict_precursors.keys():
                X_test.setdefault(prec, []).append(precursors.dict_prec_1D[prec][i])
    return y_train, X_train, y_test, X_test


def main(cl_parser: ClusteringParser, cl_config: dict):
    logger.info("Start forecast model")

    # load inifile according to variable
    # load inifile according to variable
    var = cl_parser.arguments['predictand']
    inifile = cl_parser.arguments['inifile']
    output_label = cl_parser.arguments['outputlabel']
    output_path = cl_parser.arguments['outputpath']
    predictand = Predictand(inifile, output_path, output_label, cl_config)

    # load forecast-parameters
    method_name = 'ward'
    k = 5
    forecast = Forecast(inifile, cl_config, k, method_name)
    logger.info("Clusters: " + str(forecast.k))
    diff = int(forecast.end_year) - int(forecast.beg_year)
    forecast_data = np.zeros((diff, predictand.dict_pred_1D[f"{predictand.var}"].shape[1]))
    pattern_corr_values = []

    # load precursors
    precursors = Precursors(inifile, output_path, output_label, cl_config)
    all_precs_names = [x for x in precursors.dict_precursors.keys()]

    # Create train and test dataset with an 66:33 split
    y_train, X_train, y_test, X_test = train_test_split_pred(predictand, precursors, test_size=0.66, random_state=2019)
    # Calculate clusters of precursors for var, by removing one year
    predictand.calculate_clusters_from_test_data(y_train, forecast.method_name, forecast.k, )

    # Calculate composites
    precursors.get_composites_data_1d_train_test(X_train, predictand.f, forecast.k, forecast.method_name,
                                                 predictand.var)

    for forecast_predictands in forecast.list_precursors_combinations:
        # Calculate forecast for all years
        forecast.list_precursors = forecast_predictands
        # Prediction
        for year in range(len(y_test[predictand.var])):
            print(year)
            forecast_temp = forecast.prediction(predictand.clusters, precursors.dict_composites, X_test, year)

            # Assign forecast data to array
            forecast_data[year] = forecast_temp

            # Calculate pattern correlation
            pattern_corr_values.append(stats.pearsonr(forecast_data[year],
                                                      predictand.dict_standardized_pred_1D[f"{predictand.var}"][year])[0])

        # Calculate time correlation for each point
        time_correlation, significance = forecast.calculate_time_correlation(
            predictand.dict_standardized_pred_1D[f"{predictand.var}"],
            forecast_data, predictand.time_start_file)

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
                                              significance_corr_reshape, forecast.list_precursors_all)


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
