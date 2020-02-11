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
import sys
from scipy import stats


def train_test_split_pred(predictand, precursors, test_size=0.66, random_state=2019):
    np.random.seed(random_state)
    len_predicts = len(predictand.dict_pred_1D[predictand.var])
    len_test_data = int(len_predicts * test_size)
    selected_time_steps = np.random.choice(len_test_data, test_size, replace=False)
    y_train = {}
    X_train = {}
    y_test = {}
    X_test = {}

    for i in range(len_test_data):
        if i in selected_time_steps:
            y_train[predictand.var].append(predictand.dict_pred_1D[predictand.var][i])
            for prec in precursors.dict_precursors.keys():
                X_train[prec].append(precursors.dict_prec_1D[prec][i])
        else:
            y_test[predictand.var].append(predictand.dict_pred_1D[predictand.var][i])
            for prec in precursors.dict_precursors.keys():
                X_train[prec].append(precursors.dict_prec_1D[prec][i])
    return y_train, X_train, y_test, X_test

def main():
    logger.info("Start forecast model")

    # load inifile according to variable
    var = sys.argv[1]
    output_label = sys.argv[3]
    inifile = f"ini/clusters_America_{var}_Forecast.ini"
    predictand = Predictand(inifile, output_label)

    # load forecast-parameters
    method_name = 'ward'
    k = 5
    forecast = Forecast(inifile, k, method_name)
    logger.info("Clusters: " + str(forecast.k))
    diff = int(forecast.end_year) - int(forecast.beg_year)
    forecast_data = np.zeros((diff, predictand.dict_pred_1D[f"{predictand.var}"].shape[1]))
    pattern_corr_values = []

    # load precursors
    precursors = Precursors(inifile, output_label)
    all_precs_names = [x for x in precursors.dict_precursors.keys()]

    for forecast_predictands in forecast.list_precursors:
        # Calculate forecast for all years
        forecast.list_precursors = forecast_predictands

        # Create train and test dataset with an 66:33 split
        y_train, X_train, y_test, X_test = train_test_split_pred(predictand, precursors, test_size=0.66, random_state=2019)
        # Calculate clusters of precursors for var, by removing one year
        predictand.calculate_clusters_from_test_data(y_train, forecast.method_name, forecast.k,)

        # Calculate composites
        precursors.get_composites_data_1d_train_test(X_train, predictand.f, forecast.k, forecast.method_name,
                                          predictand.var)

        # Prediction
        for year in range(len(y_test)):
            forecast_temp = forecast.prediction(predictand.clusters, precursors.dict_composites, X_test, year)

            # Assign forecast data to array
            forecast_data[year - forecast.beg_year] = forecast_temp

        # Calculate pattern correlation
        pattern_corr_values.append(stats.pearsonr(forecast_data[year - forecast.beg_year],
                                                  predictand.dict_standardized_pred_1D[f"{predictand.var}"][
                                                      year - forecast.beg_year])[0])

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
            ex = ExportVarPlot()
            ex.save_plot_and_time_correlation(forecast.list_precursors, predictand, pred_t_corr_reshape,
                                              significance_corr_reshape, all_precs_names)


if __name__ == '__main__':
    import logging.config
    from config_dict import config

    logger = logging.getLogger(__name__)
    logging.config.dictConfig(config)
    logger.info("Start clustering program")
    main()
