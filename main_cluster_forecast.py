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
from scipy import stats
from classes.ClusteringParser import ClusteringParser
from classes.Config import Config
# usage TS ini/composites_America_ICEFRAC.ini ICEFRAC prec_t 6009 99


def main(cl_parser: ClusteringParser, cl_config: dict):
    logger.info("Start forecast_nn model")

    # load inifile according to variable
    inifile = cl_parser.arguments['inifile']
    output_label = cl_parser.arguments['outputlabel']
    predictand = Predictand(inifile, output_label, cl_config)
    # load forecast_nn-parameters
    method_name = 'ward'
    k = 5
    forecast = Forecast(inifile, cl_config, k, method_name)
    logger.info("Clusters: " + str(forecast.k))
    diff = int(forecast.end_year) - int(forecast.beg_year)
    forecast_data = np.zeros((diff, predictand.dict_pred_1D[f"{predictand.var}"].shape[1]))
    pattern_corr_values = []

    # load precursors
    precursors = Precursors(inifile, output_label, cl_config)
    all_precs_names = [x for x in precursors.dict_precursors.keys()]

    # Calculate forecast_nn for all years
    for year in range(int(forecast.beg_year), int(forecast.beg_year) + 3):  # int(forecast_nn.end_year)

        # Calculate clusters of precursors for var, by removing one year
        predictand.calculate_clusters_year(forecast.method_name, forecast.k, year - forecast.beg_year)

        # Calculate composites
        precursors.get_composites_data_1d_year(year - forecast.beg_year, predictand.f, forecast.k, forecast.method_name,
                                          predictand.var)

        # Prediction
        forecast_temp = forecast.prediction(predictand.clusters, precursors.dict_composites,
                                            precursors.dict_prec_1D, year - forecast.beg_year)

        # Assign forecast_nn data to array
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
        ex = ExportVarPlot(cl_config)
        ex.save_plot_and_time_correlation(forecast.list_precursors, predictand, pred_t_corr_reshape,
                                          significance_corr_reshape, all_precs_names)


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
