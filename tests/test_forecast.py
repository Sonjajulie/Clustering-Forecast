import unittest
import numpy as np
import random
from classes.Predictand import Predictand
from classes.Config import Config
from classes.PredictandToyModel import PredictandToyModel
from classes.Precursors import Precursors
from classes.MixtureModel import MixtureGaussianModel
from classes.Forecast import Forecast
from scipy import stats
# 1.) create object of composite and cluster
# 2.) create forecast method with artificial data
# 3.)

class TestForecast(unittest.TestCase):
    """ Create test class for Forcast"""
    def setUp(self):
        """initialize class cluster and composites"""
        # cluster
        cl_inifile = "/home/sonja/Documents/Clustering-Forecast/ini/clusters_America_prec_t.ini"
        cl_output_path = "/home/sonja/Documents/Clustering-Forecast/tests/"
        cl_output_label = "TEST"
        cl_config = Config("Test.log")
        self.predictand = Predictand(cl_inifile, cl_output_path, cl_output_label, cl_config.config_dict)
        # composite
        co_inifile = "/home/sonja/Documents/Clustering-Forecast/ini/composites_America_PSL.ini"
        co_output_path =  "/home/sonja/Documents/Clustering-Forecast/tests/"
        co_output_label =  "TEST"
        co_config = Config("Test.log")
        self.precursors = Precursors(co_inifile, co_output_path, co_output_label, co_config.config_dict)

        # set cluster method parameters
        self.method_name = "ward"
        self.k = 2
        self.predictand_var = "prec_t"
        # initialize Forecast class
        self.forecast = Forecast(cl_inifile, cl_config.config_dict, self.k, self.method_name)

        self.initialize_data()

    def initialize_data(self):
        """ initialize toy data to test algorithm"""
        # create data for the two different composites
        # first two are snow data and second two data points are ice data
        self.gaussian_distributions = [
            {"mean": [-1, 1, 1, -1],
             "sigma": [[0.00001, 0., 0., 0.], [0., 0.00001, 0., 0.], [0., 0., 0.00001, 0.], [0., 0., 0., 0.00001]]},
            {"mean": [-1, 0, 1, 1],
             "sigma": [[0.00001, 0., 0., 0.], [0., 0.00001, 0., 0.], [0., 0., 0.00001, 0.], [0., 0., 0., 0.00001]]}, ]

        # create time series
        self.t_end = 5000
        self.time_series = range(self.t_end)

        # create instance to get samples for sic and sce
        precursors = MixtureGaussianModel(self.gaussian_distributions)
        # get samples
        self.X = (precursors.rvs(self.t_end))

        # array which lead with composites to clusters pf PRCP
        self.array = np.array([[1, 2, 1, 1], [-0.5, 0, -0.5, 1.], [-1, 0, -1, -1]], np.float)
        self.prcp_clusters = [{"cluster": [1, -1, 1]}, {"cluster": [1, 1, -1]}]
        self.prcp = PredictandToyModel(self.prcp_clusters, self.array)
        self.y = self.prcp.get_data_from_precursors(self.X)

        # set data to predictand input arrays
        self.predictand.dict_standardized_pred_1D[self.predictand.var] = self.y
        self.predictand.dict_pred_1D[self.predictand.var] = self.y

        # set data to precursors input data
        self.precursors.dict_precursors["snow"] = self.X[:,:2]
        self.precursors.dict_standardized_precursors["snow"] = self.X[:,:2]
        self.precursors.dict_prec_1D["snow"] = self.X[:,:2]
        self.precursors.dict_precursors["ice"] = self.X[:,2:]
        self.precursors.dict_standardized_precursors["ice"] = self.X[:,2:]
        self.precursors.dict_prec_1D["ice"] = self.X[:,2:]


        self.precursors.dict_standardized_precursors.pop("PSL")
        self.precursors.dict_prec_1D.pop("PSL")
        self.precursors.dict_precursors.pop("PSL")
        # Create train and test dataset with an 66:33 split
        self.y_train, self.X_train, self.y_test, self.X_test = self.train_test_split_pred(self.predictand, self.precursors,
                                                                      test_size=0.66, random_state=2019)

    def train_test_split_pred(self, predictand, precursors, test_size=0.66, random_state=2019):
        np.random.seed(random_state)
        len_predicts = len(predictand.dict_pred_1D[predictand.var])
        len_test_data = int(len_predicts * test_size)
        selected_time_steps = np.random.choice(len_predicts, len_test_data, replace=False)
        y_train = {}
        X_train = {}
        y_test = {}
        X_test = {}

        for i in range(len_predicts):
            if i in selected_time_steps:
                y_train.setdefault(predictand.var, []).append(predictand.dict_pred_1D[predictand.var][i])
                for prec in precursors.dict_precursors.keys():
                    X_train.setdefault(prec, []).append(precursors.dict_prec_1D[prec][i])
            else:
                y_test.setdefault(predictand.var, []).append(predictand.dict_pred_1D[predictand.var][i])
                for prec in precursors.dict_precursors.keys():
                    X_test.setdefault(prec, []).append(precursors.dict_prec_1D[prec][i])
        return y_train, X_train, y_test, X_test

    def calculate_clusters_and_composites(self):
        # Calculate clusters of precursors for var, by removing one year
        self.calculate_clusters_from_test_data(self.y_train, self.method_name, self.k)

        # Calculate composites
        self.precursors.get_composites_data_1d_train_test(self.X_train, self.predictand.f, self.k, self.method_name,
                                                     self.predictand_var)

    def calculate_forecast(self):
        """calculate forecast using toy model data"""
        self.calculate_clusters_and_composites()
        self.forecast.list_precursors_all = ["snow", "ice"]
        self.forecast.list_precursors_combinations = [["snow"], ["ice"], ["snow", "ice"]]

        self.forecast_data = np.zeros((len(self.y_test[self.predictand.var]), self.predictand.dict_pred_1D[f"{self.predictand.var}"].shape[1]))


        # go through different precursors
        for forecast_predictands in self.forecast.list_precursors_combinations:
            # Calculate forecast for all years
            self.forecast.list_precursors = forecast_predictands
            self.pattern_corr_values = []
            # Prediction
            for year in range(len(self.y_test[self.predictand.var])):  # len(y_test[predictand.var])):
                forecast_temp = self.forecast.prediction(self.predictand.clusters, self.precursors.dict_composites,
                                                         self.X_test, year)
                # Assign forecast data to array
                self.forecast_data[year] = forecast_temp

                # Calculate pattern correlation
                self.pattern_corr_values.append(round(stats.pearsonr(self.forecast_data[year],
                                                          self.y_test[self.predictand.var][year])[0]))

        # Calculate time correlation for each point
        for j in range(len(self.y_test[self.predictand.var])):
            for i in range(len(self.y_test[self.predictand.var][j])):
                self.y_test[self.predictand.var][j][i] = round(self.y_test[self.predictand.var][j][i])
                self.forecast_data[j][i] = round(self.forecast_data[j][i])

    def calculate_clusters_from_test_data(self, train_data: dict, method_name: str, k: int):
        """
        calculate clusters for predictand variable
        :param train_data: cluster data which should be used to calculate clusters
        :param method_name: name of the method used for clustering
        :param k: number of clusters
        """
        print('Calculate clusters')
        self.predictand.dict_standardized_pred_1D = train_data
        self.predictand._set_method_name(method_name)
        self.predictand._set_k(k)
        self.predictand._set_linkage()
        self.predictand._set_f()
        self.predictand._cluster_frequency()
        self.predictand._set_clusters_1d()


class TestInit(TestForecast):
    def test_initial_precursors_predictand_names(self):
        # First calculate clusters and composites and then check keys
        self.calculate_clusters_and_composites()
        # test whether names are correctly passed
        self.assertEqual(list(self.precursors.dict_composites.keys()), ["snow", "ice"])
        self.assertEqual(list(self.predictand.dict_standardized_pred_1D.keys())[0], "prec_t")

    def test_calculate_forecast_results(self):
        """ test whether toy model results lead to 100% forecast"""
        self.calculate_forecast()
        self.time_correlation, self.significance = \
            self.forecast.calculate_time_correlation(np.array(self.y_test[self.predictand.var], dtype=int),
                                                     np.array(self.forecast_data, dtype=int), 1)
        self.expected_time_correlation = [1, 1, 1]
        self.expected_significance = [0, 0, 0]
        self.assertListEqual(np.around(self.time_correlation, decimals=0).astype('int').tolist(),
                             self.expected_time_correlation)
        self.assertListEqual(np.around(self.significance, decimals=0).astype('int').tolist(),
                             self.expected_significance)
        # forecast of pattern correlation should be for each forecast 1
        # hence sum of all forecasts should be identical to the length of the forecasts
        self.assertEqual(sum(self.pattern_corr_values), len(self.y_test[self.predictand.var]))










































