import unittest
import numpy as np
from classes.MixtureModel import MixtureGaussianModel
from classes.PredictandToyModel import PredictandToyModel
from classes.Clusters import Clusters
from classes.Config import Config


class TestClusters(unittest.TestCase):
    """ Create test class for clusters"""
    def setUp(self):
        """initialize class cluster"""
        inifile = "/home/sonja/Documents/Clustering-Forecast/ini/clusters_America_prec_t.ini"
        output_path =  "/home/sonja/Documents/Clustering-Forecast/tests/"
        output_label =  "TEST"
        cl_config = Config("Test.log")
        self.clusters = Clusters(inifile, output_path, output_label, cl_config.config_dict)
        self.initialize_data()

    def initialize_data(self):
        """ initialize toy data to test algorithm"""
        # create data for the two different composites
        self.gaussian_distributions = [
            {"mean": [-1, 1, 1, -1],
             "sigma": [[0.01, 0., 0., 0.], [0., 0.01, 0., 0.], [0., 0., 0.01, 0.], [0., 0., 0., 0.01]]},
            {"mean": [-1, 0, 1, 1],
             "sigma": [[0.01, 0., 0., 0.], [0., 0.01, 0., 0.], [0., 0., 0.01, 0.], [0., 0., 0., 0.01]]}, ]

        # create time series
        self.t_end = 5000
        self.time_series = range(self.t_end)

        # create instance to get samples for sic and sce
        precursors = MixtureGaussianModel(self.gaussian_distributions)
        # get samples
        self.X = (precursors.rvs(self.t_end))

        # array which lead with composites to clusters pf PRCP
        self.array = np.array([[1, 2, 1, 1], [-0.5, 0, -0.5, 1.], [-1, 0, -1, -1]], np.float)
        self.prcp_clusters = [{"cluster": [ 1, -1, 1]}, {"cluster": [1, 1, -1]}]
        self.prcp = PredictandToyModel(self.prcp_clusters, self.array)
        self.y = self.prcp.get_data_from_precursors(self.X)
        self.method_name = "ward"
        self.k = 2
        # calculate data for toy model
        self.clusters.dict_standardized_pred_1D[self.clusters.var] = self.y
        self.clusters._set_method_name(self.method_name)
        self.clusters._set_k(self.k)
        self.clusters._set_linkage()
        self.clusters._set_f()
        self.clusters._set_clusters_1d()



class TestInit(TestClusters):

    def test_initial_composite_name(self):
        self.assertEqual(self.clusters.var, "prec_t")

    def test_list_equal(self):
        """Check number of result cluster"""
        self.expected =[[1, 1, -1],[ 1, -1, 1]]
        self.results = np.around(self.clusters.clusters, decimals=0).astype('int').tolist()
        self.assertListEqual(self.expected, self.results)

#
# # if __name__ == '__main__':
# #     unittest.main()