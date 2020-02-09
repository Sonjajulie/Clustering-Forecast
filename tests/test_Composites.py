import unittest
import numpy as np
import random
from classes.Composites import Composites


class TestComposites(unittest.TestCase):
    """ Create test class for composites"""
    def setUp(self):
        """initialize class composites"""
        inifile = "ini/composites_America_PSL.ini"
        self.composites = Composites(inifile)


class TestInit(TestComposites):
    def test_initial_composite_name(self):
        self.assertEqual(self.composites.config[self.composites.precs_sections[0]]['name'], "PSL")

    def test__calculate_significance(self):
        # idea: use 3 spatial and 100 time coords, in which for time steps 3,6,9,12,15,18,21
        # for index 0 and 2 are 1 with
        # noise in order to check bootstrap method
        key = "PSL"
        percent_boot = 90
        ik = 0
        k = 1
        self.t_end = 100
        self.spatial_coords = 3
        self.test_array = np.zeros((self.t_end, self.spatial_coords))

        for t in range(self.t_end):
            if t % 3:
                self.test_array[t, 0] = 1 + random.uniform(-0.1, 0.1)
                self.test_array[t, 1] = random.uniform(-0.1, 0.1)
                self.test_array[t, 2] = -1 + random.uniform(-0.1, 0.1)
            else:
                self.test_array[t] = [random.uniform(-0.1, 0.1) for _ in range(3)]

        self.time_steps = [t for t in range(100) if t % 3]
        self.composites.dict_standardized_precursors[key] = self.test_array
        self.composites.cluster_frequency = [len(self.time_steps)]
        self.composites.dict_composites[key] = np.zeros((1, self.spatial_coords))
        self.composites.dict_composites[key][ik] = np.mean(self.test_array[self.time_steps], axis=0)

        self.composites.composites_significance_x = {}
        self.composites.composites_significance_y = {}
        self.composites.composites_significance[key] = np.zeros((k, self.composites.dict_composites[key][ik].shape[0]))
        # Calculate end_n different randomly selected clusters to see whether our cluster is significant
        self.composites.end_n = 5000
        self.lons_all = [1, 2, 3]
        self.lats_all = [-1, -2, -3]
        self.composites.lons, self.composites.lats = np.meshgrid(self.lons_all, self.lats_all)
        self.composites.lats1 = np.reshape(self.composites.lons,
                                           [self.composites.dict_composites[key][ik].shape[0], -1])
        self.composites.lons1 = np.reshape(self.composites.lats,
                                           [self.composites.dict_composites[key][ik].shape[0], -1])
        # get time of our cluster for selecting different states
        self.composites.time_dim = self.t_end
        # initialize array for randomly selected clusters
        self.composites.bootstrap_arrays = np.zeros((self.composites.end_n,
                                                     self.composites.dict_standardized_precursors[key].shape[1]))
        self.composites._bootstrap_method(key, ik, percent_boot)
        try:
            np.testing.assert_array_equal(self.composites.composites_significance[key][0], [1., 0., 1.])
            res = True
        except AssertionError as err:
            res = False
            print(err)
        self.assertTrue(res)
