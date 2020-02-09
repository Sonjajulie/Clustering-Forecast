#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 16:11:43 2019

@author: sonja
"""
import numpy as np

class PredictandToyModel:
    """ Class to create Precursor data """
 
    def __init__(self,clusters,array):
        """ 
        Initialize arrays, composites for data creation
        arrays - Arrays which lead with composites to clusters
        composites - composites of precursors
        """
        self.array = array
        self.clusters = []
        for cl in clusters:
            self.clusters.append(cl["cluster"])
        self.b_tensor = np.ones((array.shape[0],array.shape[1],array.shape[1]))

    def get_data_from_precursors(self,data):
        """ get predicand data from precursors"""
        self.data= np.zeros((data.shape[0],self.array.shape[0]))

        for idx,v in enumerate(data):
            
            # https://docs.scipy.org/doc/numpy/reference/generated/numpy.tensordot.html
            # tensor product of v v => matrix vv
            self.vv_matrix=np.tensordot(v,v, axes=0)
            # https://stackoverflow.com/questions/41870228/understanding-tensordot
            # in axes you define the dimensions which should be summed over
            # b_tensor sum over dims 1,2; vv_matrix sum over dims 0,1 
            # => dim 0 of b_tensor remains
            self.data[idx]=(self.array.dot(v)) # + np.tensordot(self.b_tensor,self.vv_matrix, axes=((1,2),(0,1)) )
        return self.data
    
    def get_data_point_from_precursors(self,data_point):
        """ return a single predicand point"""
        self.vv_matrix=np.tensordot(data_point,data_point, axes=0)
        return (self.array.dot(data_point)) + np.tensordot(self.b_tensor,self.vv_matrix, axes=((1,2),(0,1)) )





import unittest
class TestListElements(unittest.TestCase):
    def setUp(self):
        self.expected = [ 2.772124,2.489624 ,0.408124]
        self.result = list(map(lambda x: round(x,6),
                               prcp.get_data_point_from_precursors([-1.009,0.193,0.940,1.058])))

    def test_count_eq(self):
        """Check number of result list"""
        self.assertCountEqual(self.result, self.expected)

    def test_list_eq(self):
        """Check whether elements are the same"""
        self.assertListEqual(self.result, self.expected)
        
    
if __name__=='__main__':
    prcp_clusters = [{"cluster": [-1, 1, 1, -1]},{"cluster": [-1, 1, 1, -1]}]
    
    # array which lead with composites to clusters pf PRCP
    array = np.array([[1,2,1,1],[-0.5,0,-0.5,1.],[-1,0,-1,-1]], np.float)
    # arr_composite1 = [[1,1,1,0],[-0,0,-0.5,0.5],[-1,1,-1,0]]
    # arr_composite2 = [[0,1,0,1],[-0.5,0,0,0.5],[0,-1,0,-1]]
    prcp = Predicand(prcp_clusters,array)
    
    # example data snow ice:
    sce_sic = np.array([[-1.009,0.193,0.940,1.058],[-0.994,1.073,0.909,-1.039],[-0.979,1.062,1.034,-0.886],[-0.979,1.062,1.034,-0.886],[-0.979,1.062,1.034,-0.886]],dtype='float')
    
    print(prcp.get_data_from_precursors(sce_sic))
    unittest.main()

    
    
