#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 14:57:23 2019

@author: sonja
"""
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import numpy as np

class MixtureGaussianModel():
    """ Create a mixture gaussian model"""
    def __init__(self, gaussian_distributions):
        """ add all models by given mean and sigma as well as dimensions of data
        and number of models
        """
        self.submodels = []
        self.len_models = 0
        self.dim=len(gaussian_distributions[0]["mean"])
        for dist in gaussian_distributions:
            self.submodels.append(multivariate_normal(dist["mean"], dist["sigma"]))
            self.len_models +=1
            


    def rvs(self, size):
        """ get samples of mixture gaussian model"""
        self.submodel_samples = np.zeros((size, self.dim))
        np.random.seed(0)
        self.submodel_choices = np.random.randint(self.len_models, size=size)
        for idx,sample in enumerate(self.submodel_choices):# random_state=12345
           self.submodel_samples[idx]=(self.submodels[sample].rvs())
        return self.submodel_samples






# unit tests: check sub_model choices
#             check numbers of submodel-generator

if __name__ == '__main__':
    # create gaussian characterics from which we want to sample
    gaussian_distributions = [
    {"mean": [-1, 1, 1, -1], "sigma": [[0.1,0.,0.,0.], [0.,0.1,0.,0.],[0.,0.,0.1,0.],[0.,0.,0.,0.1]]},
    {"mean": [-1, 0, 1, 1] , "sigma": [[0.1,0.,0.,0.], [0.,0.1,0.,0.],[0.,0.,0.1,0.],[0.,0.,0.,0.1]]},
    ]
    
    # create instance to get samples
    mgm = MixtureGaussianModel(gaussian_distributions)
    print(mgm.rvs(5))
    print(mgm.dim)