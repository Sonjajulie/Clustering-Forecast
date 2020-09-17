#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 11:34:43 2020

@author: sonja
"""

# noinspection PyUnresolvedReferences
import os
import matplotlib as mpl

if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
from logging import config
import logging
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
# from tensorflow.keras import activations
# Tensor flow libraries
# from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
from tensorflow.keras import initializers
# from tensorflow.keras.constraints import MaxNorm
# from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout  # , Activation

from tensorflow.keras.layers import Dense  # , Dropout, Activation
from tensorflow.keras.layers import Dense, Dropout    #, Activation

# from tensorflow.keras.layers import Embedding
# from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D
# from tensorflow.keras.datasets import imdb
# from sklearn.metrics import mean_squared_error
from classes.Forecast import Forecast
from pathlib import Path
# noinspection PyPep8Naming
import tensorflow.keras.backend as K
import tensorflow as tf

from scipy import stats
import pandas as pd
import tensorflow_probability as tfp
import sys

import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker

sns.set()


def wrapper_function_cluster_taylor(clusters_1d: np.array, observations: np.array, k: int, batch_size_in: int):
    """
    This is a wrapper function for a loss function because keras only accepts loss-functions with two input-parameter.
    Hence all other parameters must be set through this function (and then it magically works).
    :param clusters_1d: contain all k 1d clusters
    :param observations: contain all observational data
    :param k: number of clusters
    :param batch_size_in: size of batches, normally 32
    Problem is that keras accept only functions that have y_pred and y_true as arguments
    """

    def loss(y_true, y_pred):
        """
        define own loss function in order to get the correct projection coefficients of our predictand. In this case
        we calculate the mean square error function of the output
        :param y_pred: forecast calculated with the betas, which we get from the NN as input()
        :param y_true: This is a k-len vector with the first index showing the number for the observational data
        """

        # seems like that first time this function is called y_true is None, there for this workaround
        batch_size = tf.shape(y_true)[0]
        if batch_size is None:
            batch_size = tf.constant(batch_size_in, dtype=np.int32)

        # body for tf.while_loop
        def body(i_batch_body, batch_size_body, cl_loss_body, pred_corr_body, obs_corr_body):

            # to cast from tf.float32 to tf.int32
            index = tf.dtypes.cast(y_true[i_batch_body, 0], tf.int32)
            # get observation for certain year and index of batch
            observations_year = tf.convert_to_tensor(observations, np.float32)[index]
            obs_corr_body = tf.cond( tf.math.equal(i_batch_body, 0) , lambda:  tf.expand_dims(observations_year, 0), lambda: tf.concat([obs_corr_body, tf.expand_dims(observations_year, 0)], axis=0))

            # create tensorflow array with dimension of len(observations_year)
            pred = tf.zeros(shape=tf.shape(observations_year), dtype="float32")
            clusters_1d_tf = tf.convert_to_tensor(clusters_1d, np.float32)

            # go through for-loop and add y_pred to tensorflow array
            for i in range(int(k)):
                cluster_i = tf.gather(clusters_1d_tf, i)
                y_pred_projection_beta = 1 + y_pred[i_batch_body, i]
                # tf.print("y_pred_projection_beta:", [tf.shape(y_pred_projection_beta), y_pred_projection_beta],
                #          output_stream=sys.stdout)
                pred = pred + y_pred_projection_beta * cluster_i


            # calculate taylor skill
            cl_corr = tfp.stats.correlation(pred, observations_year, sample_axis=0, event_axis=None)
            cl_std_obs = tfp.stats.stddev(observations_year)
            cl_std_mod = tfp.stats.stddev(pred)
            ax = cl_std_mod / cl_std_obs
            # tf.print("cl_std_obs:", [tf.shape(cl_std_obs), cl_std_obs],output_stream=sys.stdout)
            # tf.print("cl_std_mod:", [tf.shape(cl_std_mod), cl_std_mod],output_stream=sys.stdout)
            # tf.print("cl_corr:", [tf.shape(cl_corr), cl_corr],output_stream=sys.stdout)
            cl_taylor =  (4 * tf.pow((1 + cl_corr), 4)) / (16 * tf.pow((ax + 1 / ax), 2))
            # tf.print("cl_taylor:", [tf.shape(cl_taylor), cl_taylor],output_stream=sys.stdout)

            corr = tf.cond(tf.math.is_finite(cl_taylor), lambda: 1 - cl_taylor, lambda: 1.)

            # pred_corr_body = tf.cond( tf.math.equal(i_batch_body, 0) , lambda:  tf.expand_dims(pred, 0), lambda: tf.concat([pred_corr_body, tf.expand_dims(pred, 0)], axis=0))
            cl_loss_body = tf.add(cl_loss_body, corr)
            # tf.print("cl_loss_body:", [tf.shape(cl_loss_body), cl_loss_body], output_stream=sys.stdout)
            return [tf.add(i_batch_body, 1), batch_size_body, cl_loss_body, pred_corr_body, obs_corr_body]


        # if condition for tf.while_loop
        # noinspection PyUnusedLocal
        def condition(i_batch_condition, batch_size_condition, cl_loss_condition, pred_corr, obs_corr):
            return tf.less(i_batch_condition, batch_size_condition)

        # initialize of tf.while_loop parameters
        cl_loss = tf.constant(0.)
        i_batch = tf.constant(0, dtype=np.int32)

        pred_corr_in = tf.convert_to_tensor(observations, np.float32)[0]
        obs_corr_in = tf.convert_to_tensor(observations, np.float32)[0]
        # call tf.while_loop according to the website:
        i_batch, batch_size, result, pred_corr, obs_corr = tf.while_loop(condition, body, [i_batch, batch_size, cl_loss,  pred_corr_in, obs_corr_in],
                                                                                shape_invariants=[i_batch.get_shape(),batch_size.get_shape(),cl_loss.get_shape(),
                                                                                                  tf.TensorShape(None), tf.TensorShape(None)] # pred_corr.get_shape()
                                                                               )

        # corr = tfp.stats.correlation(obs_corr, pred_corr, sample_axis=0, event_axis=None)
        # corr1d = tf.identity(tf.reshape(corr, [-1]))
        return tf.truediv(result, tf.dtypes.cast(batch_size, tf.float32))
    return loss





class ForecastNN(Forecast):
    def __init__(self, inifile_in: str, output_path: str, output_label: str, cl_config: dict, var: str, k=8,
                 method_name="ward"):
        """
        Initialize Forecast--> read forecast_nn parameters using ini-file
        :param inifile_in: file for initialization of variable
        :param output_path: path, where output should be saved
        :param output_label: name of output folder
        :param var: name of predictand, important for saving images in correct folder
        :param cl_config: dictionary, where all information of logger is stored from classes/config
        :param k: number of clusters
        :param method_name: method for clustering data
        """
        # initialize data
        self.alpha_all_years = None
        self.alphas_train = None
        self.alphas_val = None
        self.y_train_pseudo = None
        self.y_val_pseudo = None
        self.beta_all = None
        self.len_selected_data = None
        self.model = None
        self.nr_neurons = None
        self.y_pred = None
        self.clusters_1d = None
        self.batch_size = 32
        self.var = var
        super().__init__(inifile_in, cl_config, k, method_name)
        logging.config.dictConfig(cl_config)
        self.logger = logging.getLogger(__name__)
        self.logger.info('Read ini-file')
        self.output_path = output_path
        self.output_label = output_label
        self.index_df = 0

    @staticmethod
    def train_test_split_nn(k: int, alphas: np.array, y_train: np.array, validate_size=0.1, random_state=2019):
        """
        calculate number of train and val points
        : param k: cluster number
        : param alphas: projection values, calculated by cluster forecast algorithm
        : param y_train: predictand projection parameters, wich we would like to forecast
        : param test_size: how much data should be used for validate data
        : param random_state: which seed should be used
        """
        np.random.seed(random_state)
        len_predicts = len(y_train)
        len_y = k
        len_train_data = int(len_predicts * (1 - validate_size))
        selected_time_steps = np.random.choice(len_predicts, len_train_data, replace=False)

        y_train_pseudo = []
        y_val_pseudo = []
        alphas_train = []
        alphas_val = []
        test_nr = 0
        val_nr = 0
        for i in range(len_predicts):
            if i in selected_time_steps:
                y_train_pseudo.append([0 for _ in range(len_y)])
                y_train_pseudo[test_nr][0] = i
                alphas_train.append(alphas[i].tolist())
                test_nr += 1
            else:
                y_val_pseudo.append([0 for _ in range(len_y)])
                y_val_pseudo[val_nr][0] = i
                alphas_val.append(alphas[i].tolist())
                val_nr += 1
        return np.asarray(alphas_train, dtype=np.float32), np.asarray(alphas_val, dtype=np.float32), \
               np.asarray(y_train_pseudo, dtype=np.float32), np.asarray(y_val_pseudo, dtype=np.float32)

    def prediction_nn(self, forecast_predictands: list, clusters_1d: dict, composites_1d: dict, data_year_1d: dict,
                      year: int):
        """make forecast_nn
        :param forecast_predictands: list contains predictands which should be used to forecast
        :param clusters_1d: dict of all k-clusters
        :param composites_1d: dict of composites with time and one-dimensional array
        :param data_year_1d: np.ndarray with all  data of precursors
        :param year: year which should be forecasted
        """
        # Merge only those precursors which were selected
        self.selected_composites_1D = np.hstack([composites_1d[i] for i in forecast_predictands])
        self.selected_data_1d = np.hstack([data_year_1d[i] for i in forecast_predictands])

        # need those additional brackets, because it can be multiple input data and the outer brackets indicate the
        # the length of the input data
        self.alpha_all = np.asarray([self._projection_coefficients_year(year)])
        # Calculate projection coefficients
        self.beta_all = np.zeros(self.k)

        self.beta_all = self.model_ta.predict(self.alpha_all)[0]
        print(f"self.beta_all: {self.beta_all}")
        print(f"self.alpha_all: {self.alpha_all}")

        self.forecast_var = np.zeros(len(clusters_1d[0]))
        for i in range(int(self.k)):
            self.forecast_var += self.beta_all[i] * clusters_1d[int(i)]
        return self.forecast_var


    def calc_alphas_for_talos(self, X_train: dict, y_train: np.array, params):
        """
        calculate alpha values for NN training
        :param X_train: np.ndarray with all  data of precursors
        :param y_train: np.ndarray with all  data of predictands
        :param params: dictionary of all tunable hyper parameters
        """
        # Merge only those precursors which were selected
        print(params["forecast_predictands"])
        print(params["composites_1d"])
        self.selected_composites_1D = params["composites_1d"][params["forecast_predictands"][0]]

        self.clusters_1d = params["clusters_1d"]
        self.alpha_matrix = np.zeros((len(params["clusters_1d"]), len(params["clusters_1d"])))
        self.len_selected_data = len(X_train[params["forecast_predictands"][0]])
        print("X_train[params[forecast_predictands][0]].shape")
        print(X_train[params["forecast_predictands"][0]].shape)

        self.selected_composites_1D = params["composites_1d"][params["forecast_predictands"][0]]
        print("self.selected_composites_1D")
        print(self.selected_composites_1D)
        # Calculate projection coefficients for all points in training set
        print(self.k)
        print(self.len_selected_data)
        self.alpha_all_years = np.zeros((self.len_selected_data, self.k))
        print("self.composite normalization")
        print(np.std(self.selected_composites_1D[0]))
        print(np.std(self.selected_composites_1D[1]))
        print(np.std(self.selected_composites_1D[2]))
        print(np.std(self.selected_composites_1D[3]))
        print(np.std(self.selected_composites_1D[4]))

        for year in range(self.len_selected_data):
            self.selected_data_1d = X_train[params["forecast_predictands"][0]][year]
            print("selected_data_1d normalization")
            print(np.std(self.selected_data_1d ))
            self.alpha_all = self._projection_coefficients()
            print("self.alpha_all")
            print(self.alpha_all)
            self.alpha_all_years[year] = self.alpha_all

        self.alphas_train, self.alphas_val, self.y_train_pseudo, self.y_val_pseudo = \
            self.train_test_split_nn(self.k, self.alpha_all_years, y_train)
        return self.alphas_train, self.alphas_val,self.y_train_pseudo, self.y_val_pseudo


        # # Merge only those precursors which were selected
        # self.selected_composites_1D = np.hstack([params["composites_1d"][i] for i in params["forecast_predictands"]])
        # self.selected_data_1d = np.hstack([X_train[i] for i in params["forecast_predictands"]])
        # self.clusters_1d = params["clusters_1d"]
        #
        # # Calculate projection coefficients for all points in training set
        # self.len_selected_data = len(self.selected_data_1d)
        # self.alpha_all_years = np.zeros((self.len_selected_data, self.k))
        # self.alpha_matrix = np.zeros((self.k, self.k))
        # for year in range(self.len_selected_data):
        #     self.alpha_all_years[year] = self._projection_coefficients_year(year)
        #     print(self.alpha_all_years[year])
        #
        # self.alphas_train, self.alphas_val, self.y_train_pseudo, self.y_val_pseudo = \
        #     self.train_test_split_nn(self.k, self.alpha_all_years, y_train)
        # return self.alphas_train, self.alphas_val,self.y_train_pseudo, self.y_val_pseudo


    # noinspection PyPep8Naming
    def train_nn_talos(self, X_train,  y_train, X_val, y_val, params):
        """
        optimize forecast by using different parameters to train nn for forecast
        :param forecast_predictands: list contains predictands which should be used to forecast
        :param X_train: np.ndarray with all  data of precursors
        :param y_train: np.ndarray with all  data of predictands
        :param params: dictionary of all tunable hyper parameters
        """
        self.alphas_train = X_train
        self.alphas_val= X_val
        self.y_train_pseudo = y_train
        self.y_val_pseudo = y_val
        print(params["precursor"])
        self.logger.info('Create network (model): specify number of neurons in each layer:')
        np.random.seed(3456)

        self.model_ta = Sequential()  # kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.02, seed=None),
        # bias_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.02, seed=None)

        # self.model_ta.add(Dense(params["first_neuron"], input_dim=self.k, activation=params['activation'],))
        self.model_ta.add(Dense(params["first_neuron"], activation=params['activation']))
        # ,
        # kernel_initializer = tf.keras.initializers.RandomNormal(mean=1.0, stddev=0.05,
        #                                                         seed=1),
        # bias_initializer = initializers.RandomNormal(mean=1.0, stddev=0.05, seed=1)
        # ,
        # kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5,
        #                                                         seed=1),
        # bias_initializer = initializers.RandomNormal(mean=0.0, stddev=0.5, seed=1)

        # kernel_initializer=params['kernel_initializer']
        # self.model_ta.add(Dropout(params['dropout']))
        # if we want to also test for number of layers and shapes, that's possible
        # https://github.com/autonomio/talos/blob/master/examples/A%20Very%20Short%20Introduction%20to%20Hyperparameter%20Optimization%20of%20Keras%20Models%20with%20Talos.ipynb
        # https: // autonomio.github.io / docs_talos /  # models
        # https: // github.com / autonomio / talos / blob / master / talos / model / hidden_layers.py
        # https://alphascientist.com/hyperparameter_optimization_with_talos.html
        for i in range(params['hidden_layers']):
            print(f"adding layer {i + 1}")
            self.model_ta.add(
                Dense(params["first_neuron"], input_dim=len(self.alphas_train[0]), activation=params['activation'],
                      kernel_initializer=params['kernel_initializer']))

        self.model_ta.add(
            Dense(self.k, activation=params['last_activation'], kernel_initializer=params['kernel_initializer']))

        # ~ wr_cl_loss = wrapper_function_cluster_loss(np.asarray(self.clusters_1d), np.asarray(params["y_train"]), self.k,
                                                   # ~ params['batch_size'])

        # wr_cl_loss = wrapper_function_cluster_corr(np.asarray(self.clusters_1d), np.asarray(params["y_train"]), self.k,
        #                                            params['batch_size'])

        wr_cl_loss = wrapper_function_cluster_taylor(np.asarray(self.clusters_1d), np.asarray(params["y_train"]), self.k,
                                                   params['batch_size'])

        self.model_ta.compile(
            loss=wr_cl_loss, optimizer=params['optimizer'],
            metrics=['mean_squared_error'])

        # noinspection PyAttributeOutsideInit
        # print(self.alphas_train[0:20])
        self.history = self.model_ta.fit(np.array(self.alphas_train), np.array(self.y_train_pseudo),
                                       epochs=params['epochs'],
                                      batch_size=params['batch_size'], verbose=0)  #validation_data=(self.alphas_val, self.y_val_pseudo),

        # Saves the entire model into a file named as  'dnn_model.h5'
        file_path = f'{self.output_path}/output-{self.output_label}/' \
                    f'{self.var}-{str(len(params["forecast_predictands"]))}-precursor/model/'
        Path(file_path).mkdir(parents=True, exist_ok=True)
        # self.model_ta.save(f'{file_path}/dnn_model-{params["first_neuron"]}_{self.k}_cluster_{params["precursor"]}.h5')
        self.logger.info('initial loss=' + repr(self.history.history["loss"][1]) + ', final=' + repr(
            self.history.history["loss"][-1]))
        self.logger.info(repr(self.history))



        # test data
        # Calculate forecast_nn for all years
        pattern_corr_values = []
        taylor_skill_values = []

        # Prediction
        # self.x_test = np.hstack([params["x_test"][i] for i in self.list_precursors_all])
        forecast_data = np.zeros((len(params["y_test"]),
                                  len(params["y_test"][1])))
        for year in range(len(params["y_test"])):  # len(y_test[predictand.var])):
            forecast_temp = self.prediction_nn(self.list_precursors_all, self.clusters_1d,
                                                      params["composites_1d"], params["x_test"], year)
            # Assign forecast_nn data to array
            forecast_data[year] = forecast_temp

            # Calculate pattern correlation and taylor skill
            if not np.isnan(forecast_temp).any():
                cl_corr = stats.pearsonr(forecast_temp, params["y_test"][year])[0]
                pattern_corr_values.append(cl_corr)
                cl_std_obs = np.std(params["y_test"][year])
                cl_std_mod = tfp.stats.stddev(forecast_temp)
                ax = cl_std_mod / cl_std_obs
                taylor_skill_values.append((4*(1 + cl_corr)) / ( 2 * (ax + 1/ax)**2) )# ((4*(1 + corr)**4) / ( 16 * (ax + 1/ax)**2))
            print(np.nanmean(forecast_temp))


        # Calculate time correlation for each point
        if not np.isnan(forecast_data).any():
            time_correlation, significance = self.calculate_time_correlation_all_times(
                np.array(params["y_test"]), forecast_data)
        else:
            time_correlation, significance = -1, 0
        params["time_corr"] = np.nanmean(time_correlation)
        params["pattern_corr"] = np.nanmean(pattern_corr_values)
        params["taylor_skill"] = np.nanmean(taylor_skill_values)
        df_parameters_opt = pd.DataFrame({
            "time_corr": np.nanmean(time_correlation), "pattern_corr": np.nanmean(pattern_corr_values),
            "taylor_skill" : np.nanmean(taylor_skill_values),
        "lr": params["lr"],
        "activation": "relu",
        "kernel_initializer": "random_uniform",
         "optimizer": params['optimizer'],
         "losses": "logcos",
         "shapes": "brick",
         'first neuron': params["first_neuron"],
        'forecast_predictands': self.list_precursors,
         'hidden_layers': params["hidden_layers"],
         'dropout': params["dropout"],
         'batch_size': params["batch_size"],
         'epochs': params["epochs"],

        }, index=[self.index_df])
        filename = f'{file_path}/skill_correlation-{self.var}-{params["forecast_predictands"]}-opt.csv'
        with open(filename, 'a') as f:
            df_parameters_opt.to_csv(f, header=f.tell() == 0)
            self.index_df +=1
        del df_parameters_opt
        return self.history, self.model_ta


        self.forecast_var = np.zeros(len(clusters_1d[0]))
        for i in range(int(self.k)):
            self.forecast_var += self.beta_all[i] * clusters_1d[int(i)]
        return self.forecast_var


    def _projection_coefficients_year(self, year):
        """ calculate projection coefficients"""
        # composite_i*composite_j
        for i in range(int(self.k)):
            for j in range(int(self.k)):
                # scalarproduct--> inner product
                self.alpha_matrix[i, j] = sum(self.selected_composites_1D[i, :] * self.selected_composites_1D[j, :])

        # # composite_i*y(t)
        rhs = np.zeros(self.k)
        for i in range(int(self.k)):
            rhs[i] = sum(self.selected_composites_1D[i, :] * self.selected_data_1d[year])
        #
        # # Singular values smaller (in modulus) than 0.01 * largest_singular_value (again, in modulus) are set to zero
        self.pinv_matrix = np.linalg.pinv(self.alpha_matrix, 0.01)
        return np.dot(self.pinv_matrix, rhs)
