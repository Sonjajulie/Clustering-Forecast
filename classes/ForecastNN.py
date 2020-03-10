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
# from tensorflow.keras.constraints import MaxNorm
# from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense  # , Dropout, Activation
# from tensorflow.keras.layers import Embedding
# from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D
# from tensorflow.keras.datasets import imdb
# from sklearn.metrics import mean_squared_error
from classes.Forecast import Forecast
from pathlib import Path
# noinspection PyPep8Naming
import tensorflow.keras.backend as K
import tensorflow as tf


def wrapper_function_cluster_rmse(clusters_1d: np.array, observations: np.array, k: int, batch_size_in: int):
    """
    This is a wrapper function for a loss function because keras only accepts loss-functions with two input-parameter.
    Hence all other parameters must be set through this function (and then it magically works).
    :param clusters_1d: contain all k 1d clusters
    :param observations: contain all observational data
    :param k: number of clusters
    :param batch_size_in: size of batches, normally 32
    Problem is that keras accept only functions that have y_pred and y_true as arguments
    """

    def rmse(y_true, y_pred):
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
        def body(i_batch_body, batch_size_body, cl_rmse_body):
            # to cast from tf.float32 to tf.int32
            index = tf.dtypes.cast(y_true[i_batch_body, 0], tf.int32)
            # get observation for certain year and index of batch
            observations_year = tf.convert_to_tensor(observations, np.float32)[index]
            # create tensorflow array with dimension of len(observations_year)
            pred = tf.zeros(shape=tf.shape(observations_year), dtype="float32")
            clusters_1d_tf = tf.convert_to_tensor(clusters_1d, np.float32)

            # go through for-loop and add y_pred to tensorflow array
            for i in range(int(k)):
                cluster_i = tf.gather(clusters_1d_tf, i)
                y_pred_projection_beta = y_pred[i_batch_body, i]
                pred = pred + y_pred_projection_beta * cluster_i
                cl_rmse_body = tf.add(cl_rmse_body, K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)))
            # tf.print("pred:", [tf.shape(pred), pred], output_stream=sys.stdout)
            return [tf.add(i_batch_body, 1), batch_size_body, cl_rmse_body]

        # if condition for tf.while_loop
        # noinspection PyUnusedLocal
        def condition(i_batch_condition, batch_size_condition, cl_rmse_condition):
            return tf.less(i_batch_condition, batch_size_condition)

        # initialize of tf.while_loop parameters
        cl_rmse = tf.constant(0.)
        i_batch = tf.constant(0, dtype=np.int32)
        # call tf.while_loop according to the website:
        i_batch, batch_size, result = tf.while_loop(condition, body, [i_batch, batch_size, cl_rmse])

        # tf.print("result:", [tf.shape(result), result], output_stream=sys.stdout)
        # tf.truediv enforces python v3 division semantics
        return tf.truediv(result, tf.dtypes.cast(batch_size, tf.float32))

    return rmse

def wrapper_function_cluster_loss(clusters_1d: np.array, observations: np.array, k: int, batch_size_in: int):
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
        def body(i_batch_body, batch_size_body, cl_loss_body):
            # to cast from tf.float32 to tf.int32
            index = tf.dtypes.cast(y_true[i_batch_body, 0], tf.int32)
            # get observation for certain year and index of batch
            observations_year = tf.convert_to_tensor(observations, np.float32)[index]
            # create tensorflow array with dimension of len(observations_year)
            pred = tf.zeros(shape=tf.shape(observations_year), dtype="float32")
            clusters_1d_tf = tf.convert_to_tensor(clusters_1d, np.float32)

            # go through for-loop and add y_pred to tensorflow array
            for i in range(int(k)):
                cluster_i = tf.gather(clusters_1d_tf, i)
                y_pred_projection_beta = y_pred[i_batch_body, i]
                pred = pred + y_pred_projection_beta * cluster_i
                cl_loss_body = tf.add(cl_loss_body, K.mean(K.square(tf.squeeze(pred) - observations_year), axis=-1))
            # tf.print("pred:", [tf.shape(pred), pred], output_stream=sys.stdout)
            return [tf.add(i_batch_body, 1), batch_size_body, cl_loss_body]

        # if condition for tf.while_loop
        # noinspection PyUnusedLocal
        def condition(i_batch_condition, batch_size_condition, cl_loss_condition):
            return tf.less(i_batch_condition, batch_size_condition)

        # initialize of tf.while_loop parameters
        cl_loss = tf.constant(0.)
        i_batch = tf.constant(0, dtype=np.int32)
        # call tf.while_loop according to the website:
        i_batch, batch_size, result = tf.while_loop(condition, body, [i_batch, batch_size, cl_loss])

        # tf.print("result:", [tf.shape(result), result], output_stream=sys.stdout)
        # tf.truediv enforces python v3 division semantics
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

    # noinspection PyPep8Naming
    def train_nn(self, forecast_predictands: list, clusters_1d: dict, composites_1d: dict, X_train: dict,
                 y_train: np.array):
        """
        make train nn for forecast
        :param forecast_predictands: list contains predictands which should be used to forecast
        :param clusters_1d: dict of all k-clusters
        :param composites_1d: dict of composites with time and one-dimensional array
        :param X_train: np.ndarray with all  data of precursors
        :param y_train: np.ndarray with all  data of predictands
        """
        # Merge only those precursors which were selected
        self.selected_composites_1D = np.hstack([composites_1d[i] for i in forecast_predictands])
        self.selected_data_1d = np.hstack([X_train[i] for i in forecast_predictands])
        self.clusters_1d = clusters_1d

        # Calculate projection coefficients for all points in training set
        self.len_selected_data = len(self.selected_data_1d)
        self.alpha_all_years = np.zeros((self.len_selected_data, self.k))
        self.alpha_matrix = np.zeros((self.k, self.k))
        for year in range(self.len_selected_data):
            self.alpha_all_years[year] = self._projection_coefficients_year(year)

        self.alphas_train, self.alphas_val, self.y_train_pseudo, self.y_val_pseudo = \
            self.train_test_split_nn(self.k, self.alpha_all_years, y_train)
        self.logger.info('Create network (model): specify number of neurons in each layer:')
        self.nr_neurons = 16  # self.k
        np.random.seed(3456)
        print(self.k)
        self.model = Sequential()  # kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.02, seed=None),
        # bias_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.02, seed=None)

        lr_rate = 0.1  # default: 0.0001

        # input should be the alphas, right? and output should be the betas!!!!

        self.model.add(Dense(self.nr_neurons, input_dim=len(self.alphas_train[0]), activation='relu', ))
        self.model.add(Dense(self.nr_neurons, activation='relu', kernel_regularizer=regularizers.l2(lr_rate)))
        self.model.add(Dense(self.nr_neurons, activation='relu', kernel_regularizer=regularizers.l2(lr_rate)))
        self.model.add(Dense(self.nr_neurons, activation='relu', kernel_regularizer=regularizers.l2(lr_rate)))
        self.model.add(Dense(self.nr_neurons, activation='relu', kernel_regularizer=regularizers.l2(lr_rate)))
        # self.model.add(Dense(self.k, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
        self.model.add(Dense(self.k, activation='linear'))

        wr_cl_loss = wrapper_function_cluster_loss(np.asarray(self.clusters_1d), np.asarray(y_train), self.k,
                                                   self.batch_size)
        self.model.compile(
            #        loss='categorical_crossentropy' wr_cl_loss # mean_squared_error  # lr=0.2,decay=0.001
            loss=wr_cl_loss, optimizer=optimizers.Adam(),metrics=['mean_squared_error'])
        # wrapper_function_cluster_loss(np.asarray(self.clusters_1d) np.asarray(y_train), self.k)

        #    model.compile(optimizer = "Adam",loss="binary_crossentropy",
        #    metrics=["accuracy"])mean_squared_error

        # Check the sizes of all newly created datasets
        logging.debug("Shape of x_train:", self.alphas_train.shape)
        logging.debug("Shape of x_val:", self.alphas_val.shape)
        logging.debug("Shape of y_train_pseudo:", self.y_train_pseudo.shape)
        logging.debug("Shape of y_val_pseudo:", self.y_val_pseudo.shape)

        logging.info('Train (fit) the network...')
        # filepath = f"ModelWeights-{self.nr_neurons}.hdf5"
        # checkpoint = ModelCheckpoint(filepath, save_best_only=True,
        #                              monitor="val_mean_squared_error")

        # ,callbacks = [checkpoint]

        out = self.model.fit(self.alphas_train, self.y_train_pseudo,
                             validation_data=(self.alphas_val, self.y_val_pseudo), epochs=100,
                             batch_size=self.batch_size, verbose=0)  #
        # , epochs=500, batch_size=30, verbose=0, callbacks=[mcp]verbose=0)

        # Saves the entire model into a file named as  'dnn_model.h5'
        file_path = f'{self.output_path}/output-{self.output_label}/' \
                    f'{self.var} - {str(len(forecast_predictands))}-precursor/model/'
        Path(file_path).mkdir(parents=True, exist_ok=True)
        self.model.save(f'{file_path}/dnn_model-{self.nr_neurons}_{self.k}_cluster_{forecast_predictands}.h5')
        self.logger.info('initial loss=' + repr(out.history["loss"][1]) + ', final=' + repr(out.history["loss"][-1]))
        self.logger.info(repr(out))

        file_path = f'{self.output_path}/output-{self.output_label}/' \
                    f'{self.var} - {str(len(forecast_predictands))}-precursor/progress/'
        Path(file_path).mkdir(parents=True, exist_ok=True)
        # plot progress of optimization
        if 1:
            fig, axs = plt.subplots(3, figsize=(12, 18))
            # plt.figure(1, figsize=(12, 6));
            # fig.suptitle("Model's Training & Validation process across epochs")
            # axs[0].set_title("Model's Training & Validation across epochs")
            axs[0].legend(['log(loss)'])
            axs[0].semilogy(out.history["loss"][:])
            axs[0].semilogy(out.history["val_loss"][:])
            axs[0].legend(['Train', 'Validation'], loc='upper right')
            axs[0].set(ylabel='$\log_{10}$(loss)')


            axs[1].plot(self.model.history.history['loss'])
            axs[1].plot(self.model.history.history['val_loss'])
            # axs[1].set_title("Model's Training & Validation mean squared error across epochs")
            axs[1].set(ylabel='Loss')
            axs[1].legend(['Train', 'Validation'], loc='upper right')

            axs[2].plot(self.model.history.history['mean_squared_error'])
            axs[2].plot(self.model.history.history['val_mean_squared_error'])
            # axs[2].set_title("Model's Training & Validation mean squared error across epochs")
            axs[2].set(xlabel='Epochs', ylabel='mean squared diff between alphas & betas' )
            axs[2].legend(['Train', 'Validation'], loc='upper right')
            plt.savefig(f"{file_path}/progress_wr_cl_loss_{self.nr_neurons}_neurons_{self.k}_cluster_"
                        f"{forecast_predictands}.pdf", bbox_inches='tight')


    # noinspection PyPep8Naming
    def train_nn_opt(self, forecast_predictands: list, clusters_1d: dict, composites_1d: dict, X_train: dict,
                 y_train: np.array, nr_neurons: int, opt_method: str, nr_epochs: int, nr_layers: int, lr_rate: np.float,
                     nr_batch_size: int):
        """
        optimize forecast by using different parameters to train nn for forecast
        :param forecast_predictands: list contains predictands which should be used to forecast
        :param clusters_1d: dict of all k-clusters
        :param composites_1d: dict of composites with time and one-dimensional array
        :param X_train: np.ndarray with all  data of precursors
        :param y_train: np.ndarray with all  data of predictands
        :param nr_neurons: number of neurons in each layer
        :param opt_method: optimization method
        :param nr_epochs: number of epochs
        :param nr_layers: number of layers
        :param lr_rate: learning rate
        """
        # Merge only those precursors which were selected
        self.selected_composites_1D = np.hstack([composites_1d[i] for i in forecast_predictands])
        self.selected_data_1d = np.hstack([X_train[i] for i in forecast_predictands])
        self.clusters_1d = clusters_1d

        # Calculate projection coefficients for all points in training set
        self.len_selected_data = len(self.selected_data_1d)
        self.alpha_all_years = np.zeros((self.len_selected_data, self.k))
        self.alpha_matrix = np.zeros((self.k, self.k))
        for year in range(self.len_selected_data):
            self.alpha_all_years[year] = self._projection_coefficients_year(year)

        self.alphas_train, self.alphas_val, self.y_train_pseudo, self.y_val_pseudo = \
            self.train_test_split_nn(self.k, self.alpha_all_years, y_train)
        self.logger.info('Create network (model): specify number of neurons in each layer:')
        self.nr_neurons = nr_neurons # self.k
        np.random.seed(3456)
        print(self.k)
        self.model = Sequential()  # kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.02, seed=None),
        # bias_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.02, seed=None)
        self.model.add(Dense(self.nr_neurons, input_dim=len(self.alphas_train[0]), activation='relu', ))
        for _ in range(nr_layers):
            self.model.add(Dense(self.nr_neurons, activation='relu', kernel_regularizer=regularizers.l2(lr_rate)))
        self.model.add(Dense(self.k, activation='linear'))

        wr_cl_loss = wrapper_function_cluster_loss(np.asarray(self.clusters_1d), np.asarray(y_train), self.k,
                                                   nr_batch_size)
        self.dict_optimizer = {"Adam": optimizers.Adam(), "SGD": optimizers.SGD(), "Adamax": optimizers.Adamax(),
                               "Nadam": optimizers.Nadam(),}
        self.model.compile(
            loss=wr_cl_loss, optimizer=self.dict_optimizer[opt_method],metrics=['mean_squared_error'])


        # Check the sizes of all newly created datasets
        logging.debug("Shape of x_train:", self.alphas_train.shape)
        logging.debug("Shape of x_val:", self.alphas_val.shape)
        logging.debug("Shape of y_train_pseudo:", self.y_train_pseudo.shape)
        logging.debug("Shape of y_val_pseudo:", self.y_val_pseudo.shape)

        logging.info('Train (fit) the network...')
        # filepath = f"ModelWeights-{self.nr_neurons}.hdf5"
        # checkpoint = ModelCheckpoint(filepath, save_best_only=True,
        #                              monitor="val_mean_squared_error")

        # ,callbacks = [checkpoint]

        out = self.model.fit(self.alphas_train, self.y_train_pseudo,
                             validation_data=(self.alphas_val, self.y_val_pseudo), epochs=nr_epochs,
                             batch_size=self.batch_size, verbose=0)  #
        # , epochs=500, batch_size=30, verbose=0, callbacks=[mcp]verbose=0)

        # Saves the entire model into a file named as  'dnn_model.h5'
        # {self.output_path}/
        file_path = f'{self.output_path}/output-{self.output_label}/' \
                    f'{self.var}-{str(len(forecast_predictands))}-precursor/model/'
        Path(file_path).mkdir(parents=True, exist_ok=True)
        self.model.save(f'{file_path}/dnn_model-{self.nr_neurons}_{self.k}_cluster_{forecast_predictands}.h5')
        self.logger.info('initial loss=' + repr(out.history["loss"][1]) + ', final=' + repr(out.history["loss"][-1]))
        self.logger.info(repr(out))

        file_path = f'{self.output_path}/output-{self.output_label}/' \
                    f'{self.var} - {str(len(forecast_predictands))}-precursor/progress/'
        Path(file_path).mkdir(parents=True, exist_ok=True)

        file_path = f'output-{self.output_label}/' \
                    f'{self.var}-{str(len(forecast_predictands))}-precursor/progress/'
        Path(file_path).mkdir(parents=True, exist_ok=True)
        # plot progress of optimization
        if 1:

            fig, axs = plt.subplots(3, figsize=(12, 18))
            # plt.figure(1, figsize=(12, 6));
            # fig.suptitle("Model's Training & Validation process across epochs")
            # axs[0].set_title("Model's Training & Validation across epochs")
            axs[0].legend(['log(loss)'])
            axs[0].semilogy(out.history["loss"][:])
            axs[0].semilogy(out.history["val_loss"][:])
            axs[0].legend(['Train', 'Validation'], loc='upper right')
            axs[0].set(ylabel='$\log_{10}$(loss)')


            axs[1].plot(self.model.history.history['loss'])
            axs[1].plot(self.model.history.history['val_loss'])
            # axs[1].set_title("Model's Training & Validation mean squared error across epochs")
            axs[1].set(ylabel='Loss')
            axs[1].legend(['Train', 'Validation'], loc='upper right')

            axs[2].plot(self.model.history.history['mean_squared_error'])
            axs[2].plot(self.model.history.history['val_mean_squared_error'])
            # axs[2].set_title("Model's Training & Validation mean squared error across epochs")
            axs[2].set(xlabel='Epochs', ylabel='mean squared diff between alphas & betas' )
            axs[2].legend(['Train', 'Validation'], loc='upper right')
            plt.savefig(f"{file_path}/progress_wr_cl_loss_{self.nr_neurons}_neurons_{self.k}_cluster_"
                        f"{forecast_predictands}.pdf", bbox_inches='tight')
            plt.close('all')

    # noinspection PyPep8Naming
    def train_nn_opt(self, forecast_predictands: list, clusters_1d: dict, composites_1d: dict, X_train: dict,
                 y_train: np.array, nr_neurons: int, opt_method: str, nr_epochs: int, nr_layers: int, lr_rate: np.float,
                     nr_batch_size: int):
        """
        optimize forecast by using different parameters to train nn for forecast
        :param forecast_predictands: list contains predictands which should be used to forecast
        :param clusters_1d: dict of all k-clusters
        :param composites_1d: dict of composites with time and one-dimensional array
        :param X_train: np.ndarray with all  data of precursors
        :param y_train: np.ndarray with all  data of predictands
        :param nr_neurons: number of neurons in each layer
        :param opt_method: optimization method
        :param nr_epochs: number of epochs
        :param nr_layers: number of layers
        :param lr_rate: learning rate
        """
        # Merge only those precursors which were selected
        self.selected_composites_1D = np.hstack([composites_1d[i] for i in forecast_predictands])
        self.selected_data_1d = np.hstack([X_train[i] for i in forecast_predictands])
        self.clusters_1d = clusters_1d

        # Calculate projection coefficients for all points in training set
        self.len_selected_data = len(self.selected_data_1d)
        self.alpha_all_years = np.zeros((self.len_selected_data, self.k))
        self.alpha_matrix = np.zeros((self.k, self.k))
        for year in range(self.len_selected_data):
            self.alpha_all_years[year] = self._projection_coefficients_year(year)

        self.alphas_train, self.alphas_val, self.y_train_pseudo, self.y_val_pseudo = \
            self.train_test_split_nn(self.k, self.alpha_all_years, y_train)
        self.logger.info('Create network (model): specify number of neurons in each layer:')
        self.nr_neurons = nr_neurons # self.k
        np.random.seed(3456)
        print(self.k)
        self.model = Sequential()  # kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.02, seed=None),
        # bias_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.02, seed=None)
        self.model.add(Dense(self.nr_neurons, input_dim=len(self.alphas_train[0]), activation='relu', ))
        for _ in range(nr_layers):
            self.model.add(Dense(self.nr_neurons, activation='relu', kernel_regularizer=regularizers.l2(lr_rate)))
        self.model.add(Dense(self.k, activation='linear'))

        wr_cl_loss = wrapper_function_cluster_loss(np.asarray(self.clusters_1d), np.asarray(y_train), self.k,
                                                   nr_batch_size)
        self.dict_optimizer = {"Adam": optimizers.Adam(), "SGD": optimizers.SGD(), "Adamax": optimizers.Adamax(),
                               "Nadam": optimizers.Nadam(),}
        self.model.compile(
            loss=wr_cl_loss, optimizer=self.dict_optimizer[opt_method],metrics=['mean_squared_error'])


        # Check the sizes of all newly created datasets
        logging.debug("Shape of x_train:", self.alphas_train.shape)
        logging.debug("Shape of x_val:", self.alphas_val.shape)
        logging.debug("Shape of y_train_pseudo:", self.y_train_pseudo.shape)
        logging.debug("Shape of y_val_pseudo:", self.y_val_pseudo.shape)

        logging.info('Train (fit) the network...')
        # filepath = f"ModelWeights-{self.nr_neurons}.hdf5"
        # checkpoint = ModelCheckpoint(filepath, save_best_only=True,
        #                              monitor="val_mean_squared_error")

        # ,callbacks = [checkpoint]

        out = self.model.fit(self.alphas_train, self.y_train_pseudo,
                             validation_data=(self.alphas_val, self.y_val_pseudo), epochs=nr_epochs,
                             batch_size=self.batch_size, verbose=0)  #
        # , epochs=500, batch_size=30, verbose=0, callbacks=[mcp]verbose=0)

        # Saves the entire model into a file named as  'dnn_model.h5'
        file_path = f'{self.output_path}/output-{self.output_label}/' \
                    f'{self.var}-{str(len(forecast_predictands))}-precursor/model/'
        Path(file_path).mkdir(parents=True, exist_ok=True)
        self.model.save(f'{file_path}/dnn_model-{self.nr_neurons}_{self.k}_cluster_{forecast_predictands}.h5')
        self.logger.info('initial loss=' + repr(out.history["loss"][1]) + ', final=' + repr(out.history["loss"][-1]))
        self.logger.info(repr(out))

        file_path = f'{self.output_path}/output-{self.output_label}/' \
                    f'{self.var}-{str(len(forecast_predictands))}-precursor/progress/'
        Path(file_path).mkdir(parents=True, exist_ok=True)
        # plot progress of optimization
        if 1:
            fig, axs = plt.subplots(3, figsize=(12, 18))
            # plt.figure(1, figsize=(12, 6));
            # fig.suptitle("Model's Training & Validation process across epochs")
            # axs[0].set_title("Model's Training & Validation across epochs")
            axs[0].legend(['log(loss)'])
            axs[0].semilogy(out.history["loss"][:])
            axs[0].semilogy(out.history["val_loss"][:])
            axs[0].legend(['Train', 'Validation'], loc='upper right')
            axs[0].set(ylabel='$\log_{10}$(loss)')


            axs[1].plot(self.model.history.history['loss'])
            axs[1].plot(self.model.history.history['val_loss'])
            # axs[1].set_title("Model's Training & Validation mean squared error across epochs")
            axs[1].set(ylabel='Loss')
            axs[1].legend(['Train', 'Validation'], loc='upper right')

            axs[2].plot(self.model.history.history['mean_squared_error'])
            axs[2].plot(self.model.history.history['val_mean_squared_error'])
            # axs[2].set_title("Model's Training & Validation mean squared error across epochs")
            axs[2].set(xlabel='Epochs', ylabel='mean squared diff between alphas & betas' )
            axs[2].legend(['Train', 'Validation'], loc='upper right')
            plt.savefig(f"{file_path}/progress_wr_cl_loss_{self.nr_neurons}_neurons_{self.k}_cluster_"
                        f"{forecast_predictands}.pdf", bbox_inches='tight')
            plt.close('all')
            
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
        self.beta_all = self.model.predict(self.alpha_all)[0]

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
