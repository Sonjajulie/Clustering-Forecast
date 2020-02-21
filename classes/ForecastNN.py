#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 11:34:43 2020

@author: sonja
"""
from logging import config
import logging
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt


# Tensor flow libraries
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
# from tensorflow.keras.constraints import MaxNorm
# from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense  # , Dropout, Activation
# from tensorflow.keras.layers import Embedding
# from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D
# from tensorflow.keras.datasets import imdb
from sklearn.metrics import mean_squared_error
from classes.Forecast import Forecast


class ForecastNN(Forecast):
    def __init__(self, inifile_in: str, cl_config: dict, k=8, method_name="ward"):
        """
        Initialize Forecast--> read forecast parameters using ini-file
        :param inifile_in: file for initialization of variable
        :param cl_config: dictionary, where all information of logger is stored from classes/config
        :param k: number of clusters
        :param method_name: method for clustering data
        """
        super().__init__(inifile_in, cl_config, k, method_name)
        logging.config.dictConfig(cl_config)
        self.logger = logging.getLogger(__name__)
        self.logger.info('Read ini-file')

        # initialize data
        self.alpha_all_years = None
        self.beta_all = None
        self.len_selected_data = None
        self.model = None
        self.nr_neurons = None
        self.y_pred = None
        self.clusters_1d = None

    def keras_custom_loss_function(self, betas_actual, y_true):
        """
        define own loss function in order to get the correct projection coefficients of our predictand. In this case
        we calculate the mean square error function of the output
        :param betas_actual: These are the projection coefficients for the forecast (which must be multiplied by
        the corresponding clusters
        :param y_true: This is the observational data of the predictand
        """
        self.y_pred = np.zeros(len(self.clusters_1d[0]))
        for i in range(int(self.k)):
            self.y_pred += betas_actual[i] * self.clusters_1d[int(i)]
        return mean_squared_error(y_true, self.y_pred)

    # noinspection PyPep8Naming
    def train_nn(self, clusters_1d: dict, composites_1d: dict, X_train: dict, y_train: int):
        """make forecast
        :param clusters_1d: dict of all k-clusters
        :param composites_1d: dict of composites with time and one-dimensional array
        :param X_train: np.ndarray with all  data of precursors
        :param y_train: np.ndarray with all  data of predictands
        """
        # Merge only those precursors which were selected
        self.selected_composites_1D = np.hstack([composites_1d[i] for i in self.list_precursors])
        self.selected_data_1d = np.hstack([X_train[i] for i in self.list_precursors])
        self.clusters_1d = clusters_1d

        # Calculate projection coefficients for all points in training set
        self.len_selected_data = len(self.selected_data_1d)
        self.alpha_all_years = np.zeros((self.len_selected_data, self.k))
        self.alpha_matrix = np.zeros((self.k, self.k))
        for year in range(self.len_selected_data):
            self.alpha_all_years[year] = self._projection_coefficients_year(year)

        from sklearn.model_selection import train_test_split
        # Further divide training dataset into train and validation dataset with an 90:10 split
        # noinspection PyPep8Naming
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                          test_size=0.1, random_state=2019)

        self.logger.info('Create network (model): specify number of neurons in each layer:')
        self.nr_neurons = self.k
        np.random.seed(3456)
        self.model = Sequential()  # kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.02, seed=None),
        # bias_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.02, seed=None)

        self.model.add(Dense(self.nr_neurons, input_dim=self.k, activation='relu',))
        self.model.add(Dense(self.nr_neurons, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
        self.model.add(Dense(self.nr_neurons, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
        self.model.add(Dense(self.nr_neurons, activation='linear'))

        self.model.compile(
            #        loss='categorical_crossentropy'  # mean_squared_error  # lr=0.2,decay=0.001
            loss='mean_squared_error', optimizer=optimizers.Adam(), metrics=['mean_squared_error'])

        #    model.compile(optimizer = "Adam",loss="binary_crossentropy",
        #    metrics=["accuracy"])mean_squared_error

        # Check the sizes of all newly created datasets
        logging.debug("Shape of x_train:", X_train.shape)
        logging.debug("Shape of x_val:", X_val.shape)
        logging.debug("Shape of y_train:", y_train.shape)
        logging.debug("Shape of y_val:", y_val.shape)

        logging.info('Train (fit) the network...')
        filepath = f"ModelWeights-{self.nr_neurons}.hdf5"
        checkpoint = ModelCheckpoint(filepath, save_best_only=True,
                                     monitor="val_mean_squared_error")
        out = self.model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=500, batch_size=32, verbose=0,
                             callbacks=[checkpoint])  #
        # , epochs=500, batch_size=30, verbose=0, callbacks=[mcp])

        # Saves the entire model into a file named as  'dnn_model.h5'
        self.model.save(f'dnn_model-{self.nr_neurons}.h5')

        self.logger.info('initial loss=' + repr(out.history["loss"][1]) + ', final=' + repr(out.history["loss"][-1]))
        self.logger.info(repr(out))

        # plot progress of optimization
        if 1:
            plt.figure(1, figsize=(12, 6))
            plt.clf()
            plt.semilogy(out.history["loss"][:])
            plt.legend(['log(loss)'])
            plt.xlabel('epoch')
            plt.ylabel('$\log_{10}$(loss)')
            plt.savefig(f"Progress_of_Optimization_{self.nr_neurons}_neurons.pdf", bbox_inches='tight')

            plt.figure(2, figsize=(12, 6))
            plt.clf()
            plt.plot(self.model.history.history['mean_squared_error'])
            plt.plot(self.model.history.history['val_mean_squared_error'])
            plt.title("Model's Training & Validation loss across epochs")
            plt.ylabel('Loss')
            plt.xlabel('Epochs')
            plt.legend(['Train', 'Validation'], loc='upper right')
            plt.savefig(f"Loss_function_train_and_loss_{self.nr_neurons}_neurons.pdf", bbox_inches='tight')

            plt.figure(3, figsize=(12, 6))
            plt.clf()
            plt.plot(self.model.history.history['mean_squared_error'])
            plt.plot(self.model.history.history['val_mean_squared_error'])
            plt.title("Model's Training & Validation mean squared error across epochs")
            plt.ylabel('mean squared error')
            plt.xlabel('Epochs')
            plt.legend(['Train', 'Validation'], loc='upper right')
            plt.savefig(f"mean_squared_error_train_and_loss_{self.nr_neurons}_neurons.pdf", bbox_inches='tight')

    def prediction_nn(self, clusters_1d: dict, composites_1d: dict, data_year_1d: dict, year: int):
        """make forecast
        :param clusters_1d: dict of all k-clusters
        :param composites_1d: dict of composites with time and one-dimensional array
        :param data_year_1d: np.ndarray with all  data of precursors
        :param year: year which should be forecasted
        """
        # Merge only those precursors which were selected
        self.selected_composites_1D, self.selected_data_1d = self._merge_selected_precursors(composites_1d,
                                                                                             data_year_1d, year)
        # Calculate projection coefficients
        self.beta_all = np.zeros(self.k)
        self.beta_all = self.model.predict(self.alpha_all)

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
            rhs[i] = sum(self.selected_composites_1D[i, :] * self.selected_data_1d[year][:])
        #
        # # Singular values smaller (in modulus) than 0.01 * largest_singular_value (again, in modulus) are set to zero
        self.pinv_matrix = np.linalg.pinv(self.alpha_matrix, 0.01)
        return np.dot(self.pinv_matrix, rhs)
