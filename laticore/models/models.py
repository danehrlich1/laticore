import os
import redis
import pymongo
import numpy as np
from datetime import datetime, timedelta
from abc import ABCMeta, abstractmethod

from keras.models import Model as KModel
from keras.models import Sequential
from keras.models import Model as KerasModel
from keras.layers import Dense
from keras import layers

from laticore.metricsets.metricsets import SupervisedTimeSeriesMetricSet

class Model(object):
    """
    Abstract Model Class
    """
    __metaclass__ = ABCMeta


class TimeSeriesNNModel(Model):
    """
    Class for creating, training, and predicting sequence data as a supervised
    learning problem.
    """
    @abstractmethod
    def create_and_set_new_model(self, *args, **kwargs):
        """
        Abstract method that must be implemented by child class to create a new
        Keras.models.Model instance
        """
        raise NotImplementedError("Child class must implement method create")

    def train(self, metricset:SupervisedTimeSeriesMetricSet, epochs:int, batch_size:int=1,
        shuffle:bool=True, verbose:int=0):
        """
        Args:
            metricset (SupervisedTimeSeriesMetricSet, required): Instance of raw SupervisedTimeSeriesMetricSet
                that has not yet been transformed

            epochs (int, required): training epochs

            batch_size (int, optional, 1): training batch size

            shuffle (bool, optional, True): shuffle inputs

            verbose (int, optional, 0): verbosity level of tensforflow during training
        """
        if not isinstance(metricset, SupervisedTimeSeriesMetricSet):
            return TypeError("metricset must be of type SupervisedTimeSeriesMetricSet")

        metricset.full_transform()
        self._model.fit(
            metricset.X,
            metricset.Y,
            epochs      = epochs,
            batch_size  = batch_size,
            verbose     = 0,
            shuffle     = shuffle,
        )

    def predict(self, metricset:SupervisedTimeSeriesMetricSet, lookahead:int, timestep_size_seconds:int = 60):
        """
        Performs sequence prediction of n timesteps after the last metric time
        in the inputted metricset

        Args:
            metricset (SupervisedTimeSeriesMetricSet, required): Instance of raw SupervisedTimeSeriesMetricSet
                that has not yet been transformed

            lookback (int, required): number of timesteps to look back per observation

            lookahead (int, required): how many time steps to predict forward

        Returns:
            predictions ( np.ndarray(lookahead, 1) ): Predictions n=lookahead into the future
        """
        if not isinstance(metricset, SupervisedTimeSeriesMetricSet):
            return TypeError("metricset must be of type SupervisedTimeSeriesMetricSet")
        # perform partial transformation on input metricset
        # We don't go so far as to created a supervised (i.e. windowed dataset)
        # because we're going to combine this with our "future" metricset and THEN
        # create the supervised set
        metricset.normalize_Y()
        metricset.decompose_X()
        metricset.stack_transform()

        # create start and end timestamps for t + step_size...t+(lookahead * step_size)
        start_ts = metricset.X_orig[-1][0] + timestep_size_seconds
        end_ts = start_ts + (lookahead * timestep_size_seconds)

        # create a future metricset that has x_values as unix timestamps timestep_size_seconds
        # apart and y_values = 0 (because we haven't predicted the future yet)
        future_X = np.arange(start_ts, end_ts, timestep_size_seconds).reshape((lookahead, 1))
        future_Y = np.zeros_like(future_X)

        # create future metricset and process into (n_samples, features)
        future_ms = SupervisedTimeSeriesMetricSet(
            X = future_X,
            Y = future_Y,
            lookback = metricset.lookback,
            Y_norm_buffer = metricset.Y_norm_buffer,
            Y_norm_coefficient = metricset.Y_norm_coefficient,
        )
        future_ms.decompose_X()
        future_ms.stack_transform()

        # Now, create a composite metricset of stacked train_ms and future_ms
        # values, and transform into a windowed, supervised dataset where X is
        # of the shape (n_samples, timesteps, features)
        prediction_X = np.vstack((metricset.X, future_ms.X))
        prediction_Y = np.vstack((metricset.Y, future_ms.Y))
        prediction_ms = SupervisedTimeSeriesMetricSet(
            X = prediction_X,
            Y = prediction_Y,
            lookback = metricset.lookback,
            Y_norm_buffer = metricset.Y_norm_buffer,
            Y_norm_coefficient = metricset.Y_norm_coefficient,
        )
        prediction_ms.supervised_transform()

        # pre-allocate array to hold predictions
        predictions = np.zeros((lookahead, 1), np.float)

        # n_samples is length of X input
        xlen = prediction_ms.X.shape[0]
        # p is index of the predictions array we're inserting into
        p = 0
        # i is the iterator, which is used for slicing X
        i = lookahead

        # iterate through our prediction metricset, create a prediction, and then
        # backfill "zeroed" values with our prediction, which will be used in
        # subsequent iterations
        while i > 0:
            x = np.array([prediction_ms.X[xlen-i]])
            # predict value based on prediction_ms slice
            prediction = self._model.predict(x, batch_size=1)
            # add prediction value to prediction array. value must >= 0
            predictions[p] = max(0.0, prediction[0])
            # backfill prediction into future windows
            j = i - 1
            k = metricset.lookback - 1
            while j > 0:
                prediction_ms.X[xlen-j][k][prediction_ms.X.shape[2]-1] = prediction[0][0]
                j -= 1
                k -= 1

            i -=1
            p += 1

        return predictions / metricset.Y_norm_coefficient

class TimeSeriesLSTMModel(TimeSeriesNNModel):

    def create_and_set_new_model(self, input_nodes:int, input_shape:tuple, activation:str,
        dense_nodes:int, loss_function:str, optimizer:str):
        """
        Creates a new instance of a Keras Model and sets as instance attribute _model

        Args:
            input_nodes (required, int): Number of input nodes for LSTM input layer

            input_shape (required, tuple): Input shape as tuple of (timesteps, features)

            activation (required, str): Activation function to be used

            dense_nodes (required, int): Number of dense nodes in hidden layer

            loss_function (required str): Loss function to be used for training

            optimizer (required, str): Optimzier to be used for compiling the model
        """
        # create the LSTM network
        nn = Sequential()

        # Add LSTM input layer
        nn.add(layers.LSTM(
            input_nodes,
            input_shape = input_shape,
            activation = activation,
        ))

        # Add Dense Hidden Layer
        nn.add(Dense(
            units = dense_nodes,
        ))

        # Compile the model
        nn.compile(
            loss = loss_function,
            optimizer = optimizer,
        )

        self._model = nn
