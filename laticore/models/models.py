import os
import redis
import pymongo
import numpy as np
from datetime import datetime, timedelta
from keras.models import Model as KModel
from laticore.metricsets.metricsets import SupervisedTimeSeriesMetricSet

class ModelConfig(object):
    pass

class ModelNotFound(Exception):
    pass

class Model(object):
    pass


class TimeSeriesNNModel(Model):
    """
    Class for creating, training, and predicting sequence data as a supervised
    learning problem.
    """
    @abstractmethod
    def create(self) -> KModel:
        """
        Abstract method that must be implemented by child class to create a new
        Keras.models.Model instance
        """
        raise NotImplementedError("Child class must implement method create")

    def train(self, metricset:SupervisedTimeSeriesMetricSet, epochs:int, batch_size:int=1,
        shuffle:bool=True, verbose:int=0):
        """
        metricset (TimeSeries, required):
        epochs (int, required)
        batch_size (int, optional, 1)
        shuffle (bool, optional, True)
        verbose (int, optional, 0)
        """

        if not isinstance(val, SupervisedTimeSeriesMetricSet):
            return TypeError("metricset must be of type SupervisedTimeSeriesMetricSet")

        metricset.full_transform()

        self._model.fit(
            metricset.X,
            metricset.Y,
            epochs = epochs,
            batch_size = batch_size,
            verbose=0,
            shuffle=shuffle,
        )

    def predict(self):
        pass
