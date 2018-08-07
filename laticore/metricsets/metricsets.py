import math
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

class DimensionError(Exception):
    pass

class TransformError(Exception):
    pass

class MetricSet(object):

    def __init__(self, X:np.ndarray, Y:np.ndarray):
        self.validate_dimensions(X, Y)
        self.X = np.copy(X)
        self.Y = np.copy(Y)
        self.X_orig = np.copy(X)
        self.Y_orig = np.copy(Y)

    def validate_dimensions(self, X, Y):
        if len(X.shape) != 2:
            raise DimensionError("X array must be 2d numpy array.")

        if len(Y.shape) != 2:
            raise DimensionError("Y array must be 2d numpy array.")

    def max_X(self):
        return self.X_orig.max()

    def max_Y(self):
        return self.Y_orig.max()

    def min_X(self):
        return self.X_orig.min()

    def min_Y(self):
        return self.Y_orig.min()

class TimeSeries(MetricSet):
    """
    MetricSet that takes X and Y 2d arrays, where X is an array
    of Unix timestamps
    """
    def __init__(self, X:np.ndarray, Y:np.ndarray):
        super().__init__(X, Y)

    def first_metric_time(self) -> datetime:
        """
        Returns datetime object of first metric in X
        """
        return datetime.utcfromtimestamp(self.X_orig[0][0])

    def last_metric_time(self) -> datetime:
        """
        Returns datetime object of last metric in X
        """
        return datetime.utcfromtimestamp(self.X_orig[-1][0])

class SupervisedDHM(TimeSeries):
    def __init__(self, X:np.ndarray, Y:np.ndarray, lookback:int = 20,
        Y_norm_coefficient:float = None, Y_norm_buffer:float = 1.0,):
        super().__init__(X, Y)

        self._set_scalers()
        self.lookback = lookback
        self.Y_norm_coefficient = Y_norm_coefficient
        self.Y_norm_buffer = Y_norm_buffer
        self._transformed_to_supervised = False

    def _set_scalers(self):
        """
        Creates integer encoders and/or one-hot encoders for days of the week and
        hours of the day.
        """
        # day scaler
        self._day_scaler = MinMaxScaler(feature_range=(0,1))
        self._day_scaler.fit(np.array([[0.],[6.]]))

        # hour scaler
        self._hour_scaler = MinMaxScaler(feature_range=(0,1))
        self._hour_scaler.fit(np.array([[0.],[23.]]))

        # minute scaler
        self._min_scaler = MinMaxScaler(feature_range=(0,1))
        self._min_scaler.fit(np.array([[0.],[59.]]))

    def decompose_X(self):
        """
        Decomposes timestamps in X into features:
            Day of the Week
            Hour of the Day
            Minute of the Hour
        """
        xlen = self.X.shape[0]

        day_set = np.zeros((xlen))
        hour_set = np.zeros((xlen))
        min_set = np.zeros((xlen))

        for i, ts in enumerate(self.X.ravel()):
            dt = datetime.fromtimestamp(ts)
            day_set[i] = dt.weekday()
            hour_set[i] = dt.hour
            min_set[i] = dt.minute

        # reshape and scale day set
        day_set = self._day_scaler.transform(day_set.reshape(xlen, 1))

        # reshape and scale hour set
        hour_set = self._hour_scaler.transform(hour_set.reshape(xlen, 1))

        # reshape and scale min set
        min_set = self._min_scaler.transform(min_set.reshape(xlen, 1))

        self.X = np.hstack((day_set, hour_set, min_set))

    def normalize_Y(self):
        """
        Performs in place normalization of Y vector. If norm_coefficient is not
        set, it is determined using (0, Ymax * buffer)
        """
        if not self.Y_norm_coefficient:
            scaler = MinMaxScaler(feature_range=(0, 1))
            tofit = np.array([[0], [self.Y.max() * self.Y_norm_buffer]])
            scaler.fit(tofit)
            self.Y_norm_coefficient = scaler.scale_[0]

        np.multiply(self.Y, self.Y_norm_coefficient, out=self.Y)

    def stack_transform(self):
        """
        Performs horizontal stack of X and Y
        """
        self.X = np.hstack((self.X, self.Y))

    def supervised_transform(self):
        """
        Transforms X and Y matrices into supervised, windowed dataset
        """
        x = []
        y = []
        i = 0
        j = (self.Y.shape[0] - self.lookback)

        while i < j:
            a = self.X[i:(i + self.lookback)]
            x.append(a)
            y.append(self.Y[i + self.lookback])
            i += 1

        self.X = np.array(x)
        self.Y = np.array(y)
        self._transformed_to_supervised = True

    def full_transform(self):
        if self._transformed_to_supervised:
            raise TransformError("Metricset has already been transformed to a supervsied set.")

        self.normalize_Y()
        self.decompose_X()
        self.stack_transform()
        self.supervised_transform()
