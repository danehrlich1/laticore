import math
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

class DimensionError(Exception):
    """
    Error raised when the dimensions of a numpy array do not match the
    required dimensions.
    """
    pass

class TransformError(Exception):
    """
    Error raised when attempting to perform an irreversible metricset
    transformation that has already taken place.
    """
    pass

class MetricSet(object):

    def __init__(self, X:np.ndarray, Y:np.ndarray):
        self.X = np.copy(X)
        self.Y = np.copy(Y)
        self.X_orig = np.copy(X)
        self.Y_orig = np.copy(Y)

    @property
    def X(self):
        return self._X

    @X.setter
    def X(self, val):
        if isinstance(val, np.ndarray) == False:
            raise TypeError("X must be a numpy array")
        self._X = val

    @property
    def Y(self):
        return self._Y

    @Y.setter
    def Y(self, val):
        if isinstance(val, np.ndarray) == False:
            raise TypeError("Y must be a numpy array")
        self._Y = val

    @property
    def X_orig(self):
        return self._X_orig

    @X_orig.setter
    def X_orig(self, val):
        if isinstance(val, np.ndarray) == False:
            raise TypeError("X must be a numpy array")
        if len(val.shape) != 2:
            raise DimensionError("X array must be 2d numpy array.")
        self._X_orig = val

    @property
    def Y_orig(self):
        return self._Y_orig

    @Y_orig.setter
    def Y_orig(self, val):
        if isinstance(val, np.ndarray) == False:
            raise TypeError("Y must be a numpy array")
        if len(val.shape) != 2:
            raise DimensionError("Y array must be 2d numpy array.")
        self._Y_orig = val

    @property
    def max_X(self):
        return self.X_orig.max()

    @property
    def max_Y(self):
        return self.Y_orig.max()

    @property
    def min_X(self):
        return self.X_orig.min()

    @property
    def min_Y(self):
        return self.Y_orig.min()

class TimeSeriesMetricSet(MetricSet):
    """
    MetricSet that takes X and Y 2d arrays, where X is an array
    of Unix timestamps
    """
    def __init__(self, X:np.ndarray, Y:np.ndarray):
        self.X = np.copy(X)
        self.Y = np.copy(Y)
        self.X_orig = np.copy(X)
        self.Y_orig = np.copy(Y)

    @property
    def first_metric_time(self) -> datetime:
        """
        Returns datetime object of first metric in X
        """
        return datetime.utcfromtimestamp(self.X_orig[0][0])

    @property
    def last_metric_time(self) -> datetime:
        """
        Returns datetime object of last metric in X
        """
        return datetime.utcfromtimestamp(self.X_orig[-1][0])

class SupervisedTimeSeriesMetricSet(TimeSeriesMetricSet):
    def __init__(self, X:np.ndarray, Y:np.ndarray, lookback:int = 20,
        Y_norm_buffer:float = 1.0, Y_norm_coefficient:float = None):
        self.X = np.copy(X)
        self.Y = np.copy(Y)
        self.X_orig = np.copy(X)
        self.Y_orig = np.copy(Y)
        self._set_scalers()

        self.lookback = lookback
        self.Y_norm_buffer = Y_norm_buffer
        self.Y_norm_coefficient = Y_norm_coefficient

        self._transformed_to_supervised = False

    @property
    def lookback(self):
        return self._lookback

    @lookback.setter
    def lookback(self, val):
        if not isinstance(val, int):
            raise TypeError("lookback must be of type int")
        if not val >= 1:
            raise ValueError("lookback must be greater than or equal to one")

        self._lookback = val

    @property
    def Y_norm_buffer(self):
        return self._Y_norm_buffer

    @Y_norm_buffer.setter
    def Y_norm_buffer(self, val):
        if not isinstance(val, float):
            return TypeError("Y_norm_buffer must be of type float")
        if not val >= 1.0:
            return ValueError("Y_norm_buffer must be greater than or equal to 1.0")
        self._Y_norm_buffer = val

    @property
    def Y_norm_coefficient(self):
        return self._Y_norm_coefficient

    @Y_norm_coefficient.setter
    def Y_norm_coefficient(self, val):
        if val is None:
            scaler = MinMaxScaler(feature_range=(0.0, 1.0))
            tofit = np.array([[0], [self.Y.max() * self.Y_norm_buffer]])
            scaler.fit(tofit)
            self._Y_norm_coefficient = scaler.scale_[0]
        else:
            if not isinstance(val, float):
                return TypeError("Y_norm_coefficient must be of type float")
            if not (0.0 <= val <= 1.0):
                return ValueError("Y_norm_coefficient must be between 0.0 and 1.0 inclusively")

            self._Y_norm_coefficient = val

    @property
    def day_scaler(self):
        return self._day_scaler

    @property
    def hour_scaler(self):
        return self._hour_scaler

    @property
    def min_scaler(self):
        return self._min_scaler

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
        day_set = self.day_scaler.transform(day_set.reshape(xlen, 1))

        # reshape and scale hour set
        hour_set = self.hour_scaler.transform(hour_set.reshape(xlen, 1))

        # reshape and scale min set
        min_set = self.min_scaler.transform(min_set.reshape(xlen, 1))

        self.X = np.hstack((day_set, hour_set, min_set))

    def normalize_Y(self):
        """
        Performs in place normalization of Y vector. If norm_coefficient is not
        set, it is determined using (0, Ymax * buffer)
        """
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
        """
        Convenience method that calls all the methods required to perform a
        complete transformation on a metricset to be used as supervised learning
        inputs to a neural network.
        """
        if self._transformed_to_supervised:
            raise TransformError("Metricset has already been transformed to a supervsied set.")

        self.normalize_Y()
        self.decompose_X()
        self.stack_transform()
        self.supervised_transform()
