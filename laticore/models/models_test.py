import unittest
import numpy as np
import warnings
from datetime import datetime

from keras.models import Sequential
from keras.models import Model as KerasModel
from keras.layers import Dense
from keras import layers

from laticore.models.models import TimeSeriesNNModel
from laticore.metricsets.metricsets import SupervisedTimeSeriesMetricSet

class TestTimeSeriesNNModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore")

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        self.model = TimeSeriesNNModel()
        nn = Sequential()
        nn.add(layers.LSTM(
            50,
            input_shape = (20, 4),
            activation = "linear",
        ))
        nn.add(Dense(units = 1))
        nn.compile(
            loss = "binary_crossentropy",
            optimizer = "adam",
        )
        self.model._model = nn

    def tearDown(self):
        pass

    def metricset_factory(self, timesteps=1000):
        now = int(datetime.utcnow().timestamp())
        later = now + (timesteps * 60)
        X = np.arange(now, later, 60).reshape((timesteps, 1))
        Y = np.random.randint(3000, 7000, (timesteps, 1)).astype(float)
        return SupervisedTimeSeriesMetricSet(X=X, Y=Y, lookback = 20)

    def test_train(self):
        ms = self.metricset_factory(timesteps=100)
        self.model.train(
            metricset   = ms,
            epochs      = 1,
            batch_size  = 1,
            shuffle     = True,
            verbose     = 0,
        )

    def test_predict(self):
        ms = self.metricset_factory(timesteps=100)
        self.model.train(
            metricset   = ms,
            epochs      = 1,
            batch_size  = 1,
            shuffle     = True,
            verbose     = 0,
        )

        ms_predict = SupervisedTimeSeriesMetricSet(
                        X = ms.X_orig,
                        Y = ms.Y_orig,
                        lookback = ms.lookback,
                        Y_norm_buffer = ms.Y_norm_buffer,
                        Y_norm_coefficient = ms.Y_norm_coefficient,
                        )

        predictions = self.model.predict(metricset=ms_predict, lookahead=10)

        self.assertTrue(isinstance(predictions, np.ndarray))
