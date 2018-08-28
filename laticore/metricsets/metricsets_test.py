import os
import unittest
import numpy as np
from datetime import datetime

from laticore.metricsets.metricsets import *

class TestSupervisedTimeSeriesMetricSet(unittest.TestCase):
    def setUp(self):
        pass

    def metricset_factory(self, timesteps=4320):
        now = int(datetime.utcnow().timestamp())
        later = now + (timesteps * 60)

        X = np.arange(now, later, 60).reshape((timesteps, 1))
        Y = np.random.randint(3000, 7000, (timesteps, 1)).astype(float)
        return SupervisedTimeSeriesMetricSet(X=X, Y=Y)

    def test_normalize_Y(self):
        ms = self.metricset_factory()
        ms.normalize_Y()
        self.assertTrue(ms.Y.max() <= 1)
        self.assertTrue(ms.Y.min() >= 0)
        self.assertTrue(ms.max_normalized_target_metric_value <= 1)
        self.assertTrue(ms.min_normalized_target_metric_value >= 0)

    def test_decompose_X(self):
        ms = self.metricset_factory()
        xlen = ms.X.shape[0]
        ms.decompose_X()
        self.assertTrue(np.equal((xlen, 3), ms.X.shape).all())

    def test_full_transform(self):
        lookback = 20
        timesteps = 4320
        ms = self.metricset_factory(timesteps)
        ms.full_transform()

        # shape should be (samples, timesteps, features)
        self.assertTrue(np.equal(
            (timesteps - 20, lookback, 4), ms.X.shape).all()
            )

        # shape should be (samples, 1)
        self.assertTrue(np.equal(
            (timesteps - lookback, 1), ms.Y.shape).all()
        )


if __name__ == '__main__':
    unittest.main()
