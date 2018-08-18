import boto3
import unittest
import warnings
import numpy as np
from datetime import datetime, timedelta

from laticore.data.aws.aws import *

class TestCloudWatch(unittest.TestCase):
    now = datetime.now()
    configs = [
        {
            "start_time": now - timedelta(days=2),
            "end_time": now,
            "namespace": "AWS/ApplicationELB",
            "metric_name": "RequestCount",
            "dimensions": [
                {
                    "Name": "LoadBalancer",
                    "Value": "app/hw1-alb-mgmt/cd39c2db408cf22b"
                },
            ],
            "statistic": "Sum",
        },
    ]

    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore")

    def setUp(self):
        self.boto_session = boto3.Session()

    def test_get_date_ranges(self):
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=3)
        cw = CloudWatch(self.boto_session)
        date_ranges = cw._get_date_ranges(start_time, end_time)
        self.assertEqual(3, len(date_ranges))

    def test_fetch(self):
        cw = CloudWatch(self.boto_session)
        for conf in self.configs:
            with self.subTest(conf=conf):
                X, Y = cw.fetch(**conf)
                self.assertTrue(isinstance(X, np.ndarray))
                self.assertTrue(isinstance(Y, np.ndarray))
                self.assertEqual(len(X.shape), 2)
                self.assertEqual(len(Y.shape), 2)
