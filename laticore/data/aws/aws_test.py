import boto3
import unittest
import warnings
import numpy as np
from datetime import datetime, timedelta

from laticore.data.aws.aws import CloudWatch, Autoscaling

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

class TestAutoScaling(unittest.TestCase):

    asg_names = [
        "hw1_mgmt20180709213519083600000004",
        "hw2_mgmt20180710165059760100000004",
        "hw3_classic20180726160043061300000004",
    ]

    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore")

    def setUp(self):
        self.boto_session = boto3.Session()

    def test_fetch_config(self):
        for asg in self.asg_names:
            with self.subTest(asg=asg):
                autoscaling = Autoscaling(asg_name=asg, boto_session=self.boto_session)
                conf = autoscaling.fetch_config()
                self.assertTrue(isinstance(conf, dict))

    def test_locked(self):
        for asg in self.asg_names:
            with self.subTest(asg=asg):
                autoscaling = Autoscaling(asg_name=asg, boto_session=self.boto_session)
                self.assertTrue(isinstance(autoscaling.locked(), bool))
