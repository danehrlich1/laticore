import boto3
import unittest
import warnings
from laticore.auth.aws.aws import *

class TestAssumedRole(unittest.TestCase):

    boto_session = boto3.Session()

    configs = [
        {
            "external_id": "5b50de4358530577867929b6",
            "role_arn": "arn:aws:iam::670324507090:role/latitudeLocal",
            "role_session_name": "foobar",
        },
        {
            "external_id": "5b50de4358530577867929b6",
            "role_arn": "arn:aws:iam::670324507090:role/latitudeLocal",
            "duration_seconds": 2000,
            "boto_session": boto_session,
        },
    ]

    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore")

    def setUp(self):
        pass

    def test_get_session(self):
        for conf in self.configs:
            with self.subTest(conf=conf):
                sess = assume_role(**conf)
                self.assertTrue(sess != None)

if __name__ == '__main__':
    unittest.main()
