import bson
import boto3
import numpy as np
from collections import OrderedDict
from datetime import datetime, timedelta

from data.data import Data

class CloudWatch(Data):
    def __init__(self, boto_session):
        super().__init__()
        self.cloudwatch = self.session.client("cloudwatch")

    def _get_date_ranges(self, start_time:datetime, end_time:datetime):
        """
        Since cloudwatch API allows maximum 1440 minutes to be requested at a time
        we have to break up our requests into multiple time periods if we're requesting
        more than 1440 minutes.
        Thus, date_ranges is a list of (start,finish) tuples to be utlized in our
        requests to cloudwatch
        """
        date_ranges = []
        metric_request_delta = end_time - start_time
        seconds_in_day = 24 * 60 * 60
        total_seconds = metric_request_delta.total_seconds()

        while total_seconds > 0:
            start = end_time - timedelta(seconds = total_seconds)
            finish = start + min([timedelta(seconds = total_seconds), timedelta(seconds = seconds_in_day)])
            date_ranges.append((start, finish))
            total_seconds -= seconds_in_day

        return date_ranges

    def fetch(self, start_time:datetime, end_time:datetime, namespace:str, metric_name:str,
        dimensions:list, statistic:str,):
        """
        fetches the target_metric from cloudwatch in batches of <= 1440 minutes.
        Sets target_metric instance attr.
        """
        # create empty X and Y arrays that we can append to as data comes in from
        # AWS
        X = np.zeros((0,1))
        Y = np.zeros((0,1))

        for i, date in enumerate(self._get_date_ranges(start_time, end_time)):
            response = self.cloudwatch.get_metric_statistics(
                Namespace =  namespace,
                MetricName = metric_name,
                Dimensions = dimensions,
                StartTime = date[0],
                EndTime = date[1],
                Period = 60,
                Statistics = [statistic],
            )

            # Create dictionary of {timestamp : req_count} so we can sort by time
            data = {}
            for item in response["Datapoints"]:
                data[int(item["Timestamp"].timestamp())] = int(item[statistic])

            # Sort dict by timestamp and put into ordered dict
            data = OrderedDict(sorted(data.items()))

            # pre-allocate arrays for the results we just fetched
            x = np.zeros((len(data), 1),)
            y = np.zeros_like(x)

            # Fill x,y arrays with values.
            for j, item in enumerate(data.items()):
                x[j][0] = float(item[0])
                y[j][0] = item[1]

            # append the most-recently retrieved data to X and Y
            X = np.append(X, x, axis=0)
            Y = np.append(Y, y, axis=0)

        return X, Y
