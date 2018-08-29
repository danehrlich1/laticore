import bson
import boto3
import numpy as np
from collections import OrderedDict
from datetime import datetime, timedelta

from laticore.data.data import Data
from laticore.data.aws.exceptions import AutoscalingLockedException

class CloudWatch(Data):
    """
    Implements methods for retrieving large quantities of data from AWS Cloudwatch
    with minute resolution.
    """
    def __init__(self, boto_session):
        self.boto_session = boto_session
        self.cloudwatch = self.boto_session.client("cloudwatch")

    def _get_date_ranges(self, start_time:datetime, end_time:datetime):
        """
        Since cloudwatch API allows maximum 1440 minutes to be requested at a time
        we have to break up our requests into multiple time periods if we're requesting
        more than 1440 minutes.
        Thus, date_ranges is a list of (start,finish) tuples to be utlized in our
        requests to cloudwatch

        Args:
            start_time(datetime, required): Beginning of the metric window to request
            end_time(datetime, required): Ending of the metric window to request

        Returns:
            date_ranges(list(tuple(datetime,datetime))): list of tuples containing (start_time, end_time)
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
        dimensions:list, statistic:str, trim_last:int=0):
        """
        fetches the target_metric from cloudwatch in batches of <= 1440 minutes.
        Sets target_metric instance attr.
        Data is requested with minute resolution.

        Args:
            start_time (datetime, required): beginning of metric request window
            end_time (datetime, required): ending of metric request window
            namespace (str, required): CloudWatch namespace to request from
            metric_name (str, required): Cloudwatch metric to request
            dimensions (list(dict), required): dimensions to get the right targets
            statistic (str, required): the statistic to request (i.e. Sum, Average, etc.)
            trim_last (int, 0): whether to omit the last n data points from the result

        Returns:
            X (2d numpy array): 2d array of unix timestamps
            Y (2d numpy array): 2d array of the target metric
        """
        # create empty X and Y arrays that we can append to as data comes in from
        # AWS
        X = np.zeros((0,1))
        Y = np.zeros((0,1))
        date_ranges = self._get_date_ranges(start_time, end_time)

        for i, date in enumerate(date_ranges):
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

            # The last value returned from cloudwatch (i.e. the most recent value)
            # can't be trusted. It's typtically incomplete due to eventual consistency issues
            # and biases predictions downwards. So, we take a naive approach
            # here and trim the very last value of the last date range returned.
            if trim_last > 0 and i == len(date_ranges) - 1:
                    counter = 0
                    for idx in reversed(data):
                        del data[idx]
                        counter += 1
                        if counter >= trim_last:
                            break

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


class Autoscaling(Data):
    """
    Fetches data for AWS Autoscaling Groups
    """

    def __init__(self, asg_name:str, boto_session:boto3.Session):
        """
            Args:
                asg_name(str, required): Name of the autosclaing group
                boto_session(boto3.Session, required): Customer boto session with assumed role
        """
        self.asg_name = asg_name
        self.boto_session = boto_session
        self.autoscaling = boto_session.client("autoscaling")

    def fetch_config(self) -> dict:
        """
        fetches autoscaling group config from aws

        Returns:
            config(dict):
                min_size
                max_size
                default_cooldown
                desired_capacity
                instance_count
        """
        response = self.autoscaling.describe_autoscaling_groups(
            AutoScalingGroupNames = [
                self.asg_name,
            ],
        )

        asg_config_raw = response["AutoScalingGroup"][0]

        self.config = {
                    "min_size": asg_config_raw["MinSize"],
                    "max_size": asg_config_raw["MaxSize"],
                    "default_cooldown": asg_config_raw["DefaultCooldown"],
                    "desired_capacity": asg_config_raw["DesiredCapacity"],
                    "instance_count": len(asg_config_raw['Instances']),
                }

        return self.config

    def locked(self) -> bool:
        """
        checks scaling activies of the autoscaling group and reports if the asg
        is busy or not.

        Returns:
            locked(bool): True if asg is locked
        """
        response = self.autoscaling.describe_scaling_activities(
            AutoScalingGroupName = self.asg_name,
            MaxRecords = 10,
        )

        # these are the busy codes that might be reported by autoscaling
        busy_codes = [
            "PendingSpotBidPlacement",
            "WaitingForSpotInstanceRequestId",
            "WaitingForSpotInstanceId",
            "WaitingForInstanceId",
            "PreInService",
            "InProgress",
            "WaitingForELBConnectionDraining",
            "MidLifecycleAction",
            "WaitingForInstanceWarmup",
        ]

        # check if there's a busy signal listed in activities
        for activity in response["Activities"]:
            if activity["StatusCode"] in busy_codes:
                return True

        return False
