import numpy as np
from collections import OrderedDict
from laticore.metricsets import SupervisedTimeSeriesMetricSet

def create_cpu_units_metricset(utilization_ms:SupervisedTimeSeriesMetricSet,
    instnace_count_ms:SupervisedTimeSeriesMetricSet) -> SupervisedTimeSeriesMetricSet:
    # convert instance metricset data into ordered dict
    data = {}
    for i,j in enumerate(instance_count_ms.X.ravel()):
        data[j] = instance_count_ms.Y[i][0]

    data = OrderedDict(sorted(data.items()))

    # iterate through utilization_ms and multiply instance counts to get
    # cpu unit consumed
    new_y = np.zeros_like(utilization_ms.X)
    default_y = np.average(instance_count_ms.Y)
    for l, m in enumerate(utilization_ms.X.ravel()):
        new_y[l][0] = utilization_ms.Y[l][0] * data.get(m, default_y)

    # create a new metricset cpu * instance_count
    return SupervisedTimeSeriesMetricSet(
        X=utilization_ms.X,
        Y=new_y,
        Y_norm_buffer=float(utilization_ms.Y_norm_buffer),
        Y_norm_coefficient = utilization_ms.Y_norm_coefficient,
        lookback = utilization_ms.lookback,
    )
