# laticore

Laticore contains useful packages for developing predictive autoscaling services.

## Installation

`pip install --upgrade git+ssh://git@github.com/konture/laticore.git@<version>#egg=laticore`

## Packages

| Package | Description |
| --- | --- |
| Auth | Authentication mechanisms to various cloud providers and services |
| Controller | Abstract controller interface to be used by the running autoscaling service |
| Data | Classes for retrieving data from various sources |
| Metricsets | Classes format data retrieved from Data classes into supervised time series datasets |
| Models | Machine learning models, and helper classes for managing state and model storage |
| Task | Task interface is the job configuration for performing an training or autoscaling task |


## Maintainers

| Name | Contact |
| --- | --- |
| Scott Crespo | scott@konture.io |
| Jason van Brackel | jvb@rancher.io |
