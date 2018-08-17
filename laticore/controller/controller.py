from laticore.task.task import Task
from abc import ABCMeta, abstractmethod

class Controller(object):
    """
    Abstract class that defines Controller interface
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def run(self, task:Task, *args, **kwargs):
        raise NotImplementedError("Child class of Controller must implement method 'run'")
