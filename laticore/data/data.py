from abc import ABCMeta, abstractmethod

class Data(object):
    """
    Abstract class representing interface of Data fetching classes
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def fetch(self, *args, **kwargs):
        raise NotImplementedError("fetch() must be implemented by child class.")
