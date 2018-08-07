import numpy as np

class Data(object):
    def fetch(self, *args, **kwargs):
        raise NotImplementedError("fetch() must be implemented by child class.")
