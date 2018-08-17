class NullModelException(Exception):
    """
    Excpetion raised when trying to work with a Keras Model instance that is None.
    """
    pass

class ModelNotFoundException(Exception):
    """
    Exception raised when a model can't be found in storage
    """
    pass
