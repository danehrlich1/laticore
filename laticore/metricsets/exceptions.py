class DimensionError(Exception):
    """
    Error raised when the dimensions of a numpy array do not match the
    required dimensions.
    """
    pass

class TransformError(Exception):
    """
    Error raised when attempting to perform an irreversible metricset
    transformation that has already taken place.
    """
    pass
