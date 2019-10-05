import numpy

try:
    import cupy
except ImportError:
    cupy = None


def get_cupy():
    return cupy


def get_array_module(*args, **kwargs):
    if cupy is not None:
        return cupy.get_array_module(*args, **kwargs)
    else:
        return numpy


def asnumpy(x):
    if cupy is not None:
        return cupy.asnumpy(x)
    else:
        return numpy.asarray(x)