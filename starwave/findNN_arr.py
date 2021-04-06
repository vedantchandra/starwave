import numpy as np
from numba import jit

@jit
def find_nearest(array,value):
    """Find the nearest element in an array"""
    idx = (np.abs(array-value)).argmin()
    return array[idx]

@jit
def find_nearest_idx(array,value):
    """Find the index of the nearest element in an array"""
    idx = (np.abs(array-value)).argmin()
    return idx
