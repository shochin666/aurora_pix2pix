import numpy as np


def min_max(x, axis=None):
    min = x.min(axis=axis, keepdims=True)
    max = x.max(axis=axis, keepdims=True)
    result = (x - min) / (max - min) * 256
    return result


def reverse_min_max(min, max, z, axis=None):
    y = z * (max - min) / 256 + min
    x = 10 ** (y / 10)

    return x
