"""Utilities"""

import numpy as np


def mrms_normalize(a):
    """
    Normalize, from MetNet-2 code release

    [NaN, inf, -inf, -50, -.5, 0., .2, 1., 2., 10.])
    ->
    [-1, -1, -1, -1, -1, 0, .046, .172, .268, .537]
    """
    a = np.nan_to_num(np.where(a < 0, 0, a + 1), posinf=0, neginf=0)
    return np.tanh(np.log(a) / 4)
