from typing import Union

import numpy as np



def get_cos(peaks: float, points_count: int, shift: float, amplitude: float) -> np.array:
    """Returns n-pick cos structure.
    Args:
        amplitude: cos scaling factor
        shift: linear shift
        points_count: number of grid points
        peaks: should be integer or half-integer positive number
    Returns
        np.array array of points
    """
    assert (peaks % 0.5) < 0.01, f"\'picks\' should be integer or half-integer positive number, given value '{peaks}'"
    return np.cos(np.linspace(0, peaks * 2 * np.pi, points_count, dtype=np.float64)) * amplitude + shift


def get_normal_rand(points_count: int, mean: float, std_dev: float) -> Union[np.array, int, float, complex]:
    """Returns samples from N(mean, std_dev) with no-flux boundary condition.
       Args:
           points_count: number of grid points
       Returns
           Union[np.array, int, float, complex] points
       """
    res = np.random.normal(mean, std_dev, points_count)
    res[0] = res[1]
    res[-1] = res[-2]
    return res
