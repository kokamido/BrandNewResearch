from typing import Union

import numpy as np
from nptyping import NDArray


def get_cos(picks: float, points_count: int, shift: float, amplitude: float) -> NDArray[np.float64]:
    """Returns n-pick cos structure.
    Args:
        amplitude: cos scaling factor
        shift: linear shift
        points_count: number of grid points
        picks: should be integer or half-integer positive number
    Returns
        NDArray[np.float64] array of points
    """
    assert (picks % 0.5) < 0.01, f"\'picks\' should be integer or half-integer positive number, given value '{picks}'"
    return np.cos(np.linspace(0, picks * 2 * np.pi, points_count, dtype=np.float64)) * amplitude + shift


def get_normal_rand(points_count: int, mean: float, std_dev: float) -> Union[NDArray, int, float, complex]:
    """Returns samples from N(mean, std_dev) with no-flux boundary condition.
       Args:
           points_count: number of grid points
       Returns
           Union[NDArray, int, float, complex] points
       """
    res = np.random.normal(mean, std_dev, points_count)
    res[0] = res[1]
    res[-1] = res[-2]
    return res
