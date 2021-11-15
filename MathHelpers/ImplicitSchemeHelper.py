import numpy as np
from numba import njit
from typing import Dict
from nptyping import NDArray


@njit(fastmath=True, parallel=True)
def get_diags_implicit(t: float, h: float, D_coeff: float, length: int) -> Dict[str, NDArray[np.float64]]:
    """Return upper, main and lower diag of 3-diagonal matrix
    for solution of single equation from 1D Higgins model by Thomas algorithm
    Args:
        t: time step
        h: space step
        D_coeff: coefficient of diffusion
        length: number of points in a row
    Returns:
        {'upper': NDArray, 'middle': NDArray, 'lower': NDArray}
    """
    h_sq = h ** 2
    upper = np.ones(length) * D_coeff * (-t / h_sq)
    middle = np.ones(length) * (1 + 2 * D_coeff * t / h_sq)
    lower = upper.copy()
    middle[0] += upper[0]
    middle[-1] += upper[0]
    return {'upper': upper, 'middle': middle, 'lower': lower}