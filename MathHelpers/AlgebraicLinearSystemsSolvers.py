import numpy as np
from nptyping import NDArray
from numba import njit


@njit(fastmath=True)
def tdma(ac: NDArray[np.float64], bc: NDArray[np.float64], cc: NDArray[np.float64], dc: NDArray[np.float64]) \
        -> NDArray[np.float64]:
    """Solution of a linear system of algebraic equations with a
        tri-diagonal matrix of coefficients using the Thomas-algorithm.

    Args:
        ac(array): an array containing upper diagonal (a[0] is not used)
        bc(array): an array containing main diagonal
        cc(array): an array containing lower diagonal (c[-1] is not used)
        dc(array): right hand side of the system
    Returns:
        x(array): solution array of the system
    """
    nf = len(dc)  # number of equations
    for it in range(1, nf):
        mc = ac[it - 1] / bc[it - 1]
        bc[it] = bc[it] - mc * cc[it - 1]
        dc[it] = dc[it] - mc * dc[it - 1]
    xc = bc
    xc[-1] = dc[-1] / bc[-1]

    for il in range(nf - 2, -1, -1):
        xc[il] = (dc[il] - cc[il] * xc[il + 1]) / bc[il]
    return xc
