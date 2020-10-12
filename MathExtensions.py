import numpy as np
from numba import jit, njit
from nptyping import NDArray
from typing import Dict


def get_cos(picks: float, points_count: int) -> NDArray[np.float64]:
    """Returns n-pick cos structure. 

    Args:
        picks: should be integer or half-integer positive number
    """
    return np.cos(np.linspace(0, picks*2*np.pi, points_count, dtype=np.float64))


@jit(nopython=True)
def tdma(ac: NDArray[np.float64], bc: NDArray[np.float64], cc: NDArray[np.float64], dc: NDArray[np.float64]) -> NDArray[np.float64]:
    """Solution of a linear system of algebraic equations with a
        tri-diagonal matrix of coefficients using the Thomas-algorithm.

    Args:
        a(array): an array containing upper diagonal (a[0] is not used)
        b(array): an array containing main diagonal 
        c(array): an array containing lower diagonal (c[-1] is not used)
        d(array): right hand side of the system
    Returns:
        x(array): solution array of the system

    """

    nf = len(dc)  # number of equations
    for it in range(1, nf):
        mc = ac[it-1]/bc[it-1]
        bc[it] = bc[it] - mc*cc[it-1]
        dc[it] = dc[it] - mc*dc[it-1]
    xc = bc
    xc[-1] = dc[-1]/bc[-1]

    for il in range(nf-2, -1, -1):
        xc[il] = (dc[il]-cc[il]*xc[il+1])/bc[il]

    return xc


def calc_picks(points: NDArray[np.float64]) -> Dict[str, NDArray]:
    """Returns picks count in cos-like stricture
    Format {picks: float, direction: up/down}
    """
    derivative_sign = 1 if points[2]-points[1] >= 0 else -1
    half_picks = 0
    for i in range(2, len(points)-3):
        if (points[i+1]-points[i])/derivative_sign < 0:
            derivative_sign *= -1
            half_picks += 1
    return {'picks': half_picks/2 + (0.5 if half_picks > 0 else 0), 'direction': 'down' if derivative_sign < 0 else 'up'}
