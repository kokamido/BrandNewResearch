from typing import Dict

import numpy as np
from nptyping import NDArray
from numba import jit

from MathHelpers.AlgebraicLinearSystemsSolvers import tdma


@jit
def get_diags(t: float, h: float, D_coeff: float, length: int) -> Dict[str, NDArray[np.float64]]:
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


@jit(nopython=True, parallel=True)
def get_right_vec_u(u: np.array, v: np.array, t: float) -> np.array:
    return u + (1 - u * v) * t


@jit(nopython=True, parallel=True)
def get_right_vec_v(u: np.array, v: np.array, p: float, q: float, t: float) -> np.array:
    return v + t * p * v * (u - (1 + q) / (q + v))


@jit(nopython=True, parallel=True)
def integrate(dt: float, dx: float, p: float, q: float, D_u: float, D_v: float, steps: int,
              init_u: np.array, init_v: np.array, save_timeline: bool = False, timeline_save_step: int = 10_000):

    assert init_u.shape == init_v.shape, f'shape mismatch: init_u.shape is {init_u.shape}, init_v.shape = {init_v.shape}'

    diags_u = get_diags(dt, dx, D_u, init_u.shape[0])
    U_u_d = diags_u['upper']
    U_m_d = diags_u['middle']
    U_l_d = diags_u['lower']

    diags_v = get_diags(dt, dx, D_v, init_v.shape[0])
    V_u_d = diags_v['upper']
    V_m_d = diags_v['middle']
    V_l_d = diags_v['lower']
    u = init_u.copy()
    v = init_v.copy()
    u_timeline = [init_u]
    v_timeline = [init_v]
    for i in range(steps):
        u_new = tdma(U_u_d.copy(), U_m_d.copy(),
                     U_l_d.copy(), get_right_vec_u(u, v, dt))
        v_new = tdma(V_u_d.copy(), V_m_d.copy(), V_l_d.copy(),
                     get_right_vec_v(u, v, p, q, dt))

        if i % 5000 == 0:
            if not np.isfinite(u_new).all():
                print('nans', i, i * dt)
                return None, None, None, None
            if np.linalg.norm(u - u_new) < 0.000000001:
                print(i, i * dt)
                break

        if save_timeline and i % timeline_save_step == 0:
            u_timeline.append(u_new)
            v_timeline.append(v_new)

        u = u_new
        v = v_new
    return u, v, np.stack(u_timeline, axis=0), np.stack(v_timeline, axis=0)
