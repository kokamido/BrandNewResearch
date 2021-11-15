import numpy as np
from numba import njit

from MyPackage.DataContainers.Experiment import Experiment
from MyPackage.MathHelpers.AlgebraicLinearSystemsSolvers import tdma
from MyPackage.MathHelpers.ImplicitSchemeHelper import get_diags_implicit
from MyPackage.Models.Selkov1D.Selkov1DConfiguration import Selkov1DConfiguration
from MyPackage.Models.TdmaParameters1D import TdmaParameters1D


def integrate_tdma_implicit_scheme(config: Selkov1DConfiguration, method_config: TdmaParameters1D) -> Experiment:
    """
    Args:
        dump_path: if not None, data will be stored on disk

    :returns: Tuple
            - u - is final U state;
            - v - is final V state;
            - u_timeline - is 2d-array contains timeline of U if save_timeline else None;
            - v_timeline - is 2d-array contains timeline of V if save_timeline else None;
    """
    model_config = config
    method_config = method_config
    method_config['method'] = 'tdma_implicit'
    if 'seed' not in method_config:
        method_config['seed'] = np.random.randint(2147483647)
    u_init = method_config['u_init']
    del method_config['u_init']
    v_init = method_config['v_init']
    del method_config['v_init']
    steps = int(round(method_config['t_max'] / method_config['dt']))
    np.random.seed(method_config['seed'])
    u, v, u_timeline, v_timeline = __integrate_tdma_implicit__(method_config['dt'], method_config['dx'], steps,
                                                               model_config['theta'], model_config['omega'],
                                                               model_config['Du'], model_config['Dv'],
                                                               u_init, v_init,
                                                               method_config['save_timeline'],
                                                               method_config['timeline_save_step_delta'],
                                                               method_config['min_t'], method_config['noise_amp'])
    if u is None:
        raise ArithmeticError('Selkov1d evaluation failed')
    res = Experiment()
    res.fill(model_config,method_config, {'u': u_init, 'v': v_init}, {'u': u, 'v': v}, {
        'u': np.vstack(u_timeline), 'v': np.vstack(v_timeline)})
    return res


@njit(fastmath=True, parallel=True)
def __get_right_vec_u_implicit__(u: np.array, v: np.array, n: float, t: float) -> np.array:
    return u + (n - u * v * v) * t


@njit(fastmath=True, parallel=True)
def __get_right_vec_v_implicit__(u: np.array, v: np.array, w: float, t: float) -> np.array:
    return v + t * (u * v * v - w * v)


@njit(fastmath=True, parallel=True)
def __integrate_tdma_implicit__(dt: float, dx: float, steps: int, n: float, w: float, D_u: float, D_v: float,
                                init_u: np.array, init_v: np.array, save_timeline: bool = False,
                                timeline_save_step: int = 10_000, min_t: int = None, noise_amp: float = None):
    """
    :returns: Tuple
        - u - is final U state;
        - v - is final V state;
        - u_timeline - is 2d-array contains timeline of U if save_timeline else None;
        - v_timeline - is 2d-array contains timeline of V if save_timeline else None;
    """
    diags_u = get_diags_implicit(dt, dx, D_u, init_u.shape[0])
    U_u_d = diags_u['upper']
    U_m_d = diags_u['middle']
    U_l_d = diags_u['lower']

    diags_v = get_diags_implicit(dt, dx, D_v, init_v.shape[0])
    V_u_d = diags_v['upper']
    V_m_d = diags_v['middle']
    V_l_d = diags_v['lower']
    u = init_u.copy()
    v = init_v.copy()
    u_timeline = [init_u]
    v_timeline = [init_v]
    for i in range(steps):
        u_new = tdma(U_u_d.copy(), U_m_d.copy(),
                     U_l_d.copy(), __get_right_vec_u_implicit__(u, v, n, dt))
        v_new = tdma(V_u_d.copy(), V_m_d.copy(), V_l_d.copy(),
                     __get_right_vec_v_implicit__(u, v, w, dt))

        if noise_amp:
            u_new += np.random.standard_normal(u_new.shape) * noise_amp
            v_new += np.random.standard_normal(v_new.shape) * noise_amp

        if i % 5000 == 4999:
            if not np.isfinite(u_new).all():
                return None, None, None, None
            if (min_t is None or i * dt > min_t) and np.linalg.norm(u - u_new) < 1e-7:
                break
            if np.linalg.norm(u) < 0.000000001:
                break

        if save_timeline and i % timeline_save_step == 0:
            u_timeline.append(u_new)
            v_timeline.append(v_new)

        u = u_new
        v = v_new
    return u, v, u_timeline, v_timeline
