import numpy as np
from numba import njit

from MyPackage.DataContainers.Experiment import Experiment
from MyPackage.MathHelpers.AlgebraicLinearSystemsSolvers import tdma
from MyPackage.MathHelpers.ImplicitSchemeHelper import get_diags_implicit
from MyPackage.Models.SelkovStrogatz1D.SelkovStrogatz1DConfiguration import SelkovStrogatz1DConfiguration
from MyPackage.Models.TdmaParameters1D import TdmaParameters1D


def integrate_tdma_implicit_scheme(
    config: SelkovStrogatz1DConfiguration, method_config: TdmaParameters1D, eps: float = 1e-7
) -> Experiment:
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
    method_config["method"] = "tdma_implicit"
    if "seed" not in method_config:
        method_config["seed"] = np.random.randint(2147483647)
    y_init = method_config["u_init"]
    del method_config["u_init"]
    x_init = method_config["v_init"]
    del method_config["v_init"]
    steps = int(round(method_config["t_max"] / method_config["dt"]))
    np.random.seed(method_config["seed"])
    u, v, u_timeline, v_timeline = __integrate_tdma_implicit__(
        method_config["dt"],
        method_config["dx"],
        steps,
        model_config["a"],
        model_config["b"],
        model_config["Dx"],
        model_config["Dy"],
        x_init,
        y_init,
        method_config["save_timeline"],
        method_config["timeline_save_step_delta"],
        method_config["min_t"],
        method_config["noise_amp"],
        eps=eps,
    )
    if u is None:
        raise ArithmeticError("Selkov1d evaluation failed")
    res = Experiment()
    res.fill(
        model_config,
        method_config,
        {"u": y_init, "v": x_init},
        {"u": u, "v": v},
        {"u": np.vstack(u_timeline), "v": np.vstack(v_timeline)},
    )
    return res


@njit(fastmath=True)
def __get_right_vec_y_implicit__(
    x: np.array, y: np.array, a: float, b: float, t: float
) -> np.array:
    return y + (b - a * y - x**2 * y) * t


@njit(fastmath=True)
def __get_right_vec_x_implicit__(
    x: np.array, y: np.array, a: float, t: float
) -> np.array:
    return x + (-x + a * y + x**2 * y) * t


@njit(fastmath=True)
def __integrate_tdma_implicit__(
    d_t: float,
    d_h: float,
    steps: int,
    a: float,
    b: float,
    D_x: float,
    D_y: float,
    init_x: np.array,
    init_y: np.array,
    save_timeline: bool = False,
    timeline_save_step: int = 10_000,
    min_t: int = None,
    noise_amp: float = None,
    eps: float = 1e-7,
):
    """
    :returns: Tuple
        - u - is final U state;
        - v - is final V state;
        - u_timeline - is 2d-array contains timeline of U if save_timeline else None;
        - v_timeline - is 2d-array contains timeline of V if save_timeline else None;
    """
    diags_y = get_diags_implicit(d_t, d_h, D_y, init_y.shape[0])
    Y_u_d = diags_y["upper"]
    Y_m_d = diags_y["middle"]
    Y_l_d = diags_y["lower"]

    diags_x = get_diags_implicit(d_t, d_h, D_x, init_x.shape[0])
    X_u_d = diags_x["upper"]
    X_m_d = diags_x["middle"]
    X_l_d = diags_x["lower"]
    y = init_y.copy()
    x = init_x.copy()
    y_timeline = [init_y]
    x_timeline = [init_x]
    for i in range(steps):
        y_new = tdma(
            Y_u_d, Y_m_d.copy(), Y_l_d, __get_right_vec_y_implicit__(x, y, a, b, d_t)
        )
        x_new = tdma(
            X_u_d, X_m_d.copy(), X_l_d, __get_right_vec_x_implicit__(x, y, a, d_t)
        )

        if noise_amp:
            y_new += np.random.standard_normal(y_new.shape) * noise_amp
            x_new += np.random.standard_normal(x_new.shape) * noise_amp

        if i % 5000 == 4999:
            if not np.isfinite(y_new).all():
                return None, None, None, None
            if (min_t is None or i * d_t > min_t) and np.linalg.norm(y - y_new) < eps:
                break
            if np.linalg.norm(y) < eps:
                break

        if save_timeline and i % timeline_save_step == 0:
            y_timeline.append(y_new)
            x_timeline.append(x_new)

        y = y_new
        x = x_new
    return x, y, x_timeline, y_timeline
