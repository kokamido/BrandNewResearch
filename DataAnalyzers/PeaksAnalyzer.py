from typing import Union, List, Optional, Dict, Tuple

import numpy as np
from scipy.integrate import simps

from MyPackage.DataContainers.Experiment import Experiment
from MyPackage.DataContainers.ExperimentHelper import convert_time_to_indices


def calc_mean_squared_Fourier_for_experiment(ex: Experiment, coeff: Union[float, List[float]], var: str,
                                             left_border: Optional[int] = None,
                                             right_border: Optional[int] = None
                                             ) -> Dict[float, np.array]:
    left_border, right_border = convert_time_to_indices(ex, left_border, right_border)
    return calc_mean_squared_Fourier_for_transient(ex.timelines[var][left_border:right_border, ],
                                                   ex.method_parameters['dt'],
                                                   ex.method_parameters['dx'],
                                                   coeff)


def calc_mean_squared_Fourier_for_transient(trans: np.ndarray, dt: float, dx: float,
                                            coeff: Union[float, List[float]]) -> Dict[float, np.ndarray]:
    if isinstance(coeff, float):
        coeff = [coeff]
    res = {}
    for c in coeff:
        fouriers_coeffs = calc_Fourier_coeff_for_transient(trans, dx, c)
        t_steps_count = trans.shape[0]
        timeline = np.arange(0, t_steps_count, 1) * dt
        t_max = t_steps_count * dt
        res[c] = simps(fouriers_coeffs ** 2, timeline) / t_max
    return res


def calc_Fourier_coeff_for_experiment(ex: Experiment, coeff: float, var: str,
                                      left_border: Optional[float] = None,
                                      right_border: Optional[float] = None
                                      ) -> np.ndarray:
    left_border, right_border = convert_time_to_indices(ex, left_border, right_border)
    return calc_Fourier_coeff_for_transient(ex.timelines[var][left_border:right_border, ],
                                            ex.method_parameters['dx'],
                                            coeff)


def calc_Fourier_coeff_for_pattern(points: np.ndarray, dx: float, coeff: float) -> float:
    xs = np.linspace(0, points.shape[0] - 1, points.shape[0]) * dx
    x_max = (points.shape[0] - 1) * dx
    cos_mul = np.cos(2 * np.pi * coeff * xs / x_max)
    if coeff % 1 == .5:
        cos_mul *= -1
    return simps(points * cos_mul, xs)


def calc_Fourier_coeff_for_transient(trans: np.ndarray, dx: float, coeff: float) -> np.ndarray:
    return np.apply_along_axis(lambda x: calc_Fourier_coeff_for_pattern(x, dx, coeff), 1, trans)


def calc_few_Fourier_coeffs_for_experiment(e: Experiment, ks, var_to_calc: str, right_border_t: float = None,
                                           left_border_t: float = None) -> Tuple[np.ndarray, np.ndarray]:
    dt = e.method_parameters['dt'] * \
         e.method_parameters['timeline_save_step_delta']
    assert e.timelines is not None and var_to_calc in e.timelines
    left_border, right_border = convert_time_to_indices(e, left_border_t, right_border_t)
    ts = np.array(np.linspace(left_border, right_border, right_border - left_border) * dt)
    res = np.zeros(shape=(len(ks), ts.size))
    for i, k in enumerate(ks):
        res[i, :] = calc_Fourier_coeff_for_experiment(e, k, var_to_calc, left_border_t, right_border_t)
    return res, ts


def calc_peaks_by_Fourier(points: np.ndarray, dx: float, max_peaks_count: float = 10.0) -> Dict[
    str, Union[float, str, None]]:
    """Returns picks count in cos-like stricture
    Format {picks: float, direction: up/down}
    Returns 0 picks if amplitude in array < min_amplitude
    """
    coeffs_to_check = np.linspace(.5, max_peaks_count, int(max_peaks_count * 2))
    coeffs = np.apply_along_axis(lambda c: calc_Fourier_coeff_for_pattern(
        points, dx, c), 1, coeffs_to_check.reshape(coeffs_to_check.size, 1))
    max_coeff_index = np.argmax(np.abs(coeffs))
    peaks = coeffs_to_check[max_coeff_index]
    direction = 'down' if coeffs[max_coeff_index] < 0 else 'up'
    return {'peaks': peaks, 'direction': direction}
