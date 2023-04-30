from typing import Union, List, Optional, Dict, Tuple

import numpy as np
from numba import njit
from scipy.integrate import simps

from MyPackage.DataContainers.Experiment import Experiment
from MyPackage.DataContainers.ExperimentHelper import convert_time_to_indices
from MyPackage.DataContainers.PeakPatternType import PeakPatternType, PatternDirection


def calc_mean_squared_Fourier_for_experiment(ex: Experiment, coeff: Union[float, List[float]], var: str,
                                             left_border: Optional[float] = None, right_border: Optional[float] = None) \
        -> Dict[float, float]:
    left_border, right_border = convert_time_to_indices(ex, left_border, right_border)
    return calc_mean_squared_Fourier_for_transient(ex.timelines[var][left_border:right_border, ],
                                                   ex.method_parameters['dt'],
                                                   ex.method_parameters['dx'],
                                                   coeff)


def calc_mean_squared_Fourier_for_transient(trans: np.ndarray, dt: float, dx: float, coeff: Union[float, List[float]]) \
        -> Dict[float, float]:
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


def calc_presence(ex: Experiment, ks: List[float], var_to_calc: str, left_border_t: float = None,
                  right_border_t: float = None) -> Dict[float, np.ndarray]:
    coeffs, _ = calc_few_Fourier_coeffs_for_experiment(ex, ks, var_to_calc, left_border_t, right_border_t)
    presence = np.apply_along_axis(np.argmax, 0, np.abs(coeffs))
    res = {}
    for num, k in enumerate(ks):
        res[k] = (presence == num).sum() / float(len(presence))
    return res


def calc_Fourier_coeff_for_experiment(ex: Experiment, coeff: float, var: str,
                                      left_border: Optional[float] = None,
                                      right_border: Optional[float] = None
                                      ) -> np.ndarray:
    left_border, right_border = convert_time_to_indices(ex, left_border, right_border)
    return calc_Fourier_coeff_for_transient(ex.timelines[var][left_border:right_border, ],
                                            ex.method_parameters['dx'],
                                            coeff)


# improved integration
@njit(fastmath=True)
def simpson_nb(y, dx):
    s = y[0] + y[-1]

    n = y.shape[0] // 2
    for i in range(n - 1):
        s += 4. * y[i * 2 + 1]
        s += 2. * y[i * 2 + 2]

    s += 4 * y[(n - 1) * 2 + 1]
    return (dx / 3.) * s


@njit(fastmath=True)
def calc_Fourier_coeff_for_pattern(points: np.ndarray, dx: float, coeff: float) -> float:
    xs = np.linspace(0, points.shape[0] - 1, points.shape[0]) * dx
    x_max = (points.shape[0] - 1) * dx
    cos_mul = np.cos(2 * np.pi * coeff * xs / x_max)
    if coeff % 1 == .5:
        cos_mul *= -1
    return simpson_nb(points * cos_mul, xs[1] - xs[0])

def calc_multiple_Fourier_coeffs(points: np.ndarray, dx: float, coeffs:np.ndarray):
    res = {}
    for c in coeffs:
        res[c] = calc_Fourier_coeff_for_pattern(points, dx, c)
    return res


def calc_Fourier_coeff_for_transient(trans: np.ndarray, dx: float, coeff: float) -> np.ndarray:
    return np.apply_along_axis(lambda x: calc_Fourier_coeff_for_pattern(x, dx, coeff), 1, trans)


def calc_few_Fourier_coeffs_for_experiment(e: Experiment, ks, var_to_calc: str, left_border_t: float = None,
                                           right_border_t: float = None) -> Tuple[np.ndarray, np.ndarray]:
    dt = e.method_parameters['dt'] * \
         e.method_parameters['timeline_save_step_delta']
    assert e.timelines is not None and var_to_calc in e.timelines
    left_border, right_border = convert_time_to_indices(e, left_border_t, right_border_t)
    ts = np.array(np.linspace(left_border, right_border, right_border - left_border) * dt)
    res = np.zeros(shape=(len(ks), ts.size))
    for i, k in enumerate(ks):
        res[i, :] = calc_Fourier_coeff_for_experiment(e, k, var_to_calc, left_border_t, right_border_t)
    return res, ts


def calc_peaks_by_Fourier(ex: Experiment, var: str, max_peaks_count: float = 10.0, min_amplitude=0.1) -> PeakPatternType:
    if (max(ex.end_values[var]) - min(ex.end_values[var])) < min_amplitude:
        return None
    return _calc_peaks_by_Fourier(ex.end_values[var], ex.method_parameters['dx'], max_peaks_count)


def _calc_peaks_by_Fourier(points: np.ndarray, dx: float, max_peaks_count: float = 10.0) -> PeakPatternType:
    """Returns picks count in cos-like stricture
    Format {picks: float, direction: up/down}
    """
    coeffs_to_check = np.linspace(.5, max_peaks_count, int(max_peaks_count * 2))
    coeffs = np.apply_along_axis(lambda c: calc_Fourier_coeff_for_pattern(
        points, dx, float(c)), 1, coeffs_to_check.reshape(coeffs_to_check.size, 1))
    max_coeff_index = np.argmax(np.abs(coeffs))
    peaks = coeffs_to_check[max_coeff_index]
    direction = PatternDirection.DOWN if coeffs[max_coeff_index] < 0 else PatternDirection.UP
    return PeakPatternType(peaks, direction)
