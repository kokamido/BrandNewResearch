from typing import Dict, Union, List

import numpy as np
from nptyping import NDArray
from scipy.integrate import simps


def calc_mean_squared_Fourier_for_transient(trans: NDArray[np.float], dt: float, dx: float,
                                            coeff: Union[float, List[float]], right_border_index: int = None) \
        -> Dict[float, float]:
    if isinstance(coeff, float):
        coeff = [coeff]
    res = {}
    for c in coeff:
        fouriers_coeffs = calc_Fourier_coeff_for_transient(trans, dx, c, right_border_index)
        t_steps_count = trans.shape[0]
        timeline = np.arange(0, t_steps_count - 1, 1) * dt
        t_max = t_steps_count * dt
        res[c] = simps(fouriers_coeffs ** 2, timeline) / t_max
    return res


def calc_Fourier_coeff_for_pattern(points: NDArray[np.float], dx: float, coeff: float) -> float:
    xs = np.linspace(0, points.shape[0] - 1, points.shape[0]) * dx
    x_max = (points.shape[0] - 1) * dx
    cos_mul = np.cos(2 * np.pi * coeff * xs / x_max)
    if coeff % 1 == .5:
        cos_mul *= -1
    return simps(points * cos_mul, xs)


def calc_Fourier_coeff_for_transient(trans: NDArray[np.float], dx: float, coeff: float, left_border_index: int = None,
                                     right_border_index: int = None) -> NDArray[np.float]:
    if not right_border_index:
        right_border_index = trans.shape[0] - 1
    if not left_border_index:
        left_border_index = 0
    return np.apply_along_axis(lambda x: calc_Fourier_coeff_for_pattern(x, dx, coeff), 1,
                               trans[left_border_index:right_border_index, :])


def calc_peaks_by_Fourier(points: NDArray[np.float64], dx: float, max_peaks_count: float = 10.0) -> Dict[
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
