from typing import Dict, Union

import numpy as np
from nptyping import NDArray

from scipy.integrate import simps
from DataContainers.Experiment import Experiment


def calc_Fourier_coeff_for_pattern(points: NDArray[np.float], dx: float, coeff: float) -> float:
    xs = np.linspace(0, points.shape[0] - 1, points.shape[0]) * dx
    x_max = (points.shape[0] - 1) * dx
    cos_mul = np.cos(2 * np.pi * coeff * xs / x_max)
    return simps(points*cos_mul, xs)

def calc_Fourier_coeff_for_transient(trans: NDArray[np.float], dx: float, coeff: float, right_border_index: int) -> NDArray[np.float]:
    if not right_border_index:
        right_border_index = trans.shape[0] - 1
    return np.apply_along_axis(lambda x: calc_Fourier_coeff_for_pattern(x,dx,coeff), 1, trans[:right_border_index,:])


def calc_picks(points: NDArray[np.float64], min_amplitude: float = 0.0001) -> Dict[str, Union[float, str, None]]:
    """Returns picks count in cos-like stricture
    Format {picks: float, direction: up/down}
    Returns 0 picks if amplitude in array < min_amplitude
    """
    if np.max(points) - np.min(points) < min_amplitude:
        return {'picks': 0, 'direction': None}
    derivative_sign = 1 if points[2] - points[1] >= 0 else -1
    half_picks = 0
    for i in range(2, len(points) - 3):
        if (points[i + 1] - points[i]) / derivative_sign < 0:
            derivative_sign *= -1
            half_picks += 1
    return {'picks': half_picks / 2 + (0.5 if half_picks >= 0 else 0),
            'direction': 'down' if derivative_sign < 0 else 'up'}


def add_peaks_stats_Higgins1D(e: Experiment) -> Experiment:
    peaks_u = calc_picks(e.end_values['u'])
    peaks_v = calc_picks(e.end_values['v'])
    e.metadata['end_picks'] = {'u': peaks_u, 'v': peaks_v}
    return e


def add_amp_stats_Higgins1D(e: Experiment) -> Experiment:
    end_u, end_v = e.end_values.values()
    e.metadata['end_amps'] = {'u': max(end_u) - min(end_u), 'v': max(end_v) - min(end_v)}
    e.metadata['maxs'] = {'u': max(end_u), 'v': max(end_v)}
    return e
