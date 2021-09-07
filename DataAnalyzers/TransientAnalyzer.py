from typing import Dict, Tuple

import numpy as np
from nptyping import NDArray
from numpy.linalg import norm

from MyPackage.DataAnalyzers.PeaksAnalyzer import calc_peaks_by_Fourier
from MyPackage.DataContainers.Experiment import Experiment


def calc_deltas_for_timeline_1D(e: Experiment) -> Dict[str, NDArray]:
    assert e.timelines
    res = {}
    for k in e.timelines:
        tail = e.timelines[k].T
        tail_shifted = np.zeros(tail.shape)
        tail_shifted[:, 1:] = tail[:, :-1]
        res[k] = np.apply_along_axis(
            norm, 0, (tail - tail_shifted)[:, -tail.shape[1] + 1:])
    return res


def calc_min_and_max_dynamic(transient: NDArray) -> Tuple[NDArray, NDArray]:
    """
    Возвращает временные ряды, содержащие минимумы и максимумы паттерна в каждый момент времени
    """
    assert transient
    res = np.array(np.apply_along_axis(lambda x: (x.min(), x.max()), 1, transient))
    return res[0, :], res[1, :]


def add_peaks_stats_Higgins1D(e: Experiment) -> Experiment:
    dx = e.method_parameters['dx']
    peaks_u = calc_peaks_by_Fourier(e.end_values['u'], dx)
    peaks_v = calc_peaks_by_Fourier(e.end_values['v'], dx)
    e.metadata['end_picks'] = {'u': peaks_u, 'v': peaks_v}
    return e


def add_amp_stats_Higgins1D(e: Experiment) -> Experiment:
    end_u, end_v = e.end_values.values()
    e.metadata['end_amps'] = {
        'u': max(end_u) - min(end_u), 'v': max(end_v) - min(end_v)}
    e.metadata['maxs'] = {'u': max(end_u), 'v': max(end_v)}
    return e
