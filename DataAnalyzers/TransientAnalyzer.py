from numpy import zeros, ndarray, apply_along_axis
from numpy.linalg import norm
from typing import Dict, Tuple

from DataContainers.Experiment import Experiment


def calc_deltas_for_timeline_1D(e: Experiment) -> Dict[str, ndarray]:
    assert e.timelines
    res = {}
    for k in e.timelines:
        tail = e.timelines[k].T
        tail_shifted = zeros(tail.shape)
        tail_shifted[:, 1:] = tail[:, :-1]
        res[k] = apply_along_axis(
            norm, 0, (tail - tail_shifted)[:, -tail.shape[1] + 1:])
    return res


def calc_min_and_max_dynamic(transient: ndarray) -> Dict[str, Tuple[ndarray, ndarray]]:
    """
    Возвращяет временные ряды, содержащие минимумы и максимумы паттерна в каждый момент времени
    """
    assert transient
    res = np.array(apply_along_axis(lambda x: (x.min(), x.max()), 1, transient))
    return res[0, :], res[1, :]
