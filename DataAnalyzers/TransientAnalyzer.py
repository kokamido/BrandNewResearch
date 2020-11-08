from numpy import zeros, ndarray, apply_along_axis
from numpy.linalg import norm
from typing import Dict

from DataContainers.Experiment import Experiment


def calc_deltas_for_timeline_1D(e: Experiment) -> Dict[str, ndarray]:
    assert e.timelines
    res = {}
    for k in e.timelines:
        tail = e.timelines[k].T
        tail_shifted = zeros(tail.shape)
        tail_shifted[:, 1:] = tail[:, :-1]
        res[k] = apply_along_axis(norm, 0, (tail - tail_shifted)[:, -tail.shape[1] + 1:])
    return res