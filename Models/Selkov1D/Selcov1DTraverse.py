from typing import List, Tuple, Iterable

import numpy as np

from MyPackage.DataContainers.Experiment import Experiment
from MyPackage.MathHelpers.InitDataHelpers import get_cos
from MyPackage.MathHelpers.PatternTraverse import traverse_recursive
from MyPackage.Models.Selkov1D.Selkov1DConfiguration import Selkov1DConfiguration
from MyPackage.Models.Selkov1D.Selkov1DTdmaSolver import integrate_tdma_implicit_scheme
from MyPackage.Models.TdmaParameters1D import TdmaParameters1D


def __traverse_single__(conf: Tuple[Selkov1DConfiguration, TdmaParameters1D],
                        init_u_and_v: Tuple[np.array, np.array]) -> Experiment:
    system_conf, method_conf = conf
    method_conf['u_init'], method_conf['v_init'] = init_u_and_v
    return integrate_tdma_implicit_scheme(system_conf, method_conf)


def __make_new_init_data__(ex: Experiment) -> Tuple[np.array, np.array]:
    return ex.end_values.values()


def traverse(conf_base: Selkov1DConfiguration, params: TdmaParameters1D, du_to_traverse: List[float],
             init_and_dus: Iterable[Tuple[float, float]], verbose: bool = False) -> Iterable[Experiment]:
    du_to_traverse = sorted(du_to_traverse)
    selkov_conf_base = conf_base.copy()
    for init_data, start_du in init_and_dus:
        res = []

        selkov_conf_base.parameters['Du'] = start_du
        assert params['x_right'] is not None
        assert params['x_left'] is not None
        dus_to_forward_traverse = [d for d in du_to_traverse if d > start_du]
        confs_to_forward_traverse = [selkov_conf_base.copy({'Du': du}) for du in dus_to_forward_traverse]

        dus_to_backward_traverse = [d for d in du_to_traverse if d <= start_du][::-1]
        confs_to_backward_traverse = [selkov_conf_base.copy({'Du': du}) for du in dus_to_backward_traverse]

        for e in traverse_recursive(__traverse_single__, [(c, params.copy()) for c in confs_to_forward_traverse],
                                    __make_new_init_data__, init_data, verbose=verbose):
            res.append(e)

        for e in traverse_recursive(__traverse_single__, [(c, params.copy()) for c in confs_to_backward_traverse],
                                    __make_new_init_data__, init_data, verbose=verbose):
            res.append(e)
        yield res
