import copy

from MyPackage.DataAnalyzers.PeaksAnalyzer import _calc_peaks_by_Fourier
from MyPackage.DataContainers.Experiment import Experiment


def check_robustness(path: str, config_type, parameters_type, scheme, t_max, dt_div=10):
    e = Experiment()
    e.fill_from(path)
    conf = copy.deepcopy(config_type.from_params_dict(e.model_config))
    params = copy.deepcopy(e.method_parameters)
    print(params)
    params['dt'] /= dt_div
    params['t_max'] = t_max
    print(params)
    params['u_init'], params['v_init'] = e.init_values.values()
    p = parameters_type.from_params_dict(params)
    new_data = scheme(conf, p)
    return e, new_data
