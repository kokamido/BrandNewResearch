import copy

from MyPackage.DataAnalyzers.PeaksAnalyzer import calc_peaks_by_Fourier
from MyPackage.DataContainers.Experiment import Experiment
from MyPackage.Models.Higgins1D.Higgins1DConfiguration import Higgins1DConfiguration
from MyPackage.Models.Higgins1D.Higgins1DTdmaSolver import integrate_tdma_implicit_scheme
from MyPackage.Models.TdmaParameters1D import TdmaParameters1D


def check_robustness(path: str):
    e = Experiment()
    e.fill_from_file(path, load_timelines=True)
    conf = copy.deepcopy(Higgins1DConfiguration.from_params_dict(e.model_config))
    params = copy.deepcopy(e.method_parameters)
    params['u_init'], params['v_init'] = e.init_values.values()
    params['dt'] /= 10
    params['min_t'] = 2000
    p = TdmaParameters1D.from_params_dict(params)
    new_data = integrate_tdma_implicit_scheme(conf, p)
    if calc_peaks_by_Fourier(new_data.end_values['u'], 10) == calc_peaks_by_Fourier(e.end_values['u'], 10):
        print(f'Good {path}')
    else:
        print(f'Bad {path}')
    return e, new_data
