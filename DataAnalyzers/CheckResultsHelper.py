import copy

from DataContainers.Experiment import Experiment
from DataAnalyzers.PicksAnalyzer import calc_picks
from Models.Higgins1D.Higgins1DTdmaSolver import integrate_tdma_implicit_scheme
from Models.Higgins1D.Higgins1DConfiguration import Higgins1DConfiguration, Higgins1DTdmaParameters


def check_robustness(path: str):
    e = Experiment()
    e.fill_from_file(path, load_timelines=True)
    conf = copy.deepcopy(Higgins1DConfiguration.from_params_dict(e.model_config))
    params = copy.deepcopy(e.method_parameters)
    params['u_init'], params['v_init'] = e.init_values.values()
    params['dt'] /= 10
    params['min_t'] = 2000
    p = Higgins1DTdmaParameters.from_params_dict(params)
    new_data = integrate_tdma_implicit_scheme(conf, p)
    if calc_picks(new_data.end_values['u']) == calc_picks(e.end_values['u']):
        print(f'Good {path}')
    else:
        print(f'Bad {path}')
    return e, new_data