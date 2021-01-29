import sys
sys.path.insert(1, '../../')

from Models.Selkov1D.Selkov1DTdmaSolver import integrate_tdma_implicit_scheme
from Models.Selkov1D.Selkov1DConfiguration import Selkov1DConfiguration
from Models.TdmaParameters1D import TdmaParameters1D
from MathHelpers.InitDataHelpers import get_cos, get_normal_rand
from matplotlib import pyplot as plt
from Drawing.DrawHelper import set_defaults_1D
from Drawing.DrawTransient1D import draw_transient
from DataContainers.Experiment import Experiment
from DataAnalyzers.PeaksAnalyzer import calc_peacks
from collections import OrderedDict
from tqdm import tqdm
import seaborn as sns
import numpy as np


set_defaults_1D()

n = 1.1
w = 1.0
u_hat = w ** 2 / n
v_hat = n / w
conf = Selkov1DConfiguration(n, w, 0.007052, 0.001)

palette = sns.color_palette('tab10')
color_indices = {}
for peaks_count in tqdm(range(5)):
    u_init = np.ones(100) * u_hat
    v_init = np.ones(100) * v_hat
    params = TdmaParameters1D(u_init, v_init, 0.01, 0.01, 1000.0, save_timeline=True, timeline_save_step_delta=10,noise_amp=0.0075,min_t=1000)
    e = integrate_tdma_implicit_scheme(conf, params)
    curve = e.end_values['u']
    pcks = calc_peacks(curve)
    peaks = pcks['peaks']
    direction = pcks['direction']
    if peaks not in color_indices:
        color_indices[peaks] = len(color_indices)
    ind = color_indices[peaks]
    plt.plot(curve,c=palette[ind], lw=3,label = f'{peaks} {direction}')
plt.title(', '.join([f"{key}: {e.model_config[key]}" for key in ['n','w','Du','Dv']])+f', eps: {e.method_parameters["noise_amp"]}, t_max: {e.timelines["u"].shape[0]*e.method_parameters["dt"]*e.method_parameters["timeline_save_step_delta"]}')
plt.show()

