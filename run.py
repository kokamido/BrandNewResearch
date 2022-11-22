from MyPackage.Models.SelkovStrogatz1D.SelkovStrogatz1DConfiguration import SelkovStrogatz1DConfiguration
from MyPackage.Models.SelkovStrogatz1D.SelkovStrogatz1DTdmaSolver import integrate_tdma_implicit_scheme, TdmaParameters1D
from MyPackage.MathHelpers.InitDataHelpers import *

from MyPackage.Drawing.DrawTransient1D import draw_transient
from MyPackage.Drawing.DrawHelper import set_defaults_1D
from MyPackage.ResearchHelpers.UsefulNotebookDrawings import draw_quadreega
from MyPackage.DataAnalyzers.PeaksAnalyzer import calc_presence

from tqdm import tqdm
import os
import pylab as plt
from joblib import Parallel, delayed
import numpy as np
import seaborn as sns


def sas(a, b):
    config = SelkovStrogatz1DConfiguration(a=a, b=b, Dy=0.1, Dx=0.01)
    x, y = config.get_stale_homogenous_solution()
    x_init = get_cos(2, 201, x, .5)
    y_init = get_cos(2, 201, y, .5)
    tdma_parameters = TdmaParameters1D(
        u_init = y_init,
        v_init = x_init, 
        dt=0.00025,
        dx=0.005,
        save_timeline=True,
        t_max=1000,
        x_left=0,
        x_right=1, noise_amp=0
        )
    try:
        res = integrate_tdma_implicit_scheme(config=config, method_config=tdma_parameters, eps=1e-7)
    except:
        return None
    return res

aa = [round(x/400, 4) for x in range(1,61)]
bb = [round(x/100, 3) for x in range(21,150)]
to_calc = []
for a in aa:
    for b in bb:
        to_calc.append((a, b))

dirr = '/media/alexander/BigDisk/data/fst_selk_strogg'
#os.mkdir(dirr)

i = len(os.listdir(dirr))
print(i)

with tqdm(total=len(to_calc)) as bar:
    with Parallel(n_jobs=30) as parallel:
        bar.update(i)
        while i < len(to_calc):
            curr = parallel(delayed(sas)(a,b) for a,b in to_calc[i:i+280])
            for j, exp in enumerate(curr):
                os.mkdir(f'{dirr}/{i+j}')
                exp.save(f'{dirr}/{i+j}')
            i += 280
            bar.update(280)
    
