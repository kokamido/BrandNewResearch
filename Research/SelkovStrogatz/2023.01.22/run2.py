import sys
sys.path.append('/home/alexander/Desktop/univer/BrandNewResearch')
from MyPackage.Models.SelkovStrogatz.SelkovStrogatz1DConfiguration import (
    SelkovStrogatz1DConfiguration,
)
from MyPackage.Models.SelkovStrogatz.SelkovStrogatz1DTdmaSolver import (
    integrate_tdma_implicit_scheme,
    TdmaParameters1D,
)
from MyPackage.MathHelpers.InitDataHelpers import *


from tqdm import tqdm
import os
from joblib import Parallel, delayed
import numpy as np


def sas(a, b, dirr, i, j):
    config = SelkovStrogatz1DConfiguration(a=a, b=b, Dy=0.1, Dx=0.01)
    x, y = config.get_stale_homogenous_solution()
    x_init = get_normal_rand(501, x, 0.5)
    y_init = get_normal_rand(501, y, 0.5)
    tdma_parameters = TdmaParameters1D(
        u_init=y_init,
        v_init=x_init,
        dt=0.001,
        dx=0.01,
        save_timeline=True,
        t_max=1000,
        x_left=0,
        x_right=5,
        noise_amp=0,
    )
    try:
        res = integrate_tdma_implicit_scheme(
            config=config, method_config=tdma_parameters, eps=1e-7
        )
    except:
        print(a, b)
        return False
    os.mkdir(f"{dirr}/{i}_{j}")
    res.save(f"{dirr}/{i}_{j}")
    return True


aa = [round(x / 400, 4) for x in range(1, 56)][::5]
bb = [round(x / 100, 3) for x in range(60, 140)][::5]
to_calc = []
j = 0
for a in aa:
    for b in bb:
        for _ in range(500):
            to_calc.append((a, b, j))
            j+=1

dirr = "/media/alexander/BigDisk/data/fst_selk_strogg_2023_01_22_range_5"
os.makedirs(dirr, exist_ok=True)

i = len(os.listdir(dirr))
print(i)

with tqdm(total=len(to_calc)) as bar:
    with Parallel(n_jobs=30) as parallel:
        bar.update(i)
        while i < len(to_calc):
            curr = parallel(
                delayed(sas)(a, b, dirr, i, j) for a, b, j in to_calc[i : i + 3000]
            )
            i += 3000
            bar.update(3000)

