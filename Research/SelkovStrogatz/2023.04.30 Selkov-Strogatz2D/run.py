import numpy as np
import os
from tqdm import tqdm
from joblib import Parallel, delayed
import pyarrow as pa
import pyarrow.parquet as pq

import sys
sys.path.append('../../..')


from MyPackage.PythonHelpers.IOHelpers import write_to_file, to_json_np_aware
from MyPackage.Models.SelkovStrogatz.SelkovStrogatz1DConfiguration import SelkovStrogatz1DConfiguration
from MyPackage.Models.SelkovStrogatz.Explicit2D import calc_n_iters
from MyPackage.Models.ExplicitParameters2D import ExplicitParameters2D

timeline_save_points = [] \
    + [i for i in range(1,10,1)] \
    + [i for i in range(10,100,10)] \
    + [i for i in range(100,1000,100)] \
    + [i for i in range(1000,10000,1000)] \
    + [i for i in range(10000,100000,10000)] \
    + [i for i in range(100000,1000000,100000)] \
    + [i for i in range(1000000,2000000,100000)] \
    + [i for i in range(2000000,3000000,100000)] \
    + [i for i in range(3000000,4000000,100000)] \
    + [i for i in range(4000000,5000000,100000)] \

timeline_save_points = tuple(timeline_save_points)

D_u = 0.001
D_v = 0.01
d_h = 0.01
d_t = 0.001
coeff_h_t = d_t / d_h / d_h
max_t = 1000

timeline_save_points = tuple(t for t in timeline_save_points if t < max_t/d_t)


BASE_MODEL_CONF = SelkovStrogatz1DConfiguration(0,0,D_u, D_v)
BASE_METHOD_CONF = ExplicitParameters2D(None, None, d_h, d_t, max_t, 1, timeline_save_points, 0.0)
RES_FOLDER = '/media/alexander/BigDisk/data/selk_strogg_2D_2023_04_30_test'

def save(folder, config_model, config_method, res_u, res_v):
    os.makedirs(folder)
    config_method['u_init'] = None
    config_method['v_init'] = None
    configs = to_json_np_aware({'model_conf': config_model.parameters, 'method_conf': config_method.parameters})
    for i in range(1, len(res_u)):
        res_u[i] = res_u[i].round(3)
        res_v[i] = res_v[i].round(3)
    write_to_file(f'{folder}/configs.json', configs)
    table = pa.table([res_u.ravel(), res_v.ravel()], names=["u", "v"])
    pq.write_table(table, f'{folder}/process.parquet',)


def sas(a, b, base_model_conf, base_method_conf, folder):
    
    model_config = base_model_conf.copy({'a':a, 'b':b})
    h_points = int(round(base_method_conf['h_max'] / base_method_conf['d_h'])) + 1
    x, y = model_config.get_stale_homogenous_solution()
    U = np.random.rand(h_points, h_points) * 0.1 + x
    V = np.random.rand(h_points, h_points) * 0.1 + y
    method_config = base_method_conf.copy({'u_init':U, 'v_init':V})
    try:
        n_iters = int(round(method_config['t_max'] / method_config['d_t'])) + 1
        res_u, res_v = calc_n_iters(
            method_config['u_init'],
            method_config['v_init'],
            model_config['Dx'],
            model_config['Dy'],
            method_config['coeff_h_t'],
            method_config['d_t'],
            model_config['a'],
            model_config['b'],
            n_iters,
            tuple(sorted(filter(lambda x: x != 0 and x != n_iters - 1, method_config['save_timeline_points']))))
        save(folder, model_config, method_config, res_u, res_v)
    except Exception as e:
        print('WAT', e)
        raise e


to_calc = []

for a in np.linspace(0.0025,0.065, 6):
    for i in range(10):
        to_calc.append((a, 1.25, BASE_MODEL_CONF, BASE_METHOD_CONF, f'{RES_FOLDER}/{a}_{1.25}_{i}'))

i = 0        
with tqdm(total=len(to_calc)) as bar:
    with Parallel(n_jobs=30) as parallel:
        bar.update(i)
        while i < len(to_calc):
            curr = parallel(
                delayed(sas)(a, b, base_model_conf, base_method_conf, folder) for a, b, base_model_conf, base_method_conf, folder in to_calc[i : i + 30]
            )
            i += 30
            bar.update(30)


