import sys
sys.path.insert(1, '../../')

from DataContainers.Experiment import Experiment
from MathHelpers.InitDataHelpers import get_cos, get_normal_rand
from Models.Higgins1D.Higgins1DTdmaSolver import integrate_tdma_implicit_scheme
from Models.Higgins1D.Higgins1DConfiguration import Higgins1DConfiguration, Higgins1DTdmaParameters
from tqdm import tqdm
import numpy as np
import threading, queue


to_write = queue.Queue()
base_dir = 'D:/math/15.01.2021_for_boxplots'
def writer():
    for e, file in iter(to_write.get, None):
        e.save(file)

threading.Thread(target=writer).start()

conf = Higgins1DConfiguration(2.0, 2.0, 20.0, 1.0)
for q in np.arange(2, 1, -.1):
    q = round(q, 2)
    subdir = f'{base_dir}/q_{q}'
    for peaks_to_start in tqdm(np.arange(0.5, 500.5, 0.5)):
        conf.parameters['q'] = q
        u_init = get_cos(peaks_to_start, int(round(200*0.2/0.05)), 1.0, 0.1)
        v_init = get_cos(peaks_to_start, int(round(200*0.2/0.05)), 1.0, 0.1)
        params = Higgins1DTdmaParameters(
            u_init, v_init, 0.05, 0.05, 2500.0, save_timeline=True, timeline_save_step_delta=10, noise_amp=0.01)
        e = integrate_tdma_implicit_scheme(conf, params)
        del e.timelines['v']
        e.timelines['u_amps'] = np.apply_along_axis(lambda x: x.max() - x.min(), 1, e.timelines['u'])
        e.timelines['u'] = e.timelines['u'][:,::10]
        to_write.put((e, f'{subdir}/peaks_{peaks_to_start}'))


for q in np.arange(2, 1, -.1):
    q = round(q, 2)
    subdir = f'{base_dir}/q_{q}'
    for num in tqdm(range(1000)):
        conf.parameters['q'] = q
        u_init = get_normal_rand(int(round(200*0.2/0.05)), 1.0, 0.1)
        v_init = get_normal_rand(int(round(200*0.2/0.05)), 1.0, 0.1)
        params = Higgins1DTdmaParameters(
            u_init, v_init, 0.05, 0.05, 2500.0, save_timeline=True, timeline_save_step_delta=10, noise_amp=0.01)
        e = integrate_tdma_implicit_scheme(conf, params)
        del e.timelines['v']
        e.timelines['u_amps'] = np.apply_along_axis(lambda x: x.max() - x.min(), 1, e.timelines['u'])
        e.timelines['u'] = e.timelines['u'][:,::10]
        to_write.put((e, f'{subdir}/rand_{num}'))

to_write.put(None)
