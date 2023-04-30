from typing import Dict, Any
import numpy as np
from MyPackage.Models.ConfigBase import ConfigBase


class ExplicitParameters2D(ConfigBase):
    def __init__(self, u_init: np.array, v_init: np.array, d_h: float, d_t: float, t_max: float,
                 h_max: float,  save_timeline_points: list[int] = None, noise_amp: float = 0.0, seed: int = None):
        
        self.parameters = {
        'u_init' : u_init,
        'v_init' : v_init,
        'd_h' : d_h,
        'd_t' : d_t,
        't_max' : t_max,
        'h_max' : h_max,
        'save_timeline_points' : save_timeline_points,
        'noise_amp' : noise_amp,
        'seed' : seed,
        'coeff_h_t': d_t / d_h / d_h}

    def copy(self, modification=None):
        res = self.from_params_dict(self.parameters)
        if modification is not None:
            for key in modification:
                res[key] = modification[key]
        return res
    
    @staticmethod
    def from_params_dict(params: Dict[str, Any]):
        return ExplicitParameters2D(params['u_init'], params['v_init'], params['d_h'],
                                params['d_t'], params['t_max'], params.get('h_max'), params.get('save_timeline_points', None),
                                params.get('noise_amp', 0.0),  params.get('seed', None))
