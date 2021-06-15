from typing import Dict, Any

from nptyping import ndarray

from MyPackage.Models.ConfigBase import ConfigBase


class TdmaParameters1D(ConfigBase):
    def __init__(self, u_init: ndarray, v_init: ndarray, dx: float, dt: float, t_max: float,
                 x_left: float, x_right: float,  save_timeline: bool = False,
                 timeline_save_step_delta: int = 100, min_t: int = None, noise_amp: float = None, seed: int = None):
        assert u_init.shape == v_init.shape
        assert dx > 0
        assert dt > 0
        assert t_max > dt
        assert not save_timeline or timeline_save_step_delta > 0
        assert x_left is not None
        assert x_right is not None
        assert x_right > x_left
        assert int(round((x_right - x_left) / dx)) == u_init.size - 1
        self.parameters = {'u_init': u_init, 'v_init': v_init, 'dx': dx, 'dt': dt, 't_max': t_max, 'min_t': min_t,
                               'save_timeline': save_timeline, 'timeline_save_step_delta': timeline_save_step_delta,
                               'noise_amp': 0.0, 'seed': seed, 'x_right': x_right, 'x_left': x_left}


    def copy(self, modification=None):
        res = from_params_dict(self.parameters)
        if modification is not None:
            for key in modification:
                res[key] = modification[key]
        return res

    def get_actual_dt_for_saved_transient(self):
        assert self.parameters['timeline_save_step_delta']
        assert self.parameters['dt']
        return self.parameters['timeline_save_step_delta'] * self.parameters['dt']


def from_params_dict(params: Dict[str, Any]) -> TdmaParameters1D:
    return TdmaParameters1D(params['u_init'], params['v_init'], params['dx'],
                            params['dt'], params['t_max'], params.get('x_left', None), params.get('x_right', None),
                            params.get('save_timeline', True), params.get('timeline_save_step_delta', 100),
                            params.get('min_t', None), params.get('noise_amp', 0.0), params.get('seed', None))
