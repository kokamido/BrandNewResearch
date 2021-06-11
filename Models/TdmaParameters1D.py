from typing import Dict, Any

from nptyping import NDArray


class TdmaParameters1D:
    def __init__(self, u_init: NDArray, v_init: NDArray, dx: float, dt: float, t_max: float,
                 save_timeline: bool = False, timeline_save_step_delta: int = 10_000, min_t: int = None,
                 noise_amp: float = None, seed: int = None, x_right: float = None, x_left: float = None):
        assert u_init.shape == v_init.shape
        assert dx > 0
        assert dt > 0
        assert t_max > dt
        assert timeline_save_step_delta > 0
        if x_left is not None and x_right is not None:
            assert x_right > x_left
        self.__parameters__ = {'u_init': u_init, 'v_init': v_init, 'dx': dx, 'dt': dt, 't_max': t_max, 'min_t': min_t,
                               'save_timeline': save_timeline, 'timeline_save_step_delta': timeline_save_step_delta,
                               'noise_amp': noise_amp, 'seed': seed, 'x_right': x_right, 'x_left': x_left}
        self.__calc_additional_parameters__()

    def __getitem__(self, key):
        return self.__parameters__[key]

    def __setitem__(self, key, value):
        self.__parameters__[key] = value

    def copy(self, modification=None):
        res = from_params_dict(self.__parameters__)
        if modification is not None:
            for key in modification:
                res[key] = modification[key]
        return res

    def get_actual_dt_for_saved_transient(self):
        assert self.__parameters__['timeline_save_step_delta']
        assert self.__parameters__['dt']
        return self.__parameters__['timeline_save_step_delta'] * self.__parameters__['dt']

    def __calc_additional_parameters__(self):
        self.__parameters__['x_max'] = self.__parameters__['dx'] * len(self.__parameters__['v_init'])

    def __str__(self):
        return f'{self.__parameters__}'


def from_params_dict(params: Dict[str, Any]) -> TdmaParameters1D:
    return TdmaParameters1D(params['u_init'], params['v_init'], params['dx'],
                            params['dt'], params['t_max'], params.get('save_timeline', True),
                            params.get('timeline_save_step_delta', 100), params.get('min_t', None),
                            params.get('noise_amp', None), params.get('seed', None), params.get('x_right', None),
                            params.get('x_left', None))
