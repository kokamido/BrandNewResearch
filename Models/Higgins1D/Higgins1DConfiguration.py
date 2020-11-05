from nptyping import NDArray

from PythonHeplers.IOHelpers import load_python_array


class Higgins1DConfiguration:

    def __init__(self, p: float, q: float, Du: float, Dv: float):
        assert p > 0
        assert q > 0
        assert Du > 0
        assert Dv > 0
        self.parameters = {'p': p, 'q': q, 'Du': Du, 'Dv': Dv, 'model': 'Higgins'}

    def __str__(self):
        return f'{self.parameters}'


class Higgins1DTdmaParameters:
    def __init__(self, u_init: NDArray, v_init: NDArray, dx: float, dt: float, t_max: float,
                 save_timeline: bool = False,
                 timeline_save_step_delta: int = 10_000):
        assert u_init.shape == v_init.shape
        assert dx > 0
        assert dt > 0
        assert t_max > dt
        assert timeline_save_step_delta > 0
        self.parameters = {'u_init': u_init, 'v_init': v_init, 'dx': dx, 'dt': dt, 't_max': t_max,
                           'save_timeline': save_timeline, 'timeline_save_step_delta': timeline_save_step_delta}
        self.__calc_additional_parameters__()

    def __calc_additional_parameters__(self):
        self.parameters['x_max'] = self.parameters['dx'] * len(self.parameters['v_init'])

    def __str__(self):
        return f'{self.parameters}'
