from typing import Dict, Any


class Selkov1DConfiguration:

    def __init__(self, n: float, w: float, Du: float, Dv: float):
        assert Du > 0
        assert Dv > 0
        self.parameters = {'w': w, 'n': n,
                           'Du': Du, 'Dv': Dv, 'model': 'Selkov'}

    def __str__(self):
        return f'{self.parameters}'

    def __getitem__(self, key):
        return self.__parameters__[key]

    def __setitem__(self, key, value):
        self.__parameters__[key] = value

    def get_stale_homogenous_solution(self):
        n = self.parameters['n']
        w = self.parameters['w']
        v = n / w
        u = n / v ** 2
        return u, v

    def copy(self, patch: Dict[str, Any] = None):
        res = from_params_dict(self.parameters)
        if patch is not None:
            for key in patch:
                res[key] = patch[key]
        return res


def from_params_dict(params: Dict[str, Any]):
    return Selkov1DConfiguration(params['n'], params['w'], params['Du'], params['Dv'])
