from typing import Dict, Any

from MyPackage.Models.ConfigBase import ConfigBase


class SelkovStrogatz1DConfiguration(ConfigBase):
    def __init__(self, a: float, b: float, Dx: float, Dy: float):
        assert Dx > 0
        assert Dy > 0
        self.parameters = {'a': a, 'b': b,
                               'Dx': Dx, 'Dy': Dy, 'model': 'Selkov-Strogatz'}

    def get_stale_homogenous_solution(self):
        a = self.parameters['a']
        b = self.parameters['b']
        x = b
        y = b / (a + b ** 2)
        return x, y

    def copy(self, patch: Dict[str, Any] = None):
        res = self.from_params_dict(self.parameters)
        if patch is not None:
            for key in patch:
                res[key] = patch[key]
        return res

    @staticmethod
    def from_params_dict(params: Dict[str, Any]):
        return SelkovStrogatz1DConfiguration(params['a'], params['b'], params['Dx'], params['Dy'])
