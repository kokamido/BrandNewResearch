from typing import Dict, Any

from MyPackage.Models.ConfigBase import ConfigBase


class Selkov1DConfiguration(ConfigBase):

    def __init__(self, theta: float, omega: float, Du: float, Dv: float):
        assert Du > 0
        assert Dv > 0
        self.parameters = {'omega': omega, 'theta': theta,
                               'Du': Du, 'Dv': Dv, 'model': 'Selkov'}

    def get_stale_homogenous_solution(self):
        theta = self.parameters['theta']
        omega = self.parameters['omega']
        v = theta / omega
        u = theta / v ** 2
        return u, v

    def copy(self, patch: Dict[str, Any] = None):
        res = from_params_dict(self.parameters)
        if patch is not None:
            for key in patch:
                res[key] = patch[key]
        return res


def from_params_dict(params: Dict[str, Any]):
    return Selkov1DConfiguration(params['theta'], params['omega'], params['Du'], params['Dv'])
