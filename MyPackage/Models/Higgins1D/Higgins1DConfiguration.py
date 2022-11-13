from typing import Dict, Any

from MyPackage.Models.ConfigBase import ConfigBase


class Higgins1DConfiguration(ConfigBase):

    def __init__(self, p: float, q: float, Du: float, Dv: float):
        assert p > 0
        assert q > 0
        assert Du > 0
        assert Dv > 0
        self.parameters = {'p': p, 'q': q,
                               'Du': Du, 'Dv': Dv, 'model': 'Higgins'}


def from_params_dict(params: Dict[str, Any]) -> Higgins1DConfiguration:
    return Higgins1DConfiguration(params['p'], params['q'], params['Du'], params['Dv'])
