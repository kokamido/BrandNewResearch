from typing import Dict, Any


class Higgins1DConfiguration:

    def __init__(self, p: float, q: float, Du: float, Dv: float):
        assert p > 0
        assert q > 0
        assert Du > 0
        assert Dv > 0
        self.__parameters__ = {'p': p, 'q': q,
                               'Du': Du, 'Dv': Dv, 'model': 'Higgins'}

    def __str__(self):
        return f'{self.__parameters__}'

    def __getitem__(self, key):
        return self.__parameters__[key]

    def __setitem__(self, key, value):
        self.__parameters__[key] = value


def from_params_dict(params: Dict[str, Any]) -> Higgins1DConfiguration:
    return Higgins1DConfiguration(params['p'], params['q'], params['Du'], params['Dv'])
