from typing import Dict, Any

from PythonHeplers.IOHelpers import load_python_array


class Higgins1DConfiguration:

    def __init__(self, p: float, q: float, Du: float, Dv: float):
        assert p > 0
        assert q > 0
        assert Du > 0
        assert Dv > 0
        self.parameters = {'p': p, 'q': q,
                           'Du': Du, 'Dv': Dv, 'model': 'Higgins'}

    @staticmethod
    def from_params_dict(params: Dict[str, Any]):
        return Higgins1DConfiguration(params['p'], params['q'], params['Du'], params['Dv'])

    def __str__(self):
        return f'{self.parameters}'
