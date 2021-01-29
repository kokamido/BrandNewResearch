from typing import Dict, Any
from PythonHeplers.IOHelpers import load_python_array


class Selkov1DConfiguration:

    def __init__(self, n: float, w: float, Du: float, Dv: float):
        assert Du > 0
        assert Dv > 0
        self.parameters = {'w': w, 'n': n,
                           'Du': Du, 'Dv': Dv, 'model': 'Selkov'}

    @staticmethod
    def from_params_dict(params: Dict[str, Any]):
        return Selkov1DConfiguration(params['n'], params['w'], params['Du'], params['Dv'])

    def __str__(self):
        return f'{self.parameters}'
