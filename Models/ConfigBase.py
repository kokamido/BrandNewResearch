import json
from typing import Dict, Any

from MyPackage.PythonHelpers.IOHelpers import to_json_np_aware


class ConfigBase:

    parameters: Dict[str, Any]

    def to_json(self) -> str:
        return to_json_np_aware(self.parameters)

    def __str__(self):
        return f'{self.parameters}'

    def __contains__(self, key):
        return key in self.parameters

    def __getitem__(self, key):
        return self.parameters[key]

    def __setitem__(self, key, value):
        self.parameters[key] = value

    def __delitem__(self, key):
        del self.parameters[key]
