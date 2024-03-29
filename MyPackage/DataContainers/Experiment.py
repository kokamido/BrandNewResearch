import os
import re
from typing import TypeVar, Dict, Any

import numpy as np

from numpy import ndarray, array

from MyPackage.Models.ConfigBase import ConfigBase
from MyPackage.PythonHelpers.DictHelpers import check_shapes_equality
from MyPackage.PythonHelpers.IOHelpers import write_to_file, to_json_np_aware, load_json


class Experiment:
    Experiment = TypeVar('Experiment')
    NecessaryFiles = ['end_values', 'init_values', 'method_parameters', 'model_config']

    def fill(self, model_config: ConfigBase, method_parameters: ConfigBase,
             init_values: Dict[str, np.array], end_values: Dict[str, np.array],
             timelines: Dict[str, ndarray] = None) -> Experiment:
        assert model_config
        assert method_parameters
        assert len(init_values) == len(end_values)
        check_shapes_equality(init_values)
        check_shapes_equality(end_values)
        if timelines:
            assert len(init_values) == len(timelines)
            check_shapes_equality(timelines)

        self.model_config = model_config.parameters
        self.method_parameters = method_parameters.parameters
        self.init_values = init_values
        self.end_values = end_values
        self.timelines = timelines
        self.metadata: Dict[str, Any] = {}
        self.path_to_file = None
        return self

    @staticmethod
    def is_experiment(path: str) -> bool:
        if not os.path.isdir(path):
            return False
        inner_files = os.listdir(path)
        return all([os.path.isfile(os.path.join(path, x)) and x in inner_files for x in Experiment.NecessaryFiles])

    def fill_from(self, path_to_data: str, load_timelines: bool = False, verbose: bool = False) -> Experiment:
        assert Experiment.is_experiment(path_to_data)
        self.path_to_file = path_to_data
        self.model_config = load_json(os.path.join(path_to_data, 'model_config'))
        self.method_parameters = load_json(os.path.join(path_to_data, 'method_parameters'))

        self.init_values = load_json(os.path.join(path_to_data, 'init_values'))
        for key in self.init_values:
            self.init_values[key] = array(self.init_values[key])

        self.end_values = load_json(os.path.join(path_to_data, 'end_values'))
        for key in self.end_values:
            self.end_values[key] = array(self.end_values[key])

        self.timelines = None
        if load_timelines:
            self.timelines = {}
            for file in [f for f in os.listdir(path_to_data) if f.startswith('timeline_')]:
                var_name = file.replace('timeline_', '').split('.')[0]
                self.timelines[var_name] = np.load(f'{path_to_data}/{file}')

        path_to_metadata = os.path.join(path_to_data, 'metadata')
        try:
            self.metadata = load_json(path_to_metadata)
        except Exception as e:
            if verbose:
                print(f'Can`t load metadata {e}')
        return self

    def save(self, path_to_save: str) -> None:
        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)
        write_to_file(os.path.join(path_to_save, 'model_config'), to_json_np_aware(self.model_config))
        write_to_file(os.path.join(path_to_save, 'method_parameters'), to_json_np_aware(self.method_parameters))
        write_to_file(os.path.join(path_to_save, 'init_values'), to_json_np_aware(self.init_values))
        write_to_file(os.path.join(path_to_save, 'end_values'), to_json_np_aware(self.end_values))
        if self.timelines:
            for key in self.timelines:
                np.save(os.path.join(path_to_save, f'timeline_{key}'), self.timelines[key])
        if self.metadata:
            write_to_file(os.path.join(path_to_save, 'metadata'), to_json_np_aware(self.metadata))

    def get_folder_name(self):
        return re.sub(r'[<>:"/|?*\\\[\]{}\']+', '', f'{self.model_config}')

    def __str__(self):
        return f'model_config:\n{self.model_config}\nmethod_parameters:\n{self.method_parameters}\n' \
               f'init_values:\n{self.init_values}\nend_values:\n{self.end_values}\n'
