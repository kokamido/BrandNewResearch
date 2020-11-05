from numpy import ndarray, array, loadtxt

from os import path, makedirs
from typing import TypeVar, Dict, Any

from nptyping import NDArray

from PythonHeplers.DictHelpers import check_shapes_equality
from PythonHeplers.IOHelpers import append_or_create, to_json_np_aware, load_json, load_python_array


class Experiment:
    Experiment = TypeVar('Experiment')

    def fill(self, model_config: Dict[str, float], method_parameters: Dict[str, Any],
             init_values: Dict[str, NDArray], end_values: Dict[str, NDArray],
             timelines: Dict[str, ndarray] = None) -> Experiment:
        assert model_config
        assert method_parameters
        assert len(init_values) == len(end_values)
        check_shapes_equality(init_values)
        check_shapes_equality(end_values)
        if timelines:
            assert len(init_values) == len(timelines)
            check_shapes_equality(timelines)

        self.model_config = model_config
        self.method_parameters = method_parameters
        self.init_values = init_values
        self.end_values = end_values
        self.timelines = timelines
        self.metadata = {}
        return self

    def fill_from_file(self, path_to_file: str, load_timelines: bool = False) -> Experiment:
        assert path.exists(path_to_file)
        self.model_config = load_json(path.join(path_to_file, 'model_config'))
        self.method_parameters = load_json(path.join(path_to_file, 'method_parameters'))

        self.init_values = load_json(path.join(path_to_file, 'init_values'))
        for key in self.init_values:
            self.init_values[key] = array(self.init_values[key])

        self.end_values = load_json(path.join(path_to_file, 'end_values'))
        for key in self.end_values:
            self.end_values[key] = array(self.end_values[key])

        path_to_timelines = path.join(path_to_file, 'timelines')
        if load_timelines and path.exists(path_to_timelines):
            self.timelines = load_json(path_to_timelines)
            for key in self.timelines:
                self.timelines[key] = array(self.timelines[key])
        self.metadata = {}
        return self

    def fill_from_file_Higgins_legacy_format(self, path_to_file: str, load_timelines: bool = False) -> Experiment:
        assert path.exists(path_to_file)
        self.model_config = load_python_array(path.join(path_to_file, 'config'))
        self.method_parameters = load_python_array(path.join(path_to_file, 'parameters'))

        self.init_values = {}

        self.end_values = {'u': loadtxt(path.join(path_to_file, 'res_u')),
                           'v': loadtxt(path.join(path_to_file, 'res_v'))}

        path_to_timeline_u = path.join(path_to_file, 'res_u_timeline')
        path_to_timeline_v = path.join(path_to_file, 'res_v_timeline')

        if load_timelines and path.exists(path_to_timeline_u) and path.exists(path_to_timeline_v):
            self.timelines = {'u': loadtxt(path_to_timeline_u),
                              'v': loadtxt(path_to_timeline_v)}
        self.metadata = {}
        return self

    def save(self, path_to_save: str) -> None:
        if not path.exists(path_to_save):
            makedirs(path_to_save)
        append_or_create(path.join(path_to_save, 'model_config'), to_json_np_aware(self.model_config))
        append_or_create(path.join(path_to_save, 'method_parameters'), to_json_np_aware(self.method_parameters))
        append_or_create(path.join(path_to_save, 'init_values'), to_json_np_aware(self.init_values))
        append_or_create(path.join(path_to_save, 'end_values'), to_json_np_aware(self.end_values))
        append_or_create(path.join(path_to_save, 'timelines'), to_json_np_aware(self.timelines))

    def __str__(self):
        return f'model_config:\n{self.model_config}\nmethod_parameters:\n{self.method_parameters}\n' \
               f'init_values:\n{self.init_values}\nend_values:\n{self.end_values}\n' \
               f'timelines: {"Presented" if self.timelines else "Not presented"}\n'
