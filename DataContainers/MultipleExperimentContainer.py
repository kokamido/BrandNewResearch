import os
import typing as tp

from frozendict import frozendict

from MyPackage.DataContainers.Experiment import Experiment


class MultipleExperimentContainer:

    def __init__(self, folders: tp.Union[str, tp.List[str]], load_timelines: bool = True):
        if isinstance(folders, str):
            folders = [folders]
        self.experiments: tp.Dict[frozendict, tp.Any] = {}
        self.__load_timelines = load_timelines

        for folder in folders:
            self.__get_all_experiments(folder)

    def get_configs(self):
        res = set()
        for key in self.experiments:
            buf = dict(**key)
            del buf['folder']
            res.add(frozendict(buf))
        return res

    def get_configs_uniq(self, filter_condition: tp.Dict[str, tp.Any] = None,
                         params_to_exclude: tp.Set[str] = None, params_to_include: tp.Set[str] = None) \
            -> tp.Tuple[frozendict, tp.Set[frozendict]]:
        assert params_to_include is None or params_to_exclude is None

        if filter_condition is None:
            filter_condition = {}
        if params_to_exclude is None:
            params_to_exclude = set()
        if params_to_include is None:
            params_to_include = set()

        not_uniq_params = set()
        uniq_params = {}

        for key in self.experiments:
            if not MultipleExperimentContainer.__match_filter(key, filter_condition):
                continue
            for segment in key:
                if segment == 'folder':
                    continue
                if params_to_exclude and segment in params_to_exclude:
                    continue
                if params_to_include and segment not in params_to_include:
                    continue
                if segment not in uniq_params and segment not in not_uniq_params:
                    uniq_params[segment] = key[segment]
                elif segment in uniq_params:
                    if uniq_params[segment] != key[segment]:
                        not_uniq_params.add(segment)
                        del uniq_params[segment]
        res = set()
        for key in self.experiments:
            if MultipleExperimentContainer.__match_filter(key, filter_condition):
                res.add(frozendict({x: key[x] for x in not_uniq_params}))
        return frozendict(uniq_params), res

    def get_experiments_by_filter(self, filter_condition: tp.Dict[str, tp.Any]) -> tp.List[Experiment]:
        res = []
        for key in self.experiments:
            if MultipleExperimentContainer.__match_filter(key, filter_condition):
                res.append(self.experiments[key])
        return res

    @staticmethod
    def __match_filter(key: frozendict, filter: tp.Dict[str, tp.Any]) -> bool:
        if not filter:
            return True
        return all([key[x] == filter[x] for x in filter])

    def __get_experiment_and_key(self, folder: str):
        assert Experiment.is_experiment(folder)
        ex: Experiment = Experiment().fill_from(folder, self.__load_timelines)
        params = {**ex.method_parameters, **ex.model_config, 'folder': folder}
        return ex, frozendict(params)

    def __get_all_experiments(self, folder: str):
        if Experiment.is_experiment(folder):
            ex, key = self.__get_experiment_and_key(folder)
            self.experiments[key] = ex
        elif os.path.isdir(folder):
            for f in os.listdir(folder):
                self.__get_all_experiments(os.path.join(folder, f))
