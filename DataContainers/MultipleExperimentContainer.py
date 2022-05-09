import os
import typing as tp
from collections import defaultdict

from frozendict import frozendict

from MyPackage.DataContainers.Experiment import Experiment
from tqdm import tqdm


class MultipleExperimentContainer:

    def __init__(self, folders: tp.Union[str, tp.List[str]], load_timelines: bool = True,
                 max_by_key: tp.Optional[int] = None,
                 key_filter: tp.Dict[str, tp.Callable[[tp.Any], bool]] = None):
        if isinstance(folders, str):
            folders = [folders]
        self.experiments: tp.Dict[frozendict, tp.Any] = {}
        self.experiments_without_folder: defaultdict[frozendict, int] = defaultdict(lambda: 0)
        self.__load_timelines = load_timelines
        self.__total_exps = 0

        for folder in folders:
            self.__get_all_experiments(folder, max_by_key, key_filter, dry_run=True)
            self.__get_all_experiments(folder, max_by_key, key_filter, dry_run=False)
            self.__total_exps = 0
        del self.experiments_without_folder

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

    def get_uniq_values(self, param_name: str) -> tp.List[tp.Any]:
        common_conf, uniq_confs = self.get_configs_uniq()
        if param_name in common_conf:
            return [common_conf[param_name]]
        res = set()
        for c in uniq_confs:
            if param_name in c:
                res.add(c[param_name])
        return sorted(res)

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

    def __get_key_without_folder(self, exp_folder) -> frozendict:
        assert Experiment.is_experiment(exp_folder)
        ex: Experiment = Experiment().fill_from(exp_folder, load_timelines=False)
        return frozendict({**ex.method_parameters, **ex.model_config})

    def __get_all_experiments(self, folder: str, max_by_key: tp.Optional[int] = None,
                              key_filter: tp.Dict[str, tp.Callable[[tp.Any], bool]] = None,
                              bar=None,
                              dry_run: bool = False):
        if bar is None:
            bar = tqdm(desc='Experiments indexed' if dry_run else 'Experiments loaded',
                       total=None if dry_run else self.__total_exps)
        if Experiment.is_experiment(folder):
            key_no_folder = self.__get_key_without_folder(folder)
            if max_by_key:
                if (
                        key_no_folder in self.experiments_without_folder and
                        self.experiments_without_folder[key_no_folder] >= max_by_key
                ):
                    return
                self.experiments_without_folder[key_no_folder] += 1
            if key_filter:
                for key in key_filter:
                    if not key_filter[key](key_no_folder[key]):
                        return
            if not dry_run:
                ex, key = self.__get_experiment_and_key(folder)
                self.experiments[key] = ex
            else:
                self.__total_exps += 1
            bar.update(1)
        elif os.path.isdir(folder):
            for f in os.listdir(folder):
                self.__get_all_experiments(os.path.join(folder, f), max_by_key, key_filter, bar, dry_run)
