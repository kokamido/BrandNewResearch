from typing import List

from DataContainers.Experiment import Experiment
from PythonHeplers.IOHelpers import do_with_all_subfolders


def read_experiments_from_dir(dir_name, load_timelines: bool = False) -> List[Experiment]:
    def sas(a):
        return Experiment().fill_from_file(a, load_timelines=load_timelines)
    return do_with_all_subfolders(dir_name, sas)