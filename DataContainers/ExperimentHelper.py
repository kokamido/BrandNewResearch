from typing import List

from MyPackage.DataContainers.Experiment import Experiment
from MyPackage.PythonHelpers.IOHelpers import do_with_all_subfolders


def read_experiments_from_dir(dir_name, load_timelines: bool = False) -> List[Experiment]:
    return do_with_all_subfolders(dir_name, lambda f: Experiment().fill_from_file(f, load_timelines=load_timelines))
