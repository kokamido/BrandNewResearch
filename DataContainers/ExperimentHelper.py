from typing import List, Tuple, Optional

from MyPackage.DataContainers.Experiment import Experiment
from MyPackage.PythonHelpers.IOHelpers import do_with_all_subfolders


def read_experiments_from_dir(dir_name, load_timelines: bool = False) -> List[Experiment]:
    return do_with_all_subfolders(dir_name, lambda f: Experiment().fill_from(f, load_timelines=load_timelines))


def convert_time_to_indices(e: Experiment, left_border: Optional[float] = None, right_border: Optional[float] = None) \
        -> Tuple[int, int]:
    dt = e.method_parameters['dt'] * e.method_parameters['timeline_save_step_delta']
    time_step_max = e.timelines['u'].shape[0] - 1

    if left_border is None or left_border < 0:
        left_border = 0
    if right_border is None or right_border > time_step_max * dt:
        right_border = time_step_max * dt
    if left_border > right_border:
        left_border, right_border = right_border, left_border

    assert e.timelines is not None

    left_border = int(left_border / dt)
    right_border = int(right_border / dt)
    return left_border, right_border
