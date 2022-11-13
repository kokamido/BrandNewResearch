from typing import Callable, TypeVar, List, Iterable
from tqdm import tqdm

from MyPackage.DataContainers.Experiment import Experiment

CONFIG_TYPE = TypeVar('CONFIG_TYPE')
INIT_DATA_TYPE = TypeVar('INIT_DATA_TYPE')


def traverse_recursive(traverse: Callable[[CONFIG_TYPE, INIT_DATA_TYPE], Experiment], configs: List[CONFIG_TYPE],
                       make_new_init_data: Callable[[Experiment], INIT_DATA_TYPE],
                       init_data: INIT_DATA_TYPE, verbose: bool = False) -> Iterable[Experiment]:
    current_init_data = None
    for config in tqdm(configs) if verbose else configs:
        if current_init_data is None:
            current_init_data = init_data
        res = traverse(config, current_init_data)
        yield res
        current_init_data = make_new_init_data(res)

