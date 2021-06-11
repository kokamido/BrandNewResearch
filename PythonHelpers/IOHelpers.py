import datetime
import json
from os import path, walk
from typing import Dict, Any, Callable, List, Union

import numpy as np
from nptyping import NDArray
from numpy import ndarray


def write_to_file(path_to_file: str, data: str) -> None:
    assert data
    assert path_to_file
    with open(path_to_file, 'w', encoding='utf-8') as out:
        out.write(data)


def myconverter(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, datetime.datetime):
        return obj.__str__()


def to_json_np_aware(data: Dict[str, Union[NDArray, ndarray]]) -> str:
    if not data:
        return ''
    return json.dumps(data, default=myconverter, indent=2)


def load_json(path_to_file: str) -> Dict[str, Any]:
    assert path.exists(path_to_file)
    with open(path_to_file, 'r', encoding='utf-8') as inp:
        return json.load(inp)


def do_with_all_subfolders(parent_folder: str, function: Callable[[str], Any]) -> List[Any]:
    assert path.exists(parent_folder)
    res = []
    for directory in list(walk(parent_folder))[0][1]:
        res.append(function(path.join(parent_folder, directory)))
    return res
