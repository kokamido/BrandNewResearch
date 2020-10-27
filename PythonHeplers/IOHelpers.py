import json
from os import path, walk
from typing import Dict, Any, Callable, List, Union

from nptyping import NDArray
from numpy import ndarray


def append_or_create(path_to_file: str, data: str) -> None:
    assert data
    assert path_to_file
    with open(path_to_file, 'w', encoding='utf-8') as out:
        out.write(data)


def to_json_np_aware(data: Dict[str, Union[NDArray,ndarray]]) -> str:
    if not data:
        return ''
    res = {}
    for key in data:
        res[key] = data[key]
        if isinstance(res[key], ndarray):
            res[key] = res[key].tolist()
    return json.dumps(res)


def load_json(path_to_file: str) -> Dict[str, Any]:
    assert path.exists(path_to_file)
    with open(path_to_file, 'r', encoding='utf-8') as inp:
        return json.load(inp)


def load_python_array(path_to_file: str) -> Dict[str, Any]:
    assert path.exists(path_to_file)
    with open(path_to_file, 'r', encoding='utf-8') as inp:
        return json.loads(inp.readline().replace('\'', '"').lower())


def do_with_all_subfolders(parent_folder: str, function: Callable[[str], Any]) -> List[Any]:
    assert path.exists(parent_folder)
    res = []
    for directory in list(walk(parent_folder))[0][1]:
        res.append(function(path.join(parent_folder, directory)))
    return res
