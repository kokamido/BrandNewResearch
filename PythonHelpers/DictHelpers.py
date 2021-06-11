from typing import Dict, Any, Union

from nptyping import NDArray
from numpy import ndarray


def assert_if_not_present(key: str, dict_to_check: Dict[str, Any]) -> None:
    try:
        assert isinstance(key, str), f'Key must be str, not {type(key)}'
        assert key, f'Key can\'t be empty'
        assert key in dict_to_check, f"key '{key}' not presented in dict"
    except Exception as e:
        raise e


def check_shapes_equality(data: Dict[str, Union[ndarray, NDArray]]) -> None:
    if not data:
        return
    shape = None
    for key in data:
        if shape is None:
            shape = data[key].shape
        else:
            assert shape == data[key].shape
