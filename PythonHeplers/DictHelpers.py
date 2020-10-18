from typing import Dict, Any
from Logging.logger import log


def assert_if_not_present(key: str, dict: Dict[str, Any]) -> None:
    try:
        assert isinstance(key, str), f'Key must be str, not {type(key)}'
        assert key, f'Key can\'t be empty'
        assert key in dict, f"key '{key}' not presented in dict"
    except Exception as e:
        log.exception('assert_if_not_present fail')
        raise e
