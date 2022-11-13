import numpy as np
from typing import List, Tuple


def get_ranges_value_bigger_than_quantile(data: np.array, quantile: float) -> List[Tuple[float, float]]:
    q = np.quantile(data, quantile)
    left = None
    res = []
    for index, value in enumerate(data):
        if value < q:
            if left is not None:
                res.append((left, index))
                left = None
        else:
            if left is None:
                left = index
    if left is not None:
        res.append((left,data.size - 1))
    return res

