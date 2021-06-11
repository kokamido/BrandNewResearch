from typing import List, Union

from matplotlib import axes, rcParams
from numpy import arange, ndarray

from MyPackage.MathHelpers.SequenceSeriesHelper import get_ranges_value_bigger_than_quantile


def set_defaults_1D():
    rcParams['figure.figsize'] = (16, 8)
    rcParams['font.size'] = 22
    rcParams['axes.grid'] = True
    rcParams['grid.color'] = 'black'


def set_defaults_2D():
    rcParams['figure.figsize'] = (20, 10)
    rcParams['font.size'] = 24
    rcParams['axes.grid'] = True
    rcParams['grid.color'] = 'black'


def set_xticks(ax: axes, ticks: Union[List[float], ndarray], var_name: str = None, ticklabels: List[str] = None,
               rotation: int = 0):
    ax.set_xticks(ticks)
    ticklabels = ticklabels if ticklabels else [str(i) for i in ticks]
    if var_name:
        ticklabels[-1] = var_name
    ax.set_xticklabels(ticklabels, rotation=rotation)
    return ax


def set_xticks_with_borders(ax: axes, left: float, right: float, step: float, var_name: str = None) -> axes:
    xticks = arange(left, right + step, step)
    return set_xticks(ax, xticks, var_name)


def set_yticks(ax: axes, ticks: Union[List[float], ndarray], var_name: str = None, ticklabels: List[str] = None,
               rotation: int = 0) -> axes:
    ax.set_yticks(ticks)
    ticklabels = ticklabels if ticklabels else [str(i) for i in ticks]
    if var_name:
        ticklabels[-1] = var_name
    ax.set_yticklabels(ticklabels, rotation=rotation)
    return ax


def set_yticks_with_borders(ax: axes, left: float, right: float, step: float, var_name: str = None) -> axes:
    ticks = arange(left, right + step, step)
    return set_yticks(ax, ticks, var_name)


def mark_bigger_values(data: ndarray, quantile: float, ax: axes, color: str = 'r', alpha: float = .5, dx: float = 1):
    ranges = get_ranges_value_bigger_than_quantile(data, quantile)
    for left, right in ranges:
        ax.fill_betweenx(left * dx, right * dx, color=color, alpha=alpha)
    return ax
