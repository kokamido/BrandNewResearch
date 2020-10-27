from matplotlib import axes, rcParams
from numpy import arange


def set_defaults_1D():
    rcParams['figure.figsize'] = (16,8)
    rcParams['font.size'] = 22
    rcParams['axes.grid'] = True
    rcParams['grid.color'] = 'black'


def set_xticks(ax: axes, left: float, right: float, step: float, var_name:str = None) -> axes:
    xticks = arange(left, right+step, step)
    ax.set_xticks(xticks)
    if var_name:
        xticks[-1] = var_name
    ax.set_xtickslabels(xticks)
    return ax


def set_yticks(ax: axes, bot: float, top: float, step: float, var_name:str = None) -> axes:
    yticks = arange(bot, top+step, step)
    ax.set_yticks(yticks)
    if var_name:
        yticks[-1] = var_name
    ax.set_ytickslabels(yticks)
    return ax