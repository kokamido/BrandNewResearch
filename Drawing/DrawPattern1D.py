import numpy as np
from matplotlib import pyplot as plt

from MyPackage.DataContainers.Experiment import Experiment


def draw_timeline_cross_section(ex: Experiment, value_to_draw: str, t: float, color: str = 'b', title: str = None,
                                ax: plt.Axes = None) -> plt.axes:
    dt = ex.method_parameters['dt'] * ex.method_parameters['timeline_save_step_delta']
    dx = ex.method_parameters['dx']
    return draw_pattern(ex.timelines[value_to_draw][int(t / dt)], dx, color, title, value_to_draw, ax)


def draw_pattern(data: np.array, dx: float, color: str = 'b', title: str = None, label: str = None,
                 ax: plt.axes = None) -> plt.axes:
    if ax is None:
        ax = plt.gca()
    if title is not None:
        ax.set_title(title)
    ax.grid(True)
    ax.plot(np.arange(data.size) * dx, data, c=color, label=label, lw=5)
    ax.set_xlabel('$X$')
    return ax
