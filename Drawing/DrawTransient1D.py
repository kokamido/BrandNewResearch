from typing import Dict, Tuple
from typing import List, Union, Any

import numpy as np
import pylab as plt
from matplotlib.lines import Line2D
from pylab import axes
from seaborn import heatmap, color_palette

from MyPackage.DataAnalyzers.PeaksAnalyzer import calc_few_Fourier_coeffs_for_transient
from MyPackage.DataAnalyzers.TransientAnalyzer import add_peaks_stats_Higgins1D, add_amp_stats_Higgins1D, \
    calc_deltas_for_timeline_1D
from MyPackage.DataContainers.Experiment import Experiment
from MyPackage.Drawing.DrawHelper import set_xticks, set_yticks, mark_bigger_values


def draw_few_Fouriers(e: Experiment, ks, right_border_t: float = None, left_border_t: float = None, ax: axes = None,
                      logscale: bool = False, mark_bigger_quantile: float = None,
                      quantile_palette: Dict[float, str] = None, top_n: int = 3,
                      c_k_palette: Dict[float, str] = None, var_to_draw: str = 'u') -> axes:
    assert mark_bigger_quantile is None or quantile_palette is not None
    ax = ax if ax else plt.gca()
    coeffs, ts = calc_few_Fourier_coeffs_for_transient(e, ks, var_to_draw, right_border_t, left_border_t)

    def zero_except_top_n(arr, n):
        arr[(-np.abs(arr)).argsort()[n:]] *= 0
        return arr

    res = np.apply_along_axis(lambda arr: zero_except_top_n(arr, top_n), 0, coeffs)
    for i, k in enumerate(ks):
        if c_k_palette is not None and k in c_k_palette:
            ax.plot(ts, res[i, :], lw=3, label=f'$C_{{{k}}}$', c=c_k_palette[k])
        else:
            ax.plot(ts, res[i, :], lw=3, label=f'$C_{{{k}}}$')
        if mark_bigger_quantile is not None:
            mark_bigger_values(res[i, :], mark_bigger_quantile, ax, quantile_palette[k], e.method_parameters['dx'])
    if logscale:
        ax.set_yscale('log')
    plt.legend()
    return ax


def draw_amps_stat(results: List[Experiment], L_min: float, L_max: float, xticks: Union[List[float], np.ndarray] = None,
                   x_name: str = None, yticks: Union[List[float], np.ndarray] = None, y_name: str = None,
                   logscale: bool = False) -> None:
    for r in results:
        assert r.model_config['model'] == 'Higgins'
        r.method_parameters['x_max'] = r.method_parameters['dx'] * \
                                       len(r.init_values['u'])
        add_peaks_stats_Higgins1D(r)
        add_amp_stats_Higgins1D(r)
    current_data = [r for r in results if L_min <=
                    r.method_parameters['x_max'] <= L_max]
    ax = plt.gca()
    color_indices = {key: i for i, key in
                     enumerate(set(map(lambda x: x.metadata['end_picks']['u']['peaks'], current_data)))}
    palette = color_palette('tab10').as_hex()
    for r in current_data:
        color = palette[color_indices[r.metadata['end_picks']['u']['peaks']]]
        ax.scatter(
            r.method_parameters['x_max'], r.metadata['end_amps']['v'], c=color, marker='+', s=90)
        ax.scatter(
            r.method_parameters['x_max'], r.metadata['end_amps']['u'], c=color, marker='x', s=80)
    legend_elements = []
    for peaks in sorted(color_indices):
        legend_elements.append(
            Line2D([0], [0], marker='8', label=str(peaks), markerfacecolor=palette[color_indices[peaks]],
                   markersize=10))
    ax.legend(handles=legend_elements, loc='center right')
    ax.set_title(
        f'Амплитуды итоговых паттернов для u(x) и v(+) для значений L от {L_min} до {L_max}.')

    if logscale:
        ax.set_yscale('log')

    if xticks is None:
        xticks = np.linspace(L_min, L_max, 5)
    set_xticks(ax, xticks, '$L$' if x_name is None else x_name)

    if yticks is None:
        yticks = [round(i, 3) for i in ax.get_yticks()[0:-2]]
    set_yticks(ax, yticks, 'Amp' if y_name is None else y_name)
    plt.show()


def draw_timeline_deltas(e: Experiment, left_border=None, right_border=None) -> None:
    assert e.method_parameters['dt']
    assert e.timelines
    deltas = calc_deltas_for_timeline_1D(e)
    if left_border is None:
        left_border = 0
    if right_border is None:
        right_border = len(deltas['u'])
    ax = plt.gca()
    for k in deltas:
        ax.plot(np.linspace(0, right_border, right_border - left_border) *
                e.method_parameters['dt'], deltas[k][left_border:right_border], label=k)
    set_xticks(ax, [round(i, 2) for i in ax.get_xticks()[1:-1]], '$t$')
    ax.set_yscale('log')
    plt.title('Норма разности между смежными состояниями системы')
    plt.legend()
    plt.show()


def draw_transient(e: Experiment, left_border: float = None, right_border: float = None,
                   xticks: Union[List[float], np.ndarray] = None,
                   yticks: Union[List[float], np.ndarray] = None,
                   draw_u_only: bool = True, cmap: str = 'nipy_spectral_r', ax: axes = None,
                   cbar_kws: Dict[str, Any] = None) -> axes:
    assert e.timelines
    assert e.method_parameters['dt']

    dt = e.method_parameters['dt'] * e.method_parameters['timeline_save_step_delta']
    dx = e.method_parameters['dx']
    time_step_max = e.timelines['u'].shape[0] - 1

    cbar_kws = cbar_kws if cbar_kws is not None else {}
    if left_border is None or left_border < 0:
        left_border = 0
    if right_border is None or right_border > time_step_max * dt:
        right_border = time_step_max * dt

    if left_border >= right_border:
        left_border = right_border - 100

    time_range = (right_border - left_border)
    left_border = int(left_border / dt)
    right_border = int(right_border / dt)

    if xticks is None:
        quant = max(int(time_range / dt // 6), 1)
        xticklabels = [i for i in np.linspace(
            left_border, right_border, (right_border - left_border) // quant)]
    else:
        xticklabels = xticks

    if yticks is None:
        ytickslabels = [round(i, 3) for i in np.linspace(0, int(e.timelines['u'].shape[1]), 5)]
    else:
        ytickslabels = yticks

    ax = __draw_transient__(
        e.timelines['u'].T[::-1, left_border:right_border], dt, dx, '$t$', '$x$', cmap, xticklabels, ytickslabels, ax,
        cbar_kws)
    if not draw_u_only:
        ax = __draw_transient__(
            e.timelines['v'].T[::-1, left_border:right_border], dt, dx, '$t$', '$V$', cmap, xticklabels, ytickslabels,
            ax, cbar_kws)
    return ax


def draw_transient_parts_with_max_coeffs(e: Experiment, coeffs_to_search: List[float], var_to_draw: str,
                                         skip_from_start: int = 100, window_size_t: int = 100,
                                         axs: List[axes] = None) -> Tuple[plt.figure, List[axes]]:
    assert e.timelines is not None
    assert var_to_draw in e.timelines
    assert axs is None or len(axs) == len(coeffs_to_search)
    fig = None
    if axs is None:
        fig, axs = plt.subplots(len(coeffs_to_search), 1)
    coeffs, ts = calc_few_Fourier_coeffs_for_transient(e, coeffs_to_search, var_to_draw)
    coeffs = coeffs[:, skip_from_start:]
    ts = ts[skip_from_start:, ]
    max_indices = np.apply_along_axis(np.argmax, 1, np.abs(coeffs))
    dt = e.method_parameters['dt'] * e.method_parameters['timeline_save_step_delta']
    window_size = int(window_size_t / dt)
    res = []
    for coeff_val, max_ind, ax in zip(coeffs_to_search, max_indices, axs):
        timeline = e.timelines[var_to_draw]
        left_border = max(0, max_ind - window_size)
        right_border = min(max_ind + window_size, timeline.shape[0] - 1)
        to_draw_trans = timeline[left_border: right_border, :]
        xlabels = np.linspace(0, right_border - left_border + 1, 5)  + ts[max_ind - window_size] / dt
        ylabels = np.round(np.linspace(e.method_parameters['x_left'], e.method_parameters['x_right'], 5) / e.method_parameters['dx'], 1)
        res.append(__draw_transient__(to_draw_trans.T, dt, e.method_parameters['dx'], '$t$', '$x$',
                           'nipy_spectral_r', xlabels, ylabels, ax, {}))
        ax.set_title(f'$C_{{{coeff_val}}}$')
    return fig, res


def __draw_transient__(data: np.ndarray, dt: float, dx: float, x_name: str, y_name: str, cmap: str,
                       xticklabels: Union[List[float], np.ndarray], yticklabels: Union[List[float], np.ndarray],
                       ax: axes, cbar_kws: Dict[str, Any]) -> axes:
    ax = heatmap(data, cmap=cmap, ax=ax, cbar_kws=cbar_kws)
    if data.shape[1] - xticklabels[-1] > 50:
        xticklabels += [data.shape[1]]
    x_min = min(xticklabels)
    y_min = min(yticklabels)
    set_xticks(ax, [x - x_min for x in xticklabels], x_name,
               ticklabels=[str(round(x * dt)) for x in xticklabels])
    ticklabels = [str(round(i * dx, 3)) for i in yticklabels[::-1]]
    ticklabels[0] = y_name
    set_yticks(ax, [y - y_min for y in yticklabels], ticklabels=ticklabels)
    return ax
