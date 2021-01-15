from typing import List, Union, Tuple

import pylab as plt
from pylab import axes
from pylab import figure
from matplotlib.lines import Line2D
from numpy import linspace, ndarray
from seaborn import heatmap, color_palette

from DataAnalyzers.PicksAnalyzer import add_peaks_stats_Higgins1D, add_amp_stats_Higgins1D, calc_picks, calc_Fourier_coeff_for_transient
from DataAnalyzers.TransientAnalyzer import calc_deltas_for_timeline_1D
from DataContainers.Experiment import Experiment
from Drawing.DrawHelper import set_xticks, set_yticks


def draw_few_Fouriers(e: Experiment, ks, right_border_t: float = None):
    dt = e.method_parameters['dt'] * \
        e.method_parameters['timeline_save_step_delta']
    ts_count = e.timelines['u'].shape[0]
    if right_border_t:
        right_border_t = int(round(right_border_t/dt))
    else:
        right_border_t = ts_count
    xs = linspace(0,right_border_t, right_border_t+1)*dt
    for k in ks:
        plt.plot(xs, calc_Fourier_coeff_for_transient(e.timelines['u'],e.method_parameters['dx'],k,right_border_t+1), lw=3, label=str(k))
    plt.legend()
    plt.show()


def draw_amps_stat(results: List[Experiment], L_min: float, L_max: float, xticks: Union[List[float], ndarray] = None,
                   x_name: str = None, yticks: Union[List[float], ndarray] = None, y_name: str = None, logscale: bool = False) -> None:
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
                     enumerate(set(map(lambda x: x.metadata['end_picks']['u']['picks'], current_data)))}
    palette = color_palette('tab10').as_hex()
    for r in current_data:
        color = palette[color_indices[r.metadata['end_picks']['u']['picks']]]
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
        xticks = linspace(L_min, L_max, 5)
    set_xticks(ax, xticks, '$L$' if x_name is None else x_name)

    if yticks is None:
        yticks = [round(i, 3) for i in ax.get_yticks()[0:-2]]
    set_yticks(ax, yticks, 'Amp' if y_name is None else y_name)
    plt.show()


def draw_transient(e: Experiment, left_border: float = None, right_border: float = None,
                   xticks: Union[List[float], ndarray] = None,
                   yticks: Union[List[float], ndarray] = None,
                   draw_u_only: bool = True, cmap: str = 'nipy_spectral_r', ax: axes = None) -> axes:
    assert e.timelines
    assert e.timelines['u'].shape == e.timelines['v'].shape
    assert e.method_parameters['dt']

    dt = e.method_parameters['dt'] * \
        e.method_parameters['timeline_save_step_delta']
    time_step_max = e.timelines['u'].shape[0] - 1

    if left_border is None or left_border < 0:
        left_border = 0
    if right_border is None or right_border > time_step_max*dt:
        right_border = time_step_max*dt

    if left_border >= right_border:
        left_border = right_border - 100

    time_range = (right_border - left_border)
    left_border = int(left_border/dt)
    right_border = int(right_border/dt)

    if xticks is None:
        quant = max(int(time_range / dt // 6), 1)
        xticklabels = [i for i in linspace(
            left_border, right_border, (right_border - left_border) // quant)]

    if yticks is None:
        ytickslabels = [round(i, 3)
                        for i in linspace(0, e.timelines['u'].shape[1], 5)]

    ax = draw_transient_internal(
        e.timelines['u'].T[::-1, left_border:right_border], dt, '$t$', '$U$', cmap, xticklabels, ytickslabels, ax)
    if not draw_u_only:
        ax = draw_transient_internal(
            e.timelines['v'].T[::-1, left_border:right_border], dt, '$t$', '$V$', cmap, xticklabels, ytickslabels, ax)
    return ax


def draw_transient_internal(data: ndarray, dt: float, x_name: str, y_name: str, cmap: str,
                            xticklabels: Union[List[float], ndarray], yticklabels: Union[List[float], ndarray], ax: axes) -> axes:
    ax = heatmap(data, cmap=cmap, ax=ax)
    if data.shape[1] - xticklabels[-1] > 50:
        xticklabels += [data.shape[1]]
    x_min = min(xticklabels)
    y_min = min(yticklabels)
    set_xticks(ax, [x - x_min for x in xticklabels], x_name,
               ticklabels=[round(x * dt) for x in xticklabels])
    ticklabels = [str(i) for i in yticklabels[::-1]]
    ticklabels[0] = y_name
    set_yticks(ax, [y - y_min for y in yticklabels], ticklabels=ticklabels)
    return ax


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
        ax.plot(linspace(0, right_border, right_border - left_border) *
                e.method_parameters['dt'], deltas[k][left_border:right_border], label=k)
    set_xticks(ax, [round(i, 2) for i in ax.get_xticks()[1:-1]], '$t$')
    ax.set_yscale('log')
    plt.title('Норма разности между смежными состояниями системы')
    plt.legend()
    plt.show()


def draw_arr_WxH(res: List[Experiment], left_border: float = None, right_border: float = None, w: int = 3, h: int = 3) -> Tuple[figure, List[axes]]:
    fig, axes = plt.subplots(w, h, figsize=(30, 15))
    for index, ex in enumerate(res):
        ax = axes[index//w][index % h]
        draw_transient(ex, ax=ax, left_border=left_border, right_border=right_border)
        ax.set_title(f"$dx = {ex.method_parameters['dx']} \\quad  dt = {ex.method_parameters['dt']}$", fontsize=22)
    title = f"$p={ex.model_config['p']}, q={ex.model_config['q']}, D_u={ex.model_config['Du']}, init={calc_picks(ex.init_values['u'])}$"
    fig.suptitle(title)
    title = title[1:-1].replace("'","").replace("{", "").replace("}", "").replace(":", "")
    return fig, axes
