from typing import List, Union

import pylab as plt
from matplotlib.lines import Line2D
from numpy import linspace, ndarray
from seaborn import heatmap, color_palette

from DataAnalyzers.PicksAnalyzer import add_peaks_stats_Higgins1D, add_amp_stats_Higgins1D
from DataAnalyzers.TransientAnalyzer import calc_deltas_for_timeline_1D
from DataContainers.Experiment import Experiment
from Drawing.DrawHelper import set_xticks, set_yticks


def draw_amps_stat(results: List[Experiment], L_min: float, L_max: float, xticks: Union[List[float], ndarray] = None,
                   x_name: str = None, yticks: Union[List[float], ndarray] = None, y_name: str = None) -> None:
    for r in results:
        assert r.model_config['model'] == 'Higgins'
        r.method_parameters['x_max'] = r.method_parameters['dx'] * len(r.init_values['u'])
        add_peaks_stats_Higgins1D(r)
        add_amp_stats_Higgins1D(r)
    current_data = [r for r in results if L_min <= r.method_parameters['x_max'] <= L_max]
    ax = plt.gca()
    color_indices = {key: i for i, key in
                     enumerate(set(map(lambda x: x.metadata['end_picks']['u']['picks'], current_data)))}
    palette = color_palette('tab10').as_hex()
    for r in current_data:
        color = palette[color_indices[r.metadata['end_picks']['u']['picks']]]
        ax.scatter(r.method_parameters['x_max'], r.metadata['end_amps']['v'], c=color, marker='+', s=90)
        ax.scatter(r.method_parameters['x_max'], r.metadata['end_amps']['u'], c=color, marker='x', s=80)
    legend_elements = []
    for peaks in sorted(color_indices):
        legend_elements.append(
            Line2D([0], [0], marker='8', label=str(peaks), markerfacecolor=palette[color_indices[peaks]],
                   markersize=10))
    ax.legend(handles=legend_elements, loc='center right')
    ax.set_title(f'Амплитуды итоговых паттернов для u(x) и v(+) для значений L от {L_min} до {L_max}.')

    if xticks is None:
        xticks = linspace(L_min, L_max, 5)
    set_xticks(ax, xticks, '$L$' if x_name is None else x_name)

    if yticks is None:
        yticks = [round(i, 3) for i in ax.get_yticks()[0:-2]]
    set_yticks(ax, yticks, 'Amp' if y_name is None else y_name)
    plt.show()


def draw_transient(e: Experiment, left_boder: float = None, right_border: float = None,
                   xticks: Union[List[float], ndarray] = None,
                   yticks: Union[List[float], ndarray] = None,
                   draw_u_only: bool = True) -> None:
    assert e.timelines
    assert e.timelines['u'].shape == e.timelines['v'].shape
    assert e.method_parameters['dt']

    if left_boder is None:
        left_boder = 0
    assert left_boder >= 0

    if right_border is None:
        right_border = e.timelines['u'].shape[0] - 1
    assert right_border <= e.timelines['u'].shape[0] - 1

    draw_transient_internal(e.timelines['u'].T[::-1, left_boder:right_border], e.method_parameters['dt'],'$t$', '$U$', xticks, yticks)
    if not draw_u_only:
        draw_transient_internal(e.timelines['v'].T[::-1, left_boder:right_border], e.method_parameters['dt'], '$t$', '$V$', xticks, yticks)


def draw_transient_internal(data: ndarray, dt: float, x_name: str, y_name: str,
                            xticks: Union[List[float], ndarray] = None,
                            yticks: Union[List[float], ndarray] = None) -> None:
    ax = heatmap(data)
    if xticks is None:
        quant = 100 if data.shape[1] < 1500 else 500
        r_border = data.shape[1] - data.shape[1] % quant
        xticks = [round(i) for i in linspace(0, r_border, r_border // quant + 1)]
    if data.shape[1] - xticks[-1] > 50:
        xticks += [data.shape[1]]
    set_xticks(ax, xticks, x_name, ticklabels=[x * dt for x in xticks])

    if yticks is None:
        yticks = [round(i, 3) for i in linspace(0, data.shape[0], 5)]
    ticklabels = [str(i) for i in yticks[::-1]]
    ticklabels[0] = y_name
    set_yticks(ax, yticks, ticklabels=ticklabels)
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
        ax.plot(linspace(0, right_border, right_border) * e.method_parameters['dt'], deltas[k][left_border:right_border], label=k)
    set_xticks(ax, [round(i, 2) for i in ax.get_xticks()[1:-1]],'$t$')
    ax.set_yscale('log')
    plt.title('Норма разности между смежными состояниями системы')
    plt.legend()
    plt.show()
