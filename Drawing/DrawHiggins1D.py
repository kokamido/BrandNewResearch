from typing import List

import pylab as plt
import seaborn as sns
from matplotlib.lines import Line2D

from DataAnalyzers.PicksAnalyzer import add_peaks_stats_Higgins1D, add_amp_stats_Higgins1D
from DataContainers.Experiment import Experiment


def draw_amps_stat(results: List[Experiment], L_min: float, L_max: float) -> None:
    for r in results:
        assert r.model_config['model'] == 'Higgins'
        r.method_parameters['x_max'] = r.method_parameters['dx'] * len(r.init_values['u'])
        add_peaks_stats_Higgins1D(r)
        add_amp_stats_Higgins1D(r)
    current_data = [r for r in results if L_min <= r.method_parameters['x_max'] <= L_max]
    ax = plt.gca()
    color_indices = {key: i for i, key in
                     enumerate(set(map(lambda x: x.metadata['end_picks']['u']['picks'], current_data)))}
    palette = sns.color_palette('tab10').as_hex()
    for r in current_data:
        L = r.method_parameters['x_max']
        color = palette[color_indices[r.metadata['end_picks']['u']['picks']]]
        ax.scatter(r.method_parameters['x_max'], r.metadata['end_amps']['v'], c=color, marker='+', s=90)
        ax.scatter(r.method_parameters['x_max'], r.metadata['end_amps']['u'], c=color, marker='x', s=80)
    legend_elements = []
    for peaks in sorted(color_indices):
        legend_elements.append(
            Line2D([0], [0], marker='8', label=str(peaks), markerfacecolor=palette[color_indices[peaks]],
                   markersize=10))
    ax.legend(handles=legend_elements, loc='center right')
    ax.set_title(f'Амплитуды итоговых паттернов для u и v для значений L от {L_min} до {L_max}.')
    xtcks = [str(x) for x in ax.get_xticks()]
    xtcks[-2] = 'L'
    ax.set_xticklabels(xtcks)
    ytcks = [str(round(y,3)) for y in ax.get_yticks()]
    ytcks[-2] = 'Amp'
    ax.set_yticklabels(ytcks)
    plt.show()
