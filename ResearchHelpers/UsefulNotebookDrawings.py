import typing as tp
from collections import defaultdict

import numpy as np
import pylab as plt
from frozendict import frozendict
from tqdm import tqdm

from MyPackage.DataAnalyzers.PeaksAnalyzer import calc_mean_squared_Fourier_for_experiment
from MyPackage.DataAnalyzers.PeaksAnalyzer import calc_peaks_by_Fourier
from MyPackage.DataContainers.Experiment import Experiment
from MyPackage.DataContainers.MultipleExperimentContainer import MultipleExperimentContainer
from MyPackage.DataContainers.PeakPatternType import PeakPatternType, PatternDirection
from MyPackage.Drawing.DrawHelper import set_defaults_1D
from MyPackage.Drawing.DrawPattern1D import draw_pattern

COLORS = dict(zip([round(i / 2, 1) for i in range(1, 11)], plt.cm.get_cmap('tab10').colors))


def get_and_draw_patterns(data: MultipleExperimentContainer, var_name: str, key_fields: tp.Set[str],
                          filter_condition: tp.Dict[str, tp.Any] = None, max_peaks_to_check: float = 5.0) \
        -> tp.Dict[str, tp.List[str]]:
    set_defaults_1D()

    conf_common, confs_uniq = data.get_configs_uniq(params_to_include=key_fields, filter_condition=filter_condition)
    configs_to_peaks: tp.DefaultDict[frozendict, tp.Set[PeakPatternType]] = defaultdict(set)
    for c in sorted(list(confs_uniq), key=lambda x: tuple(x.items())):
        to_draw: tp.Dict[PeakPatternType, Experiment] = {}
        exs = data.get_experiments_by_filter(filter_condition=dict(**filter_condition, **conf_common, **c))
        ax = plt.gca()
        for e in exs:
            peaks = calc_peaks_by_Fourier(e, var_name, max_peaks_count=max_peaks_to_check)
            configs_to_peaks[c].add(peaks)
            if peaks not in to_draw:
                to_draw[peaks] = e
        for pattern in sorted(list(to_draw), key=lambda x: x.peaks_count):
            draw_pattern(to_draw[pattern], var_name, color=COLORS[pattern.peaks_count], label=str(pattern),
                         linestyle='-' if pattern.direction == PatternDirection.UP else '--', ax=ax)
        ax.set_title(str(dict(**c)).replace("'", '')[1:-1])
        plt.legend()
        plt.show()
    res: tp.Dict[str, tp.List[str]] = {}
    for key in configs_to_peaks:
        new_key = str(dict(**key)).replace("'", '')[1:-1]
        res[new_key] = []
        for pattern in configs_to_peaks[key]:
            res[new_key].append(str(pattern))
    return res


def draw_w_k_dynamics(data: MultipleExperimentContainer, var_name: str, key_field: str,
                      k_to_check: tp.Union[float, tp.List[float]], filter_condition: tp.Dict[str, tp.Any] = None,
                      logscale: bool = False, left_border: float = None, right_border: float = None) \
        -> None:
    conf_common, confs_uniq = data.get_configs_uniq(params_to_include={key_field}, filter_condition=filter_condition)
    confs_uniq_sorted = sorted(list(confs_uniq), key=lambda x: tuple(x.items()))
    to_draw: tp.Dict[frozendict, tp.Dict[float, float]] = {}
    set_defaults_1D()
    ax = plt.gca()
    for i, c in enumerate(confs_uniq_sorted):
        exs = data.get_experiments_by_filter(filter_condition=dict(**filter_condition, **conf_common, **c))
        c = c[key_field]
        wks_merged = defaultdict(list)
        for current_wks in tqdm(
                map(lambda ex: calc_mean_squared_Fourier_for_experiment(ex, k_to_check, var_name, left_border,
                                                                        right_border), exs),
                total=len(exs),
                desc=f'{i + 1}/{len(confs_uniq_sorted)}'):
            for key in current_wks:
                wks_merged[key].append(current_wks[key])
        to_draw[c] = {}
        for w_k in wks_merged:
            to_draw[c][w_k] = float(np.mean(wks_merged[w_k]))
    xs = list(range(len(to_draw)))
    for k in k_to_check:
        ys = []
        for primary_key in sorted(list(to_draw)):
            ys.append(to_draw[primary_key][k])
        ax.plot(xs, ys, c=COLORS[k], label=f'$W_{{{k}}}$', lw=2)
    ax.set_xticks(xs)
    ax.set_xticklabels(sorted(list(to_draw.keys())), rotation=45)
    ax.set_ylabel('$W_k$')
    ax.set_xlabel(key_field)
    if logscale:
        ax.set_yscale('log')
    ax.set_title(str(filter_condition).replace("'",'')[1:-1])
    plt.legend()
    plt.show()
