from datetime import datetime
import typing as tp
from collections import defaultdict

import numpy as np
import pylab as plt
from frozendict import frozendict
from tqdm import tqdm

from MyPackage.DataAnalyzers.PeaksAnalyzer import calc_mean_squared_Fourier_for_experiment
from MyPackage.DataAnalyzers.PeaksAnalyzer import calc_peaks_by_Fourier
from MyPackage.DataContainers.Experiment import Experiment
from MyPackage.DataContainers.ExperimentHelper import suggest_time_borders
from MyPackage.DataContainers.MultipleExperimentContainer import MultipleExperimentContainer
from MyPackage.DataContainers.PeakPatternType import PeakPatternType, PatternDirection
from MyPackage.Drawing.DrawHelper import set_defaults_1D
from MyPackage.Drawing.DrawPattern1D import draw_pattern
from MyPackage.Drawing.DrawTransient1D import draw_few_Fouriers, draw_transient, draw_means_squared_Fouriers, \
    draw_few_Fouriers_abs_heatmap

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
                      logscale: bool = False, left_border: float = None, right_border: float = None, legend_loc=4, ax = None,
                      divide_by: tp.Optional[float] = None) \
        -> None:
    conf_common, confs_uniq = data.get_configs_uniq(params_to_include={key_field}, filter_condition=filter_condition)
    confs_uniq_sorted = sorted(list(confs_uniq), key=lambda x: tuple(x.items()))
    to_draw: tp.Dict[frozendict, tp.Dict[float, float]] = {}
    set_defaults_1D()
    ax = ax or plt.gca()
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
            if divide_by:
                ys[-1] /= to_draw[primary_key][divide_by]
        ax.plot(xs, ys, c=COLORS[k], label= f'$W_{{{k}}}/W_{{{divide_by}}}$' if divide_by is not None else f'$W_{{{k}}}$', lw=4)
    ax.set_xticks(xs)
    ax.set_xticklabels([round(x,5) for x in sorted(list(to_draw.keys()))], rotation=45)
    ax.set_ylabel('$W_k$')
    ax.set_xlabel(key_field)
    if logscale:
        ax.set_yscale('log')
    ax.set_title(str(filter_condition).replace("'", '')[1:-1])
    plt.legend(loc=legend_loc)
    plt.show()


def draw_quadreega(ex: Experiment, ks: tp.List[float], var_name: str, left_border_t: tp.Optional[float] = None,
                   right_border_t: tp.Optional[float] = None, top_n_coeffs: int = 1,
                   coeffs_legend_loc: str = None) -> None:
    set_defaults_1D()
    left_border_t, right_border_t = suggest_time_borders(ex, left_border_t, right_border_t)
    fig, axes = plt.subplots(2, 2, figsize=(28, 14))
    draw_transient(ex, var_name,left_border_t, right_border_t, ax=axes[0][0],
                   cbar_kws={'use_gridspec': False, 'location': 'top'})
    draw_few_Fouriers(ex, ks, left_border_t, right_border_t, ax=axes[0][1], top_n=top_n_coeffs,
                      coeffs_legend_loc=coeffs_legend_loc)
    draw_means_squared_Fouriers(ex, ks, var_name, left_border_t, right_border_t, ax=axes[1][1])
    draw_few_Fouriers_abs_heatmap(ex, ks, var_name, left_border_t, right_border_t, ax=axes[1][0],
                                  cmap='nipy_spectral_r')
    fig.suptitle(str({**ex.model_config, 'noise_amp': ex.method_parameters['noise_amp']}))
    plt.show()


def draw_mean_wk_points(data: MultipleExperimentContainer, var_name: str, ks: tp.List[float], logscale: bool = False):
    set_defaults_1D()
    ax = plt.gca()
    res = {k: 0 for k in ks}
    for i, noise in enumerate(data.get_uniq_values('noise_amp')[:10]):
        exps = data.get_experiments_by_filter({'noise_amp': noise})
        total = len(exps)
        exps = [e for e in exps if e.timelines['u'].size > 500]
        for wks in tqdm(map(lambda e: calc_mean_squared_Fourier_for_experiment(e, ks, var_name), exps), total=total):
            for k in wks:
                res[k] += wks[k]
        for k in ks:
            res[k] /= len(exps)
        ax.scatter(res.keys(), res.values(), color=COLORS[round((i + 1) / 2, 1)], s=200, label=noise)
    if logscale:
        ax.set_yscale('log')
    plt.suptitle(str(data.get_experiments_by_filter({})[0].model_config)[1:-1])
    plt.legend()
    plt.show()
