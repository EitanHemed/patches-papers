import typing
from collections import defaultdict
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from . import constants as cn
from .analysis import calc_ci, run_t_test, run_anova
from .reporting import save_analyses_text

TEST_DICT_KEYS = ('x', 'y', 'hypothesis', 'paired')

USE_MARGINS = True

# General figure parameters
DPI = 300
PLOT_EXTEN = '.png'
OUTPUT_PATH = 'Output/{method}/{hashed}/Figures'
INFERENCE_PLOTS_IMG_FNAME = f'{OUTPUT_PATH}/inferential_plots{PLOT_EXTEN}'
DYNAMICS_MODEL_CI_IMG_FNAME = f'{OUTPUT_PATH}/group_cis{PLOT_EXTEN}'

COLORS = ['purple', 'green', 'deepskyblue']
LEGEND_ENTRY_FONTSIZE = 14
AXIS_LABEL_FONTSIZE = 16
FIGURE_FACECOLOR = 'none'

# Parameters related to Bayesian sequential analysis plot
INCONCLUSIVENESS_REGION_LINESTYLE = '--'
INCONCLUSIVENESS_REGION_LINECOLOR = 'k'
LOW_INCONCLUSIVE_LIMIT = 1 / 3
HIGH_INCONCLUSIVE_LIMIT = 3
INCONCLUSIVENESS_REGION_FACECOLOR = 'silver'
SEQUETIAL_BAYES_X_LABEL = 'N (Sample size)'
SEQUENTIAL_BAYES_Y_AXIS_LABEL = ('Δ[$\mathbf{Context_{3}}$ - '
                                 '$\mathbf{Context_{0}}]$\n(Bayes Factor, log-scale)')
SEQUENTIAL_BAYES_SCATTER_MARKER = 'x'
SEQUENTIAL_BAYES_LINE_WIDTH = 3
SEQUENTIAL_BAYES_LINE_STYLE = 'dashed'
INDIVIDUAL_MEANS_SCATTER_MARKER = '.'
INDIVIDUAL_MEANS_SCATTER_SIZE = 15

PLOT_GROUPER_LEVEL_1 = [cn.COLUMN_NAME_PRIOR, cn.COLUMN_NAME_CYCLE_LENGTH]

DYNAMICS_FIG_SIZE = [[8, 6], [9, 9]]
DYNAMICS_FIG_ARRAY = [(1, 2), (2, 2)]

# Parameters related to the contrasts plot
CONTRASTS_PLOT_X_AXIS_TICKS = [[4, 8], [2, 6, 10]]

CONTRASTS_PLOT_SCATTER_DODGE_SIZE = 0.05

CONTRASTS_PLOT_COLUMN_TITLES_NO_FEEDBACK_MANIPULATION = [
    '$\mathbf{Prior_{0}}$ \n No Feedback on Trial N-1',
    '$\mathbf{Prior_{1}}$ \n Feedback on Trial N-1'
]

CONTRASTS_PLOT_COLUMN_TITLES_PERTURBED_FEEDBACK_MANIPULATION = [
    '$\mathbf{Prior_{0}}$ \n Perturbed Feedback on Trial N-1',
    '$\mathbf{Prior_{1}}$ \n Unperturbed Feedback on Trial N-1'
]

COLUMN_TITLES_DICT = dict(
    zip(range(2), [CONTRASTS_PLOT_COLUMN_TITLES_NO_FEEDBACK_MANIPULATION,
                   CONTRASTS_PLOT_COLUMN_TITLES_PERTURBED_FEEDBACK_MANIPULATION]))

LARGE_SCATTER_AREA = 100
SMALL_SCATTER_AREA = 60
LARGE_SCATTER_ALPHA = 1
SMALL_SCATTER_ALPHA = 0.4

# This controls the type of plots that will be produced
CLASSIC_PLOT_MODE = 0
NOVEL_PLOT_MODE = 1
NOVEL_PLOT_MODE = 1

CONTRASTS_FIG_SIZE = [[9, 10], [10, 12]]
CONTRASTS_FIG_ARRAY = [(2, 2), (4, 2)]
LEGEND_STUB_LABELS = (tuple(cn.CYCLE_LENGTH_LEVELS),
                      tuple(cn.NEW_FEEDBACK_LEVELS))
MULTI_FEEDBACK_TYPE_EXPERIMENT_DYNAMICS_ROW_LABELS = tuple(
    cn.CYCLE_LENGTH_LEVELS)

LEGENDS_TITLE = ['Cycle Duration', '']
ROW_INDEXER = [cn.COLUMN_NAME_FEEDBACK_TYPE, cn.COLUMN_NAME_CYCLE_LENGTH]
LINE_INDEXER = ROW_INDEXER[
               ::-1]  # [c.COLUMN_NAME_CYCLE_LENGTH, c.COLUMN_NAME_FEEDBACK_TYPE]

COLUMN_INDEXER = cn.COLUMN_NAME_PRIOR

X_AXIS_TICKS = range(4)
X_AXIS_INDEXER = cn.COLUMN_NAME_CONTEXT
X_AXIS_TITLE = ('$\mathbf{Context}$ - Sum of feedback\n occurrences'
                ' on trials N-4 through N-2')

Y_AXIS_INDEXER = cn.COLUMN_NAME_RESP_TIME
Y_LABEL_TITLE = ('$\mathbf{Mean\ RT}$ \n (ms)')

NUM_OF_X_INDICES = 4

HYPOTHESES = {5: {False: 'x!=y', True: 'x<y'},
              10: {False: 'x!=y', True: 'x<y'}}

# General Figure Aesthetics
PLOTTING_PARAMS = {"font.size": 16,
                   "axes.titlesize": 16,
                   "axes.labelsize": 16,
                   'xtick.labelsize': 14,
                   'ytick.labelsize': 14,
                   'legend.fontsize': 14,
                   'legend.title_fontsize': 14,
                   'legend.markerscale': 0.5,
                   'font.family': 'Arial',
                   'figure.dpi': DPI,
                   'figure.facecolor': FIGURE_FACECOLOR,
                   'legend.borderaxespad': 0.,
                   'legend.facecolor': 'none',
                   'legend.framealpha': 0,
                   }

# The following is copied from Seaborn's `plotting_context` function.
FIGURE_OBJS_SIZES_DICT = {
    "axes.linewidth": 1.25,
    "grid.linewidth": 1,
    "lines.linewidth": 1.5,
    "lines.markersize": 6,
    "patch.linewidth": 1,

    "xtick.major.width": 1.25,
    "ytick.major.width": 1.25,
    "xtick.minor.width": 1,
    "ytick.minor.width": 1,

    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "xtick.minor.size": 4,
    "ytick.minor.size": 4,

}
scaling = 0.8
FIGURE_OBJS_SIZES_DICT = {k: v * scaling for k, v in
                          FIGURE_OBJS_SIZES_DICT.items()}

plt.rcParams.update(PLOTTING_PARAMS)
plt.rcParams.update(FIGURE_OBJS_SIZES_DICT)

STARS_P_VALUE_CUTOFFS = (.05, .01, .001)
STARS_ANNOT_SIG_LABELS = zip(STARS_P_VALUE_CUTOFFS, (0.04, 0.009, 0.0009))


def _get_stars(x):
    return ''.join(
        ['*' for t in STARS_P_VALUE_CUTOFFS if x < t])


STARS_ANNOT_STR = ', '.join([f"{_get_stars(sig)}<{str(lab).lstrip('0')}"
                             for (lab, sig) in STARS_ANNOT_SIG_LABELS
                             ])


class NoReprDefaultDict(defaultdict):
    """A class to suppress output from defaultdict"""

    def __repr__(self):
        return ''


def gen_nested_defaultdict():
    """Returns an instance of a `collections.defaultdict` for which the default
     value is a nested defaultdict (recursively).

    Examples
    --------

    Getting or setting the value under a-non existent key in the dictionary
    recursively creates nested dictionaries to match the required structure.

    >>> d = gen_nested_defaultdict()
    >>> d.values()
    dict_values([])
    >>> d['x']['y']['z'] = 'random value'
    >>> d['x']['y']['z']
    'random value'
    """
    return NoReprDefaultDict(gen_nested_defaultdict)


def analyze_and_plot(df, method):
    """Runs the plotting and analysis pipeline for the current experiment. """
    feedback_manipulation_type = df[
        cn.COLUMN_NAME_FEEDBACK_MANIPULATION_TYPE].unique()[0]

    is_multi_feedback_type_experiment = (
            df[cn.COLUMN_NAME_FEEDBACK_TYPE].nunique() > 1)

    plot_args = get_plot_args(is_multi_feedback_type_experiment, method,
                              feedback_manipulation_type)

    figures_and_axis = get_figure_and_axes(is_multi_feedback_type_experiment,
                                           **plot_args)

    plot_args.update(**figures_and_axis)

    t_test_results = run_t_tests(df, is_multi_feedback_type_experiment)
    anova_results = run_anova(df, is_multi_feedback_type_experiment)

    save_analyses_text(
        t_test_results, anova_results, df,
        is_multi_feedback_type_experiment=is_multi_feedback_type_experiment,
        dynamics_model_coding_method=method)

    plot_dynamics(
        df, anova_results[cn.TEST_KEYS_FREQ],
        is_multi_feedback_type_experiment=is_multi_feedback_type_experiment,
        **plot_args)

    plot_contrasts(df, t_test_results, is_multi_feedback_type_experiment,
                   plot_args)

    return t_test_results, anova_results


def get_figure_and_axes(
        is_multi_feedback_type_experiment: bool,
        dynamics_fig_array=None, dynamics_fig_size=None,
        contrasts_fig_array=None, contrasts_fig_size=None,
        **kwargs, ) -> typing.Dict:
    if is_multi_feedback_type_experiment:

        dynamics_fig, dynamics_axs = plt.subplots(
            *dynamics_fig_array, sharex=True, sharey=True,
            figsize=dynamics_fig_size)

        contrasts_fig = plt.figure(constrained_layout=True,
                                   figsize=contrasts_fig_size, facecolor='none')
        contrasts_subfigs = contrasts_fig.subfigures(2, 1, hspace=0.05,
                                                     facecolor='none')

        subplot_nrows, subplot_ncols = contrasts_fig_array

        axs_top = contrasts_subfigs[0].subplots(subplot_nrows // 2,
                                                subplot_ncols,
                                                sharey=True, sharex=True,
                                                gridspec_kw={'hspace': 0,
                                                             'wspace': 0}
                                                )
        axs_bottom = contrasts_subfigs[1].subplots(
            subplot_nrows // 2, subplot_ncols, sharey=True, sharex=True,
            gridspec_kw={'hspace': 0, 'wspace': 0})

        contrasts_axs = np.stack([axs_top, axs_bottom], axis=0)

    else:

        fig = plt.figure(constrained_layout=True,
                         figsize=contrasts_fig_size, facecolor='none')

        dynamics_fig, top_contrasts_subfig, bottom_contrasts_subfig = (
            fig.subfigures(3, 1, hspace=0.025, wspace=0,
                           height_ratios=[0.45, 0.35, 0.3], facecolor='none'))

        dynamics_axs = dynamics_fig.subplots(
            1, 2, sharey=True, sharex=True,
            gridspec_kw={'hspace': 0.0375, 'wspace': 0})

        subplot_nrows, subplot_ncols = contrasts_fig_array

        axs_top = top_contrasts_subfig.subplots(
            subplot_nrows // 2, subplot_ncols, sharey=True, sharex=True,
            gridspec_kw={'hspace': .005, 'wspace': 0})

        axs_bottom = bottom_contrasts_subfig.subplots(
            subplot_nrows // 2, subplot_ncols, sharey=True, sharex=True,
            gridspec_kw={'hspace': .05, 'wspace': 0})

        contrasts_axs = np.stack([axs_top, axs_bottom], axis=0)

        # Remove the references to the nested figures
        contrasts_fig = fig
        dynamics_fig = fig

    return {'dynamics_fig': dynamics_fig,
            'dynamics_axs': dynamics_axs,
            'contrasts_fig': contrasts_fig,
            'contrasts_axs': contrasts_axs, }


def get_plot_args(is_multi_feedback_type_experiment: bool,
                  dynamics_model_coding_method: str,
                  feedback_manipulation_type: int):
    if is_multi_feedback_type_experiment:
        plot_mode = NOVEL_PLOT_MODE
    else:
        plot_mode = CLASSIC_PLOT_MODE

    return {
        'plot_mode': plot_mode,
        'dynamics_fig_size': DYNAMICS_FIG_SIZE[plot_mode],
        'dynamics_fig_array': DYNAMICS_FIG_ARRAY[plot_mode],
        'contrasts_fig_array': CONTRASTS_FIG_ARRAY[plot_mode],
        'contrasts_fig_size': CONTRASTS_FIG_SIZE[plot_mode],
        'contrasts_x_axis_ticks': CONTRASTS_PLOT_X_AXIS_TICKS[plot_mode],
        'legend_stub_labels': LEGEND_STUB_LABELS[plot_mode],
        'legends_title': LEGENDS_TITLE[plot_mode],
        'row_indexer': ROW_INDEXER[plot_mode],
        'line_indexer': LINE_INDEXER[plot_mode],
        'columns_titles': COLUMN_TITLES_DICT[feedback_manipulation_type],
        'contrasts_plot_output_path': INFERENCE_PLOTS_IMG_FNAME.format(
            method=dynamics_model_coding_method,
            hashed=cn.HASHED_SCREENING_PARAMS),
        'dynamics_model_plot_output_path': DYNAMICS_MODEL_CI_IMG_FNAME.format(
            method=dynamics_model_coding_method,
            hashed=cn.HASHED_SCREENING_PARAMS),
    }


def run_t_tests(df, multi_feedback_type_experiment):
    t_test_results = gen_nested_defaultdict()

    for name, group in df.groupby([cn.COLUMN_NAME_CYCLE_LENGTH,
                                   cn.COLUMN_NAME_PRIOR,
                                   cn.COLUMN_NAME_FEEDBACK_TYPE]):

        y, x = group.query(
            f'{cn.COLUMN_NAME_CONTEXT}.isin([0, 3])').pivot(
            index=cn.COLUMN_NAME_UID, columns=cn.COLUMN_NAME_CONTEXT,
            values=cn.COLUMN_NAME_RESP_TIME).values.T

        patch_length, prior, feedback_type = name

        _t_args = dict(zip(TEST_DICT_KEYS,
                           (x, y, HYPOTHESES[patch_length][prior], True)))

        for i in [cn.TEST_KEYS_FREQ, cn.TEST_KEYS_BAYES,
                  cn.T_TEST_KEYS_SEQUENTIAL_BAYES]:
            t_test_results[name][i] = run_t_test(test_type=i, **_t_args)

    return t_test_results


def _prep_summaries_for_errorbar(df, margins, line_indexer=None, ):
    grouper = [line_indexer, X_AXIS_INDEXER]

    if USE_MARGINS:
        mean_y_vals = margins.groupby(grouper)['emmean'].first().values.reshape(
            df[line_indexer].nunique(),  # len(LEGEND_STUB_LABELS),
            NUM_OF_X_INDICES)

        cis = margins.groupby(grouper)[['emmean', 'upper.CL']].first(
        ).diff(axis=1).dropna(axis=1).values.reshape(
            df[line_indexer].nunique(), NUM_OF_X_INDICES
        )
    else:
        mean_y_vals = df.groupby(grouper)[Y_AXIS_INDEXER].mean().values.reshape(
            df[line_indexer].nunique(),  # len(LEGEND_STUB_LABELS),
            NUM_OF_X_INDICES)

        cis = df.groupby(grouper)[Y_AXIS_INDEXER].apply(
            lambda a: calc_ci(a)).values.reshape(
            df[line_indexer].nunique(), NUM_OF_X_INDICES)

    y_ns = df.groupby(line_indexer)[cn.COLUMN_NAME_UID].nunique().values

    return mean_y_vals, cis, y_ns


def _plot_errorbar(y_means, y_cis, y_ns, ax, legend_labels=None):
    num_of_groups = len(legend_labels)

    context_x_vals = np.tile(np.arange(0, NUM_OF_X_INDICES),
                             num_of_groups).reshape(
        num_of_groups, NUM_OF_X_INDICES) + np.linspace(
        -1, 1, num_of_groups).reshape(-1, 1
                                      ) * CONTRASTS_PLOT_SCATTER_DODGE_SIZE

    unique_colors = COLORS[:num_of_groups]

    ax.scatter(
        x=context_x_vals[:, [0, -1]], y=y_means[:, [0, -1]],
        s=LARGE_SCATTER_AREA,
        c=np.repeat(unique_colors, 2))

    ax.scatter(x=context_x_vals[:, 1: -1], y=y_means[:, 1: -1],
               alpha=SMALL_SCATTER_ALPHA,
               s=SMALL_SCATTER_AREA,
               c=list(np.reshape(
                   np.repeat(unique_colors, y_means.shape[1] - 2),
                   (num_of_groups, -1)).flat)
               )

    for i in range(num_of_groups):
        ax.plot(context_x_vals[i, :], y_means[i, :], c=unique_colors[i],
                label=f'{legend_labels[i]} (N = {y_ns[i]})')

        # Unfortunately alpha can't be specified as an array, only as scaler,
        # and we repeat ourselves a bit
        for slc, alpha in zip([[0, 3], [1, 2]],
                              [LARGE_SCATTER_ALPHA, SMALL_SCATTER_ALPHA]):
            ax.errorbar(x=context_x_vals[i, slc], y=y_means[i, slc],
                        yerr=y_cis[i, slc], ls='none', alpha=alpha,
                        color=unique_colors[i], )


def plot_dynamics(df, anova_results, row_indexer=ROW_INDEXER,
                  column_indexer=COLUMN_INDEXER, line_indexer=None,
                  legend_title=None, dynamics_fig_array=None,
                  dynamics_fig_size=None, plot_mode=None,
                  legend_stub_labels=None, x_axis_indexer=X_AXIS_INDEXER,
                  y_axis_values=Y_AXIS_INDEXER,
                  columns_titles=CONTRASTS_PLOT_COLUMN_TITLES_NO_FEEDBACK_MANIPULATION,
                  y_label_title=Y_LABEL_TITLE, x_label_title=X_AXIS_TITLE,
                  x_ticks=X_AXIS_TICKS, dynamics_model_plot_output_path=None,
                  dynamics_axs=None, dynamics_fig=None,
                  is_multi_feedback_type_experiment: bool = False,
                  **kwargs):
    numeric_margins_terms = [cn.COLUMN_NAME_CONTEXT,
                             cn.COLUMN_NAME_CYCLE_LENGTH,
                             cn.COLUMN_NAME_PRIOR]

    margins_terms = deepcopy(numeric_margins_terms)

    # On multi feedback type experiments
    if line_indexer == cn.COLUMN_NAME_FEEDBACK_TYPE:
        margins_terms.append(cn.COLUMN_NAME_FEEDBACK_TYPE)
        groupby_terms = [row_indexer, column_indexer]
    else:
        groupby_terms = column_indexer

    margins_results = anova_results.get_margins(
        margins_terms=margins_terms)

    margins_results[numeric_margins_terms] = margins_results[
        numeric_margins_terms].apply(
        lambda s: s.str.replace("X", "")).apply(pd.to_numeric).values

    for (name, group), (_, margins), ax in zip(
            df.groupby(groupby_terms),
            margins_results.groupby(groupby_terms),
            dynamics_axs.flat):
        _plot_errorbar(
            *_prep_summaries_for_errorbar(
                group, margins, line_indexer=line_indexer), ax,
            legend_labels=legend_stub_labels)

    if plot_mode == NOVEL_PLOT_MODE:
        [ax.text(s=i, x=1.05, y=0.5,
                 transform=ax.transAxes,
                 rotation=-90, ha='center', va='center',
                 fontsize=AXIS_LABEL_FONTSIZE, weight='bold'
                 ) for i, ax in
         zip(MULTI_FEEDBACK_TYPE_EXPERIMENT_DYNAMICS_ROW_LABELS,
             dynamics_axs[:, 1].flat)]

    dynamics_axs = dynamics_axs.reshape((-1, 2))

    [ax.set_title(i, fontsize=AXIS_LABEL_FONTSIZE) for i, ax in
     zip(columns_titles, dynamics_axs[0, :])]

    dynamics_axs[-1, 0].set_ylabel(y_label_title, )
    dynamics_axs[-1, 0].set_xlabel(x_label_title, )
    dynamics_axs[-1, 0].set(xticks=x_ticks)

    legends = []
    for _ax in dynamics_axs[:, 1]:
        leg = _ax.legend(title=legend_title, facecolor=None,
                         # fontsize=LEGEND_ENTRY_FONTSIZE,
                         columnspacing=1,
                         labelspacing=0.2,
                         handlelength=0,
                         markerscale=0,
                         )
        for _color, text in zip(COLORS, leg.get_texts()):
            text.set_color(_color)

    dynamics_fig.tight_layout(h_pad=0.01, w_pad=0.01)

    if is_multi_feedback_type_experiment:
        # For the other experiments, we continue to plot using the same figure
        dynamics_fig.savefig(dynamics_model_plot_output_path, dpi=DPI,
                             facecolor=FIGURE_FACECOLOR)


def plot_contrasts(
        df: pd.DataFrame, t_test_results: typing.Dict,
        is_multi_feedback_type_experiment: bool,
        plot_args: dict):
    top_axs, bottom_axs = np.split(plot_args['contrasts_axs'], 2, axis=0)
    fig = plot_args['contrasts_fig']

    if is_multi_feedback_type_experiment:
        axis_grouper = [cn.COLUMN_NAME_CYCLE_LENGTH, cn.COLUMN_NAME_PRIOR]
        line_grouper = cn.COLUMN_NAME_FEEDBACK_TYPE
    else:
        axis_grouper = [cn.COLUMN_NAME_PRIOR, cn.COLUMN_NAME_FEEDBACK_TYPE]
        line_grouper = cn.COLUMN_NAME_CYCLE_LENGTH

    plot_wingplot(df, t_test_results,
                  top_axs,  # axs[:lowest_row_of_ci_plots, :],
                  is_multi_feedback_type_experiment,
                  axis_grouper, line_grouper,
                  **plot_args
                  )
    plot_sequential_bayes(df, t_test_results,
                          bottom_axs,  # axs[lowest_row_of_ci_plots:, :],
                          is_multi_feedback_type_experiment,
                          axis_grouper, line_grouper,
                          **plot_args)

    # fig.tight_layout(h_pad=0.01, w_pad=0.01)
    fig.savefig(
        plot_args['contrasts_plot_output_path'],
        dpi=DPI, facecolor='white')

    # return fig


def plot_wingplot(raw_data, test_data, axs,
                  is_multi_feedback_type_experiment,
                  axis_grouper,
                  line_grouper,
                  contrasts_x_axis_ticks=None,
                  legend_stub_labels=None,
                  columns_titles=None,
                  **kwargs):
    max_delta, min_delta = 0, 0

    for (name, group), (ax_index, ax) in zip(raw_data.groupby(
            axis_grouper), enumerate(axs.flat)):

        for (n, g), _color, pos in zip(group.groupby(line_grouper),
                                       COLORS, contrasts_x_axis_ticks):

            # patch_length, prior, feedback_type
            if is_multi_feedback_type_experiment:
                k = tuple([name[0], name[1], n])
            else:
                k = tuple([n, name[0], name[1]])

            current_t = test_data[k][cn.TEST_KEYS_FREQ]

            # Get the vector of differences between the RTs
            x, y = current_t.x, current_t.y
            delta = x - y

            _cur_min_delta = np.min(delta)
            _cur_max_delta = np.max(delta)

            if _cur_max_delta > max_delta:
                max_delta = _cur_max_delta
            if _cur_min_delta < min_delta:
                min_delta = _cur_min_delta

            test_cohen_and_pvalue = list(current_t.report_table().iloc[0][
                                             ['Cohen-d',
                                              'p-value']].to_dict().values())

            s = (
                f'{test_cohen_and_pvalue[0]:.2f}{_get_stars(test_cohen_and_pvalue[1])}')

            ax.scatter(np.linspace(pos - 3, pos + 3, x.size),
                       np.sort(delta), alpha=0.4,
                       marker=INDIVIDUAL_MEANS_SCATTER_MARKER, c=_color,
                       s=INDIVIDUAL_MEANS_SCATTER_SIZE)

            ax.errorbar(pos, delta.mean(), yerr=calc_ci(delta), c=_color,
                        label=s, elinewidth=2)

        ax.set(
            xlim=[contrasts_x_axis_ticks[0] - 4,
                  contrasts_x_axis_ticks[-1] + 4],
            xticks=contrasts_x_axis_ticks, xticklabels=[], )

        ax.axhline(0, ls='--', c='k')

        leg = ax.legend(title=f"Cohen's d ({STARS_ANNOT_STR})"
        if ax_index == 0 else '',
                        labelspacing=0.1, borderpad=0.1,
                        title_fontproperties={'weight': 'bold'},
                        ncol=3,
                        handletextpad=0.25, handlelength=0, markerscale=0,
                        # 1.5,
                        columnspacing=1, )

        leg._legend_box.align = "left"

        for _color, text in zip(COLORS, leg.get_texts()):
            text.set_color(_color)

    axs = axs.reshape((-1, 2))

    min_x_lim, max_x_lim = axs[-1, 0].get_xlim()

    for _color, pos, l in zip(COLORS, contrasts_x_axis_ticks,
                              legend_stub_labels):
        axs[-1, 0].annotate(
            xy=[(pos + abs(min_x_lim)) / np.sum(np.abs((min_x_lim, max_x_lim))),
                -0.125],
            ha='center', va='center', c=_color, rotation=30,
            text=l.replace(' ', '\n'),
            xycoords='axes fraction',
            fontsize=AXIS_LABEL_FONTSIZE * 0.75
        )

    axs[-1, 0].set_ylabel(
        'Δ[$\mathbf{Context_{3}}$ - $\mathbf{Context_{0}}]$ \n(ms)')

    if is_multi_feedback_type_experiment:
        [ax.set_title(i, fontsize=AXIS_LABEL_FONTSIZE) for i, ax in
         zip(columns_titles, axs[0, :])]

        [ax.text(s=str(i).replace(' ', '\n'), x=1.1, y=0.5,
                 fontsize=AXIS_LABEL_FONTSIZE,
                 transform=ax.transAxes, rotation=-90, ha='center',
                 ) for i, ax in zip(cn.CYCLE_LENGTH_LEVELS,
                                    axs[:, 1].flat)]


def plot_sequential_bayes(raw_data: pd.DataFrame, test_data: typing.Dict,
                          axs: np.ndarray,
                          is_multi_feedback_type_experiment,
                          axis_grouper,
                          line_grouper,
                          legends_title=None,
                          **kwargs
                          ) -> None:
    """
    Plot the sequential bayes test results.
    :param raw_data:
        A dataframe with the raw data.
    :param test_data:
        A dictionary with the test results.
    :param axs:
        The axes to plot on.
    :return: None
    """

    max_group_size = raw_data.groupby([cn.COLUMN_NAME_CYCLE_LENGTH,
                                       cn.COLUMN_NAME_FEEDBACK_TYPE]).apply(
        lambda s: s[cn.COLUMN_NAME_UID].nunique()
    ).max() + 4

    for (name, group), (ax_index, ax) in zip(
            raw_data.groupby(axis_grouper), enumerate(axs.flat)):


        rect = Rectangle(
            (0, LOW_INCONCLUSIVE_LIMIT), max_group_size,
            HIGH_INCONCLUSIVE_LIMIT - LOW_INCONCLUSIVE_LIMIT,
            linewidth=1, edgecolor=None,
            facecolor=INCONCLUSIVENESS_REGION_FACECOLOR)
        ax.add_patch(rect)

        ax.axhline(LOW_INCONCLUSIVE_LIMIT, ls=INCONCLUSIVENESS_REGION_LINESTYLE,
                   c=INCONCLUSIVENESS_REGION_LINECOLOR)
        ax.axhline(HIGH_INCONCLUSIVE_LIMIT,
                   ls=INCONCLUSIVENESS_REGION_LINESTYLE,
                   c=INCONCLUSIVENESS_REGION_LINECOLOR)

        texts_to_adjust = []

        for (n, g), _color in zip(raw_data.groupby(line_grouper),
                                  COLORS):

            # patch_length, prior, feedback_type
            if is_multi_feedback_type_experiment:
                k = tuple([name[0], name[1], n])
            else:
                k = tuple([n, name[0], name[1]])

            bfs = test_data[k][cn.T_TEST_KEYS_SEQUENTIAL_BAYES]

            ax.plot(np.arange(len(bfs)), bfs, c=_color,
                    lw=SEQUENTIAL_BAYES_LINE_WIDTH,
                    ls=SEQUENTIAL_BAYES_LINE_STYLE, label=None)

            ax.scatter(len(bfs), bfs[-1],
                       # _color=_color,
                       # marker=SEQUENTIAL_BAYES_SCATTER_MARKER,
                       facecolors='none', edgecolors=_color,
                       s=LARGE_SCATTER_AREA,
                       label=f'{bfs[-1]:.2E}')

        leg = ax.legend(  # ncol=len(line_grouper),
            title='$\mathbf{BF_{1:0}}$' if ax_index == 0 else " ",
            # fontsize=LEGEND_ENTRY_FONTSIZE,
            columnspacing=0.5,
            labelspacing=0.2,
            loc='upper left',
            handletextpad=0, handlelength=0, markerscale=0,  # 1.5,
        )

        for _color, text in zip(COLORS, leg.get_texts()):
            text.set_color(_color)

        ax.set_yscale('log')
        ax.set(
            # ylim=[0.01, 1e6],
            xlim=[0, max_group_size],
            xticklabels=[],
        )

    axs = axs.reshape((-1, 2))

    if is_multi_feedback_type_experiment:
        [ax.text(s=str(i).replace(' ', '\n'), x=1.1, y=0.5,
                 fontsize=AXIS_LABEL_FONTSIZE,
                 transform=ax.transAxes, rotation=-90, ha='center'
                 ) for i, ax in zip(cn.CYCLE_LENGTH_LEVELS,
                                    axs[:, 1].flat)]

    _tick_distances = 10 if max_group_size < 80 else 15
    axs[-1, 0].set(xticks=range(0, max_group_size, _tick_distances),
                   xticklabels=range(0, max_group_size, _tick_distances
                                     ))
    axs[-1, 0].set_xlabel(SEQUETIAL_BAYES_X_LABEL, fontsize=AXIS_LABEL_FONTSIZE)

    axs[-1, 0].set_ylabel(SEQUENTIAL_BAYES_Y_AXIS_LABEL,
                          fontsize=AXIS_LABEL_FONTSIZE)
