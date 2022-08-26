import itertools
import os
import typing

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy as sp
from matplotlib import pyplot as plt
from scipy.stats import pearsonr

from po_utils import constants as cn
from po_utils import plotting

# from po_utils import save_json

FIGSIZE = (8.5, 11)

COLORS = plotting.COLORS[:2]

PLOT_SUBPLOTS_TUPLE = (2, 1)

POOLED_RESULTS_EXPERIMENT_ID = ''
COLUMN_NAME_EXPERIMENT_ID = 'experiment'
COLUMN_NAME_PAPER_ID = 'paper'
COLUMN_NAME_EFFECT_SIZE = 'cohen'

COLUMN_NAME_RAW_FREQUENCY_WEIGHT = 'raw_frequency_weight'
COLUMN_NAME_RAW_INVERSE_VARIANCE_WEIGHTS = 'raw_delta_inverse_variance'

COLUMN_NAME_COLUMN_NAME_STANDARDIZED_FREQUENCY_WEIGHT = 'standardized_frequency_weight'
COLUMN_NAME_STANDARDIZED_INVERSE_VARIANCE_WEIGHTS = 'standardized_delta_inverse_variance_weights'

SUMMARY_ROW_GROUP_ORDER = ['patch_length', 'prior']

OUTPUT_COLUMN_NAMES = [
    cn.COLUMN_NAME_CYCLE_LENGTH, cn.COLUMN_NAME_PRIOR,
    cn.COLUMN_NAME_FEEDBACK_TYPE,
    'delta_m', 'delta_sd', 'n', 'mx', 'my', 'sdx', 'sdy', 'cohen', 'ci_low',
    'ci_high', 't', 'p_value', 'r', 'r_p_value',
]

META_ANALYSIS_TABLE_COLUMNS = ['Paper', 'Exp. #', 'N',
                               'Weight (%)',
                               'ES', '95%-CI', 'p-value', ]

OUTPUT_PATH = 'Output/{method}/{hashed_json}/Texts/summary.csv'

plt.rcParams.update(plotting.PLOTTING_PARAMS)


def extract_contrast_summary(t):
    """

    :param t:

    :return:
    """

    delta_m = (t.x - t.y).mean()
    delta_sd = np.std(t.x - t.y)

    mx, my, sdx, sdy, n = (t.x.mean(), t.y.mean(), np.std(t.x), np.std(t.y),
                           t.x.size)

    ci_low, cohen, ci_high = t.report_table().filter(
        regex='Cohen').values.T.reshape(-1)

    p_value, t_statistic = t.report_table()[['p-value', 't']].values.T.reshape(
        -1)

    r, r_p_value = pearsonr(t.x, t.y)

    return (delta_m, delta_sd, n, mx, my, sdx, sdy, cohen, ci_low, ci_high,
            t_statistic, p_value,
            r, r_p_value)


def pool_t_test_results_for_single_experiment(ts_dict, method):
    current_exp_name = os.getcwd().split('\\')[-1]

    # We only need control feedback
    feedback_types = cn.FEEDBACK_TYPE_SINGLE
    results = []

    for patch_length in cn.CYCLE_LENGTHS:
        for prior in cn.PRIOR_LEVELS:
            for feedback_type in feedback_types:
                group_name = (patch_length, prior, feedback_type)
                freq_test = ts_dict[group_name][cn.TEST_KEYS_FREQ]
                results.append(group_name + extract_contrast_summary(freq_test))

    df = pd.DataFrame(results,
                      columns=OUTPUT_COLUMN_NAMES)

    df[COLUMN_NAME_EXPERIMENT_ID] = current_exp_name

    df.to_csv(OUTPUT_PATH.format(method=method,
                                 hashed_json=cn.HASHED_SCREENING_PARAMS),
              index=False)


def save_summaries_from_all_experiments(exp_list: typing.List[str],
                                        method: str) -> None:
    os.makedirs(f'meta_analysis/{method}/{cn.HASHED_SCREENING_PARAMS}',
                exist_ok=True)

    df = pd.concat(
        [pd.read_csv(f'{e}/' + OUTPUT_PATH.format(
            method=method, hashed_json=cn.HASHED_SCREENING_PARAMS)
                     ).assign(paper=e.split('\\')[-2]) for e in exp_list])

    df = df.sort_values([COLUMN_NAME_PAPER_ID, COLUMN_NAME_EXPERIMENT_ID, ],
                        ascending=[False, True])
    df.to_csv(
        f'meta_analysis/{method}/{cn.HASHED_SCREENING_PARAMS}/summaries.csv',
        index=False)

    # save_json(method, 'meta_analysis')


def load_data_for_meta_analysis_plot(method: str) -> pd.DataFrame:
    return pd.read_csv(
        f'meta_analysis/{method}/{cn.HASHED_SCREENING_PARAMS}/summaries.csv')


def draw_meta_fig(method: str) -> pd.DataFrame:
    data = prep_data_for_meta_analysis(method)

    fig, axs = plt.subplots(
        *PLOT_SUBPLOTS_TUPLE, figsize=FIGSIZE, sharex=True)

    draw_forest_plot(data, axs)
    draw_table(data, fig, axs)

    fig.tight_layout()

    fig.savefig(
        f'meta_analysis/{method}/{cn.HASHED_SCREENING_PARAMS}/forest_plot.jpg',
        dpi=300, bbox_inches='tight')

    return data


def draw_forest_plot(data, axs):

    n_studies = data.groupby([COLUMN_NAME_EXPERIMENT_ID,
                              COLUMN_NAME_PAPER_ID]).size().shape[0]

    y_locs = np.linspace(0, n_studies * 2 - 1, n_studies * 2)
    y_vals = np.fliplr(y_locs.reshape(n_studies, 2).T)

    scatter_sizes = np.ones(n_studies) * plt.rcParams['lines.markersize'] ** 2
    scatter_sizes[-1] *= 4

    for (name, group), ax in zip(
            data.loc[
                data[COLUMN_NAME_EXPERIMENT_ID] == POOLED_RESULTS_EXPERIMENT_ID
            ].groupby('prior', sort=False), axs.flat):
        for (n, g), c, y in zip(group.groupby('patch_length'),
                                COLORS, y_vals.T[-1][::-1]):
            ax.axvline(g[COLUMN_NAME_EFFECT_SIZE].values[-1], c=c,
                       ls='--', label=None, linewidth=3)
            ax.scatter(x=g[COLUMN_NAME_EFFECT_SIZE].values[-1], y=y, c=c,
                       marker='D', label=None,
                       s=scatter_sizes[-1])

    for (name, group), ax in zip(
            data.loc[
                data[COLUMN_NAME_EXPERIMENT_ID] != POOLED_RESULTS_EXPERIMENT_ID
            ].groupby(cn.COLUMN_NAME_PRIOR, sort=False), axs.flat):
        ax.yaxis.set_tick_params(rotation=30)

        ax.axvline(0, c='k')
        group = group.sort_values(cn.COLUMN_NAME_CYCLE_LENGTH, ascending=False)

        for (n, g), y, c in zip(
                group.groupby(cn.COLUMN_NAME_CYCLE_LENGTH, sort=False),
                y_vals[:, :-1], COLORS[::-1]):
            ax.errorbar(x=g[COLUMN_NAME_EFFECT_SIZE].values, y=y, c=c,
                        xerr=(g[['ci_low', 'ci_high']].apply(
                            lambda s: s - g[
                                COLUMN_NAME_EFFECT_SIZE])).abs().values.T,
                        ls='none', label=None)

            ax.scatter(x=g[COLUMN_NAME_EFFECT_SIZE].values, y=y, c=c,
                       marker='s', label=n,
                       s=scatter_sizes[:-1])

            # For the pooled estimate, we need a seperate call to scatter as
            # matplotlib won't allow multiple markers

        if name:  # prior is True
            s = (
                '$\mathbf{Context_{3}}$ - $\mathbf{Context_{0}}$ given $\mathbf{Prior_{1}}$')
        else:
            s = (
                '$\mathbf{Context_{3}}$ - $\mathbf{Context_{0}}$ given $\mathbf{Prior_{0}}$')
        ax.set_title(s,  # f'{"" if name else "No "}Feedback on trial N-1',
                     weight='bold')

        ax.set_xlabel("Cohen's d Effect Size (95%-CI)", weight='bold')
        ax.set(yticks=[], ylabel='')

    leg = axs.flat[0].legend()

    handles, labels = axs.flat[0].get_legend_handles_labels()
    leg = axs.flat[0].legend(handles[::-1], labels[::-1],
                             borderaxespad=0., facecolor='none',
                             # fontsize=LEGEND_ENTRY_FONTSIZE,
                             columnspacing=1,
                             labelspacing=0.25,
                             title='Cycle Duration',
                             framealpha=0.5,
                             loc='upper right'
                             )

    # Need to invert the order of colors as the legend handles are sorted as
    # strings, so '5' comes after '10'
    for t, c in zip(leg.get_texts(), COLORS):
        t.set_color(c)
    leg.get_title().set_weight('bold')


def prep_data_for_meta_analysis(method: str) -> pd.DataFrame:
    data = load_data_for_meta_analysis_plot(method)

    data[COLUMN_NAME_PAPER_ID] = data[COLUMN_NAME_PAPER_ID].map(
        {'revisited': 'Current',
         'relevance': 'Hemed & Eitam (2022)'})

    meta_rows = data.iloc[-4:, :].copy()
    meta_rows.iloc[:, 3:] = np.nan
    meta_rows[COLUMN_NAME_EXPERIMENT_ID] = POOLED_RESULTS_EXPERIMENT_ID
    meta_rows[COLUMN_NAME_PAPER_ID] = 'Meta Analysis'

    meta_rows[['ci_low', 'ci_high']] = np.repeat(
        meta_rows[COLUMN_NAME_EFFECT_SIZE].values, 2).reshape(-1, 2)

    meta_rows['n'] = data.groupby(
        [cn.COLUMN_NAME_CYCLE_LENGTH, cn.COLUMN_NAME_PRIOR, ])['n'].sum().values

    data[COLUMN_NAME_EXPERIMENT_ID] = (
        data[['paper', COLUMN_NAME_EXPERIMENT_ID]].apply(
            lambda s: s[0].title() + " " + s[1].title()[1:], axis=1))

    data = pd.concat([data, meta_rows], ignore_index=True)

    data = calc_frequency_weights(data)
    data = calc_inverse_variance_weights(data)

    data.loc[data[
                 COLUMN_NAME_EXPERIMENT_ID] == POOLED_RESULTS_EXPERIMENT_ID, COLUMN_NAME_EFFECT_SIZE] = (
        data.loc[data[
                     COLUMN_NAME_EXPERIMENT_ID] != POOLED_RESULTS_EXPERIMENT_ID].groupby(
            SUMMARY_ROW_GROUP_ORDER).apply(
            lambda s: np.average(s[COLUMN_NAME_EFFECT_SIZE],
                                 weights=s[
                                     COLUMN_NAME_STANDARDIZED_INVERSE_VARIANCE_WEIGHTS], ))
    ).values

    data = get_p_values(data)

    data = data.sort_values([cn.COLUMN_NAME_PRIOR, COLUMN_NAME_PAPER_ID,
                             COLUMN_NAME_EXPERIMENT_ID,
                             cn.COLUMN_NAME_CYCLE_LENGTH],
                            ascending=[False, True, True, True])

    return data


def calc_frequency_weights(data):
    # Raw weight is SQRT(1/N)
    data.loc[data[
                 COLUMN_NAME_EXPERIMENT_ID] != POOLED_RESULTS_EXPERIMENT_ID, COLUMN_NAME_RAW_FREQUENCY_WEIGHT] = np.sqrt(
        data.loc[data[
                     COLUMN_NAME_EXPERIMENT_ID] != POOLED_RESULTS_EXPERIMENT_ID, 'n'])

    # Summing raw weights is the square root of the sum of squared weights
    data.loc[data[
                 COLUMN_NAME_EXPERIMENT_ID] == POOLED_RESULTS_EXPERIMENT_ID, COLUMN_NAME_RAW_FREQUENCY_WEIGHT] = (
        data.loc[data[
                     COLUMN_NAME_EXPERIMENT_ID] != POOLED_RESULTS_EXPERIMENT_ID].groupby(
            [cn.COLUMN_NAME_CYCLE_LENGTH,
             cn.COLUMN_NAME_PRIOR]
        )[
            COLUMN_NAME_RAW_FREQUENCY_WEIGHT].apply(
            lambda s: np.sqrt(np.sum(np.power(s, 2))))
    ).values

    data.loc[
        data[
            COLUMN_NAME_EXPERIMENT_ID] != POOLED_RESULTS_EXPERIMENT_ID, 'COLUMN_NAME_STANDARDIZED_FREQUENCY_WEIGHT'] = (
        data.loc[data[
                     COLUMN_NAME_EXPERIMENT_ID] != POOLED_RESULTS_EXPERIMENT_ID].groupby(
            SUMMARY_ROW_GROUP_ORDER)[
            COLUMN_NAME_RAW_FREQUENCY_WEIGHT].transform(
            lambda s: 100 * s / s.sum()
        )
    )

    data['COLUMN_NAME_STANDARDIZED_FREQUENCY_WEIGHT'] = data[
        'COLUMN_NAME_STANDARDIZED_FREQUENCY_WEIGHT'].fillna(100)

    return data


def calc_inverse_variance_weights(data):
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5287121/
    data[COLUMN_NAME_RAW_INVERSE_VARIANCE_WEIGHTS] = 1 / np.power(
        data['delta_sd'], 2)

    data.loc[
        data[
            COLUMN_NAME_EXPERIMENT_ID] != POOLED_RESULTS_EXPERIMENT_ID, COLUMN_NAME_STANDARDIZED_INVERSE_VARIANCE_WEIGHTS] = (
        data.loc[data[
                     COLUMN_NAME_EXPERIMENT_ID] != POOLED_RESULTS_EXPERIMENT_ID].groupby(
            SUMMARY_ROW_GROUP_ORDER)[
            COLUMN_NAME_RAW_INVERSE_VARIANCE_WEIGHTS].transform(
            lambda s: 100 * s / s.sum()
        )
    )
    data[COLUMN_NAME_STANDARDIZED_INVERSE_VARIANCE_WEIGHTS] = data[
        COLUMN_NAME_STANDARDIZED_INVERSE_VARIANCE_WEIGHTS].fillna(100)

    return data


def get_p_values(data):
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5287121/
    data.loc[data[
                 COLUMN_NAME_EXPERIMENT_ID] == POOLED_RESULTS_EXPERIMENT_ID, 'delta_inverse_variance_weighted_z_score'] = \
        data.loc[data[
                     COLUMN_NAME_EXPERIMENT_ID] != POOLED_RESULTS_EXPERIMENT_ID].groupby(
            SUMMARY_ROW_GROUP_ORDER).apply(
            lambda s: s[COLUMN_NAME_EFFECT_SIZE].mean() / np.sqrt(
                (s[COLUMN_NAME_RAW_INVERSE_VARIANCE_WEIGHTS].sum())
            )
        ).values

    data.loc[data[
                 COLUMN_NAME_EXPERIMENT_ID] == POOLED_RESULTS_EXPERIMENT_ID, 'p_value'] = 2 * sp.stats.norm.cdf(
        - np.abs(data.loc[data[
                              COLUMN_NAME_EXPERIMENT_ID] == POOLED_RESULTS_EXPERIMENT_ID, 'delta_inverse_variance_weighted_z_score']))

    return data


def draw_table(data: pd.DataFrame, fig: plt.Figure, axs: npt.ArrayLike) -> None:
    """

    References
    ----------
    https://stackoverflow.com/a/57169705/8522898
    """
    # axs = fig.subplots(2, 1, sharex=True)

    t = 0.05
    b = 0.125

    table_to_figure_width = 7 / 10
    left_margin = table_to_figure_width

    for (name, group), ax in zip(data.groupby('prior', sort=False), axs.flat):
        n = group.shape[0]

        fig.subplots_adjust(left=left_margin,
                            top=1 - t - (1 - t - b) / (n + 1), wspace=0)

        ax.margins(y=1 / 2 / n)

        group.loc[
            group[COLUMN_NAME_PAPER_ID].eq(group[COLUMN_NAME_PAPER_ID].shift()),
            COLUMN_NAME_PAPER_ID] = ''

        paper_labels = group[COLUMN_NAME_PAPER_ID].values

        experiment_labels = [x.split(' ')[-1] if not i % 2 else '' for i, x in
                             enumerate(group[COLUMN_NAME_EXPERIMENT_ID].values)]

        ci_values = group[['ci_low', 'ci_high']].apply(
            lambda s: f'[{s[0]:.2f} {s[1]:.2f}]', axis=1).values
        ci_values[-2:] = ''  # For the pooled results

        cohen_vals = [f"{i:.2f}" if np.abs(i) > 0.01 else '<|.01|'
                      for i in group[COLUMN_NAME_EFFECT_SIZE].values]

        weights = [
            f"{i:.2f}".zfill(5) for i in
            group[COLUMN_NAME_STANDARDIZED_INVERSE_VARIANCE_WEIGHTS].round(
                2).values
        ]

        p_values = [
            f'{i:.3f}' if i >= .001 else '<.001' for i in
            group['p_value'].round(4).values]

        cell_vals = [paper_labels, experiment_labels,
                     group['n'].fillna('').astype(str).str.replace(".0", "",
                                                                   regex=False),
                     weights, cohen_vals, ci_values, p_values]

        cell_vals = np.vstack([np.array(i).reshape(1, -1) for i in cell_vals])

        table = ax.table(cellText=cell_vals.T,
                         colLabels=META_ANALYSIS_TABLE_COLUMNS,
                         bbox=(-(left_margin + table_to_figure_width), 0.0,
                               left_margin + table_to_figure_width,
                               (n + 1) / n),
                         cellLoc='center')

        # Colorize specific columns
        [[table[(i, j)].get_text().set_color(c) for i, c in
          zip(range(1, n + 1),
              itertools.cycle(COLORS))] for j in range(2, 7)]

        # Set column headers to bold
        [[table[(j, i)].get_text().set_weight('bold') for i in range(7)]
         for j in (0, cell_vals.shape[1], cell_vals.shape[1] - 1)
         ]

        for i, child in enumerate(table.get_children()):
            child.set(linewidth=0)

        table.auto_set_font_size(False)
        table.set_fontsize(16)

        table.auto_set_column_width(col=range(7))

    fig.canvas.draw()  # need to draw the figure twice
