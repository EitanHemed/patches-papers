import itertools
import os
import pdb
import re
import typing

import numpy as np
import pandas as pd
import scipy as sp
from matplotlib import pyplot as plt
from po_utils import constants as cn
from po_utils import plotting
from scipy.stats import pearsonr

FIGSIZE = (8.5, 11)

COLORS = plotting.COLORS[:2]

PLOT_SUBPLOTS_TUPLE = (3, 1)

POOLED_RESULTS_EXPERIMENT_ID = ' '
COLUMN_NAME_EXPERIMENT_ID = 'experiment'
COLUMN_NAME_PAPER_ID = 'paper'
COLUMN_NAME_EFFECT_SIZE = 'cohen'

AX_TITLES = ['Probe-less experiments', 'Probe experiments (Amended preprocessing)',
             'Probe experiments (Confounded preprocessing)']

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

META_ANALYSIS_TABLE_COLUMNS = ['Paper - Exp. #', 'N',
                               'Wt. (%)',
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
    os.makedirs(f'meta_analysis/{cn.HASHED_SCREENING_PARAMS}',
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


def run_meta_analyses() -> typing.Tuple[pd.DataFrame]:
    # Load raw
    confounded = load_data_for_meta_analysis_plot(cn.PREPROCESSING_METHOD_CONFOUNDED)
    amended = load_data_for_meta_analysis_plot(cn.PREPROCESSING_METHOD_AMENDED)
    # Select relevant experiments for meta-analysis

    amended_probeless_experiments = amended.loc[(amended[COLUMN_NAME_PAPER_ID] == 'revisited').values & (amended[
                                                                                                             COLUMN_NAME_EXPERIMENT_ID].str.extract(
        '(\d+)').astype(int) < 3).values[:, 0], :]
    amended_probed_experiments = amended[~amended.index.isin(amended_probeless_experiments.index)]
    confounded = confounded.loc[~amended.index.isin(amended_probeless_experiments.index)]
    # Prepare data for meta-analysis
    confounded = prep_data_for_meta_analysis(confounded)
    amended_probeless_experiments = prep_data_for_meta_analysis(amended_probeless_experiments)
    amended_probed_experiments = prep_data_for_meta_analysis(amended_probed_experiments)

    return amended_probeless_experiments, amended_probed_experiments, confounded


def visualize_meta_analyses(data: typing.Tuple[pd.DataFrame]) -> None:
    for i in cn.PRIOR_LEVELS:

        _data = [d.loc[d[cn.COLUMN_NAME_PRIOR] == i] for d in data]

        fig, axs = plt.subplots(
            *PLOT_SUBPLOTS_TUPLE, figsize=FIGSIZE, sharex='col',
            # Do this by the number of experiments on each dataframe
            gridspec_kw={'height_ratios': [i.shape[0] + 0.1 for
                                           i in _data],
                         #  'width_ratios': [2, 1]
                         })
        for df, ax, title in zip(_data, axs, AX_TITLES):
            draw_forest_plot(df, ax, title)
            draw_table(df, fig, ax)

        fig.tight_layout()

        fig.savefig(
            f'meta_analysis/{cn.HASHED_SCREENING_PARAMS}/{i}_forest_plot.jpg',
            dpi=300, bbox_inches='tight')


def draw_forest_plot(data, ax, title) -> None:
    if ax == ax.figure.axes[-1]:
        ax.set_xlabel("Cohen's d Effect Size (95%-CI)", weight='bold')

    ax.set_title(title, weight='bold', size=14)

    n_studies = data.groupby([COLUMN_NAME_EXPERIMENT_ID,
                              COLUMN_NAME_PAPER_ID]).size().shape[0]

    y_locs = np.linspace(0, n_studies * 2 - 1, n_studies * 2)
    y_vals = np.fliplr(y_locs.reshape(n_studies, 2).T)

    for (n, g), c, y in zip(data
                                    # .loc[data[COLUMN_NAME_EXPERIMENT_ID] == POOLED_RESULTS_EXPERIMENT_ID]
                                    .groupby('patch_length'),
                            COLORS, y_vals[::-1]  # [-1][::-1],
                            ):
        ax.axvline(g[COLUMN_NAME_EFFECT_SIZE].values[-1], c=c,
                   ls='--', label=None, linewidth=1)
        ax.scatter(x=g[COLUMN_NAME_EFFECT_SIZE], y=y, c=c,
                   marker='s', label=None,
                   s=g[COLUMN_NAME_STANDARDIZED_INVERSE_VARIANCE_WEIGHTS].div(100).values * plt.rcParams[
                       'lines.markersize'] * 25
                   )

        ax.errorbar(x=g[COLUMN_NAME_EFFECT_SIZE].values, y=y, c=c,
                    xerr=(g[['ci_low', 'ci_high']].apply(
                        lambda s: s - g[
                            COLUMN_NAME_EFFECT_SIZE])).abs().values.T,
                    ls='none', label=n)

    _data = data.loc[data[COLUMN_NAME_EXPERIMENT_ID] != POOLED_RESULTS_EXPERIMENT_ID]

    ax.yaxis.set_tick_params(rotation=30)

    ax.axvline(0, c='k')

    ax.grid(linestyle='-', alpha=0.5)

    ax.set(ylabel='', yticks=sorted(y_vals.flatten()), yticklabels=[], xlim=(-1.5, 1.5),
           xticks=np.linspace(-1.5, 1.5, 7),
           )
    ax.yaxis.set_tick_params(width=0)

    if ax == ax.figure.axes[0]:

        # leg = ax.legend()
        handles, labels = ax.get_legend_handles_labels()

        leg = ax.legend(handles, labels,
                        borderaxespad=0., facecolor='none',
                        # fontsize=LEGEND_ENTRY_FONTSIZE,
                        columnspacing=1,
                        labelspacing=0.25,
                        title='Cycle Duration',
                        framealpha=0,
                        # loc='upper right'
                        )
        # Need to invert the order of colors as the legend handles are sorted as
        # strings, so '5' comes after '10'
        for t, c in zip(leg.get_texts(), COLORS):
            t.set_color(c)
        leg.get_title().set_weight('bold')


def prep_data_for_meta_analysis(data) -> pd.DataFrame:
    data[COLUMN_NAME_PAPER_ID] = data[COLUMN_NAME_PAPER_ID].map(
        {'revisited': 'Current',
         'relevance': 'Hemed & Eitam (2022)'})

    # Expand the data to include the meta-analysis rows
    meta_rows = data.iloc[-4:, :].copy()
    # Expand the data to include the meta-analysis relevant columns
    meta_rows.iloc[:, 3:] = np.nan
    meta_rows[COLUMN_NAME_EXPERIMENT_ID] = POOLED_RESULTS_EXPERIMENT_ID
    meta_rows[COLUMN_NAME_PAPER_ID] = 'Pooled'

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

    ci_width = (data.loc[data[COLUMN_NAME_EXPERIMENT_ID] != POOLED_RESULTS_EXPERIMENT_ID
                         ].groupby([cn.COLUMN_NAME_CYCLE_LENGTH, cn.COLUMN_NAME_PRIOR
                                    ])[COLUMN_NAME_RAW_INVERSE_VARIANCE_WEIGHTS].sum() ** 0.5 * 1.96).values

    data.loc[data[COLUMN_NAME_EXPERIMENT_ID] == POOLED_RESULTS_EXPERIMENT_ID, 'ci_low'] = (
            data.loc[data[
                         COLUMN_NAME_EXPERIMENT_ID] == POOLED_RESULTS_EXPERIMENT_ID, COLUMN_NAME_EFFECT_SIZE] - ci_width
    ).values

    data.loc[data[COLUMN_NAME_EXPERIMENT_ID] == POOLED_RESULTS_EXPERIMENT_ID, 'ci_high'] = (
            data.loc[data[
                         COLUMN_NAME_EXPERIMENT_ID] == POOLED_RESULTS_EXPERIMENT_ID, COLUMN_NAME_EFFECT_SIZE] + ci_width
    ).values

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


def draw_table(data: pd.DataFrame, fig: plt.Figure, ax: plt.Axes) -> None:
    """

    References
    ----------
    https://stackoverflow.com/a/57169705/8522898
    """
    data = data.copy()
    t = 0.05
    b = 0.125

    table_to_figure_width = 6.5 / 10
    left_margin = table_to_figure_width

    n = data.shape[0]
    # ax.axis('off')

    fig.subplots_adjust(left=left_margin,
                        )
    ax.margins(y=1 / 2 / n)

    top_ax = ax == fig.axes[0]
    data.loc[
        data[COLUMN_NAME_PAPER_ID].eq(data[COLUMN_NAME_PAPER_ID].shift()),
        COLUMN_NAME_PAPER_ID] = ''

    paper_labels = data[COLUMN_NAME_PAPER_ID].values

    experiment_labels = [x.split(' ')[-1] if not i % 2 else '' for i, x in
                         enumerate(data[COLUMN_NAME_EXPERIMENT_ID].values)]

    lower_ci = data['ci_low'].apply(lambda s: f'{s:.2f}' if s < 0 else f'{s:.2f} ').values
    upper_ci = data['ci_high'].apply(lambda s: f'{s:.2f}' if s < 0 else f' {s:.2f}').values

    ci_values = [f'[{l} {u}]' for l, u in zip(lower_ci, upper_ci)]

    ci_values = ['' if 'nan' in i else i for i in ci_values]

    cohen_vals = [f"{i:.2f}" if np.abs(i) > 0.01 else '<|.01|'
                  for i in data[COLUMN_NAME_EFFECT_SIZE].values]
    max_len = max([len(i) for i in cohen_vals])
    cohen_vals = [i.rjust(max_len) for i in cohen_vals]

    weights = [
        f"{i:.2f}".zfill(5) for i in
        data[COLUMN_NAME_STANDARDIZED_INVERSE_VARIANCE_WEIGHTS].round(
            2).values
    ]

    p_values = [
        f'{i:.3f}' if i >= .001 else '<.001' for i in
        data['p_value'].round(4).values]

    # Modify paper and experiment labels
    maximum_col_width = len('Hemed & Eitam (2022) - EX')
    paper_and_experiment_labels = [
        f"{i} - E{j}" if j else i
        for i, j in zip(paper_labels, experiment_labels)
    ]

    cell_vals = [paper_and_experiment_labels,
                 data['n'].fillna('').astype(str).str.replace(".0", "",
                                                              regex=False),
                 weights, cohen_vals, ci_values, p_values]

    cell_vals = np.vstack([np.array(i).reshape(1, -1) for i in cell_vals])

    # Match the width of the first table to the rest
    if top_ax:
        cell_vals[0, -3] = ' ' * len('Hemed & Eitam (2022)')

    table = ax.table(cellText=cell_vals.T,
                     colLabels=META_ANALYSIS_TABLE_COLUMNS,
                     bbox=(-(left_margin + table_to_figure_width), 0.0,
                           left_margin + table_to_figure_width,
                           (n + 1) / n),
                     cellLoc='center',
                     colWidths=[0.2,  # Manuscript and experiment label
                                0.06,  # N
                                0.175,  # Weight
                                0.09,  # ES
                                0.2,  # CI
                                0.125],  # p-value

                     )

    # Colorize specific columns
    [[table[(i, j)].get_text().set_color(c) for i, c in
      zip(range(1, n + 1),
          itertools.cycle(COLORS))] for j in range(1, 6)]
    #
    # Set column headers to bold
    [[table[(j, i)].get_text().set_weight('bold') for i in range(6)]
     for j in (0, cell_vals.shape[1], cell_vals.shape[1] - 1)
     ]

    for i, child in enumerate(table.get_children()):
        child.set(linewidth=0)

    table.auto_set_font_size(False)
    table.set_fontsize(14)

    for i in range(0, cell_vals.shape[1]):
        table[i, 0]._loc = 'right'
        table[i, 0]._text.set_horizontalalignment('right')

    # Ugly hack to force the first column to have a fixed width on all tables
    if top_ax:
        text = table[cell_vals.shape[1], 0].get_text()
        text.set_text('Hemed & Eitam (2022) - E1')
        text.set_weight('normal')
        text.set_color('none')

    table.auto_set_column_width(False)  # col=range(1, 6))

    # # Create a horizontal line between the 2nd to last row and the third to last row, to separate the pooled results
    # # from the individual experiments. Do not add cells, but rather draw a line.
    # for i in range(1, 6):
    #     for row in range(n, n - 2, -1):
    #         table[(row, i)].set_edgecolor('black')
    #         table[(row, i)].set_linewidth(0.5)

    fig.canvas.draw()  # need to draw the figure twice

    if ax == ax.figure.axes[-1]:
        prior = data[cn.COLUMN_NAME_PRIOR].values[0]

        if prior:  # prior is True
            s = (
                '$\mathbf{Context_{3}}$ - $\mathbf{Context_{0}}$ $\mathbf{given}$ $\mathbf{Prior_{1}}$')
        else:
            s = (
                '$\mathbf{Context_{3}}$ - $\mathbf{Context_{0}}$ $\mathbf{given}$ $\mathbf{Prior_{0}}$')
        ax.figure.suptitle(s,  # f'{"" if name else "No "}Feedback on trial N-1',
                           weight='bold', fontsize=18, x=0.5, ha='right')
