import re
import typing

import numpy as np
import pandas as pd
import robusta as rst

from . import constants as cn

FNAME_EXT = '.txt'
OUTPUT_PATH = 'Output/{method}/{hashed}/Texts/'
ANOVA_PATH = f'{OUTPUT_PATH}anovas{FNAME_EXT}'
T_TESTS_REPORT_PATH = f'{OUTPUT_PATH}t_tests{FNAME_EXT}'

SIGNIFICANCE_LABELS = ('significant', 'non-significant')
SIGNIFICANT_CHANGE_LABELS = ('decrease', 'increase')
NON_SIGNIFICANT_CHANGE_LABEL = "change"

DESCRIPTIVES_COLUMNS_NAMES = ['N', 'Mean', 'Median', 'SD']
DECSRIPTIVES_AGGREGATION_FUNCTIONS = [pd.Series.count, 'mean', 'median', np.std,
                                      ]
DESCRIPTIVES_PATH = f'{OUTPUT_PATH}dynamics_model_descriptives.csv'


def save_analyses_text(ts_dict: typing.Dict, anova_dict: typing.Dict,
                       df: pd.DataFrame,
                       is_multi_feedback_type_experiment: bool,
                       dynamics_model_coding_method: str) -> None:
    save_descriptives(
        df, path=DESCRIPTIVES_PATH.format(method=dynamics_model_coding_method,
                                          hashed=cn.HASHED_SCREENING_PARAMS))
    save_anova(
        anova_dict, path=ANOVA_PATH.format(method=dynamics_model_coding_method,
                                           hashed=cn.HASHED_SCREENING_PARAMS))
    save_t_tests(
        ts_dict,
        is_multi_feedback_type_experiment=is_multi_feedback_type_experiment,
        path=T_TESTS_REPORT_PATH.format(method=dynamics_model_coding_method,
                                        hashed=cn.HASHED_SCREENING_PARAMS))


def save_descriptives(df: pd.DataFrame, path: str) -> None:
    agg_df = df.groupby(
        [cn.COLUMN_NAME_CYCLE_LENGTH, cn.COLUMN_NAME_FEEDBACK_TYPE,
         cn.COLUMN_NAME_PRIOR,
         cn.COLUMN_NAME_CONTEXT])[cn.COLUMN_NAME_RESP_TIME].agg(
        DECSRIPTIVES_AGGREGATION_FUNCTIONS
    )

    agg_df = agg_df.round(2)

    agg_df['count'] = agg_df['count'].round(0).values
    agg_df = agg_df.rename(dict(zip(agg_df.columns,
                                    DESCRIPTIVES_COLUMNS_NAMES)))

    agg_df.to_csv(path)


def save_anova(anovas_dict: typing.Dict, path: str) -> None:
    freq_anova_terms = list(filter(None,
                                   re.split(']. ', anovas_dict[
                                       cn.TEST_KEYS_FREQ].report_text())))

    terms = [f'{fa}]'.replace(
        'Partial Eta-Sq. = 0.00', 'Partial Eta-Sq. < 0.01') for fa in
        freq_anova_terms]

    with open(path, 'w') as f:
        print('\n'.join(terms), file=f)


def save_t_tests(ts_dict: typing.Dict, is_multi_feedback_type_experiment: bool,
                 path: str) -> None:
    if is_multi_feedback_type_experiment:
        feedback_types = cn.NEW_FEEDBACK_LEVELS
    else:
        feedback_types = cn.FEEDBACK_TYPE_SINGLE

    with open(path, 'w') as f:

        for patch_length in cn.CYCLE_LENGTHS:
            print(f'FOR CYCLE DURATION {patch_length}: ', file=f)
            for prior in cn.PRIOR_LEVELS:
                print(f'FOR PRIOR VALUE {bool(prior)}: ', file=f)
                for feedback_type in feedback_types:
                    group_name = (patch_length, prior, feedback_type)
                    freq_test = ts_dict[group_name][cn.TEST_KEYS_FREQ]
                    bayes_test = ts_dict[group_name][cn.TEST_KEYS_BAYES]
                    print(
                        interpret_t_result(
                            feedback_type if is_multi_feedback_type_experiment else '',
                            freq_test, bayes_test),
                        file=f)
                print('\n', file=f)
            print('\n\n', file=f)


def interpret_t_result(group_name: str, freq_test: rst.groupwise.T2Samples,
                       bayes_test: rst.groupwise.BayesT2Samples) -> str:
    """Construct a string which includes the frequentist and Bayesian t-test
    results, along with raw effect size and label of significance.
    """
    t_clause = form_t_clause(freq_test, bayes_test)

    delta = freq_test.x - freq_test.y

    test_diff_mean = delta.mean()
    test_diff_std = np.std(delta)
    is_test_significant = freq_test.report_table().iloc[0].to_dict(
    )['p-value'] < .05
    sig_str = np.where(
        is_test_significant,
        *SIGNIFICANCE_LABELS
    )
    change_str = np.where(test_diff_mean < 0, *SIGNIFICANT_CHANGE_LABELS
                          ) if is_test_significant else NON_SIGNIFICANT_CHANGE_LABEL

    change_units = f'{test_diff_mean:.2f} MS ({test_diff_std:.2f})'

    return (f"{group_name} a {sig_str}"
            f" mean {change_str} [{change_units}; {t_clause}]")


def form_t_clause(t_test: rst.groupwise.T2Samples,
                  bayes_t_test: rst.groupwise.BayesT2Samples) -> str:
    """Concat the results from the frequentist and Bayesian t-tests.
    """
    return (
        f'{t_test.report_text(effect_size=True)},'
        f" BF1:0 = {(bayes_t_test.report_table().iloc[0].to_dict()['bf']):.4f}")
