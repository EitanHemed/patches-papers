import logging

import numpy as np
import pandas as pd
import robusta as rst
from numpy import typing as npt
from rpy2.rinterface_lib.callbacks import logger as rpy2_logger
from rpy2.rinterface_lib.embedded import RRuntimeError

rpy2_logger.setLevel(logging.ERROR)

from . import constants as cn

SE_TO_95_CI = 1.96

ANOVA_KWARG_KEYS = ('subject', 'between', 'within', 'dependent', 'data',
                    )


def run_t_test(x, y, hypothesis, paired, test_type):
    return _select_test(test_type)(x, y, hypothesis, paired)


def _select_test(test_type):
    if test_type == cn.TEST_KEYS_FREQ:
        return _run_freq_t_test
    if test_type == cn.TEST_KEYS_BAYES:
        return _run_bayes_t_test
    if test_type == cn.T_TEST_KEYS_SEQUENTIAL_BAYES:
        return _run_sequence_of_bayes_t_tests


def _run_freq_t_test(x: npt.ArrayLike,
                     y: npt.ArrayLike, hypothesis: str,
                     paired: bool) -> rst.groupwise.T2Samples:
    return rst.groupwise.T2Samples(
        x=x, y=y, tail=hypothesis, paired=paired)


def _run_bayes_t_test(x: npt.ArrayLike,
                      y: npt.ArrayLike, hypothesis: str,
                      paired: bool) -> rst.groupwise.BayesT2Samples:
    return rst.groupwise.BayesT2Samples(
        x=x, y=y, tail=hypothesis, paired=paired)


def _run_sequence_of_bayes_t_tests(x: npt.ArrayLike,
                                   y: npt.ArrayLike, hypothesis: str,
                                   paired: bool):
    x = np.array(x)
    y = np.array(y)

    assert x.size == y.size

    vals = np.empty(x.size)

    m = rst.groupwise.BayesT2Samples(x=x, y=y, tail=hypothesis,
                                     paired=paired,
                                     fit=False)

    for i in range(x.size):
        m.reset(x=x[:i + 1], y=y[:i + 1], data=None, refit=False)
        try:
            m.fit()
            bf = m.report_table().iloc[0]['bf']
        except RRuntimeError as e:
            if "not enough observations" in str(e):
                bf = np.nan
            else:
                raise RuntimeError
        vals[i] = bf
    return vals


def calc_ci(a):
    return SE_TO_95_CI * np.std(a) / np.sqrt(a.size)


def run_anova(df: pd.DataFrame, multi_feedback_type_experiment: bool) -> dict:
    cols = [cn.COLUMN_NAME_PRIOR, cn.COLUMN_NAME_CONTEXT,
            cn.COLUMN_NAME_CYCLE_LENGTH]

    df = df.assign(
        **dict(zip(cols, df[cols].astype(str).values.T))
    )

    between_group_variable_column_names = (
        [cn.COLUMN_NAME_CYCLE_LENGTH, cn.COLUMN_NAME_FEEDBACK_TYPE]
        if multi_feedback_type_experiment else [cn.COLUMN_NAME_CYCLE_LENGTH])

    anova_args = dict(zip(ANOVA_KWARG_KEYS,
                          (cn.COLUMN_NAME_UID,
                           between_group_variable_column_names,
                           [cn.COLUMN_NAME_PRIOR, cn.COLUMN_NAME_CONTEXT],
                           cn.COLUMN_NAME_RESP_TIME, df)))

    return {
            cn.TEST_KEYS_FREQ: rst.groupwise.Anova(**anova_args), }
