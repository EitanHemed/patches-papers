""""
This module contains the functions to calculate 'Prior' and 'Context, two
running sum functions which are used to bin trials, analyzed in a factorial
design.

For more details about the two different method to calculate Prior and Context,
see papers [1-2].

1. Hemed, E., Bakbani-Elkayam, S., Teodorescu, A. R., Yona, L., & Eitam, B.
         (2020). Evaluation of an action’s effectiveness by the motor system in
         a dynamic environment. Journal of Experimental Psychology: General,
         149(5), 935.
2. Hemed, E., Bakbani Elkayam, S., Teodorescu, A., Yona, L., & Eitam, B.
        (2022). Evaluation of an Action’s Effectiveness by the Motor System in a
        Dynamic Environment: Amended.
"""

import numpy as np
import pandas as pd

from po_utils import constants as cn


def prep_for_dynamics_model(df: pd.DataFrame, method: int) -> pd.DataFrame:
    df = df.copy()

    if method == cn.PREPROCESSING_METHOD_AMENDED:
        _create_prior_column = _create_prior_column_amended
        _create_context_column = _create_context_column_amended
        _select_trials = _select_trials_for_dynamics_model_amended

    elif method == cn.PREPROCESSING_METHOD_CONFOUNDED:
        _create_prior_column = _create_prior_column_confounded
        _create_context_column = _create_context_column_confounded
        _select_trials = _select_trials_for_dynamics_model_confounded

    else:
        raise ValueError(
            f"Method {method} not in allowed methods (["
            f"{cn.PREPROCESSING_METHOD_CONFOUNDED}",
            f"{cn.PREPROCESSING_METHOD_AMENDED}])")

    df.sort_values([cn.COLUMN_NAME_UID, cn.COLUMN_NAME_TRIAL_NUMBER],
                   inplace=True)

    df = _create_prior_column(df)
    df = _create_context_column(df)
    df = _select_trials(df)[cn.DYNAMICS_MODEL_COLUMNS_OF_INTEREST]

    return df


def _create_prior_column_amended(df: pd.DataFrame) -> pd.DataFrame:
    """Create prior, the sum of the number of feedback occurrences on trial N-1
    {0, 1}.

    This method treats the series of task trials with correct responses only,
    and skips probed trials or trials with incorrect response or omissions.

    Returns
    -------
    df: pd.DataFrame
        A copy of the original dataframe, with the `Prior` column.
    """

    df = df.copy()

    df.loc[
        df[cn.COLUMN_NAME_RSP_FEEDBACK].notna(), cn.COLUMN_NAME_PRIOR] = (
        df.loc[df[cn.COLUMN_NAME_RSP_FEEDBACK].notna()].groupby(
            cn.COLUMN_NAME_UID)[
            cn.COLUMN_NAME_RSP_FEEDBACK].shift().values)

    df.loc[df[cn.COLUMN_NAME_PRIOR].notna(), cn.COLUMN_NAME_PRIOR] = (
        df.loc[
            df[cn.COLUMN_NAME_PRIOR].notna(), cn.COLUMN_NAME_PRIOR].astype(
            int)).values

    return df


def _create_context_column_amended(df: pd.DataFrame) -> pd.DataFrame:
    """Create the Context values, the sum of the number of feedback occurrences
    on trials N-4 through N-2.

    This method treats the series of task trials with correct responses only,
    and skips probed trials or trials with incorrect response or omissions.
    Create the `Context` values.

    Parameters
    ----------
    df: pd.DataFrame

    Returns
    -------
    df: pd.DataFrame
        A copy of the original dataframe, with the `Context` column.
    """

    df = df.copy()

    df.loc[
        df[cn.COLUMN_NAME_RSP_FEEDBACK].notna(), cn.COLUMN_NAME_CONTEXT] = (
        df.loc[df[cn.COLUMN_NAME_RSP_FEEDBACK].notna()].groupby(
            cn.COLUMN_NAME_UID)[
            cn.COLUMN_NAME_RSP_FEEDBACK].shift(2).rolling(3).sum().values)

    return df


def _create_prior_column_confounded(df: pd.DataFrame) -> pd.DataFrame:
    """Code effect trials and "prior" (effect on n-1) This gives you
    the confounded stata coding of prior and effect (nb, some post-processing
    takes place later).

    Effect is 0 (no-effect) unless this is an effect patch and this was not a
    correct response, and no probed trial.

    This is the original Stata code:

    1. First, label trials as 1 (n-1 is "effect") if they are trials which
    follow a task-trial on effect patch, with a correct response.

    ```
    by part, sort: gen prior = 0
    replace prior = 1 if ///
        effect_patch[_n-1] == 1 & output_correct[_n-1] == 1 ///
        & (trial_num[_n-1] + 1 == trial_n & probed_trial[_n-1] != 1)
    ```
    """

    df = df.copy()

    # First label all trials as 0 (i.e., no prior feedback)
    df[cn.COLUMN_NAME_PRIOR] = 0

    # Then label trials as 1 (i.e., prior feedback) if they are trials which
    # follow a task-trial on effect patch, with a correct response and belong
    # to the same participant.

    df.loc[
        # the preceding trial was not a probed trial
        (df[cn.COLUMN_NAME_PROBED_TRIAL].shift(1) != 1)
        # Where action-effects are available
        & (df[cn.COLUMN_NAME_FEEDBACK_CYCLE].astype(bool).shift(1))
        # And the previous response is correct
        & (df[cn.COLUMN_NAME_RESP_ACCURACY].shift(1) == 1)
        # And this trial does not belong to the next participant
        & (df[cn.COLUMN_NAME_UID] == df[cn.COLUMN_NAME_UID]),
        cn.COLUMN_NAME_PRIOR] = 1

    return df


def _create_context_column_confounded(df: pd.DataFrame) -> pd.DataFrame:
    """
    Next we aggregate `prior` as the sum of 3 most recent trials and assign
    it to `context`. In Stata we used:

    ```
    gen counter_`num' = 0 // //generate a specific counter
    forvalues i = 1(1)`num' { //for the length of the current counter, usually 3
        by part, sort: replace counter_`num' = ///
        counter_`num' + prior[_n-(`i'-1)] if probed_trial == 0
                            //accumulate the priors
    }
    ```
    """

    df = df.copy()

    # By default, create a context column with all NaNs for probed trials and 0
    # for non-probed trials.
    df[cn.COLUMN_NAME_CONTEXT] = np.where(
        df[cn.COLUMN_NAME_PROBED_TRIAL], np.nan, 0)

    # On task trials (PROBED_TRIAL = 0)
    df.loc[df[cn.COLUMN_NAME_PROBED_TRIAL] != 1, cn.COLUMN_NAME_CONTEXT] = (
        # On task trials (PROBED_TRIAL = 0)
        df.loc[df[cn.COLUMN_NAME_PROBED_TRIAL] != 1].groupby(
            cn.COLUMN_NAME_UID)[
            # Aggregate 'prior' values in a 3 trials running window
            cn.COLUMN_NAME_PRIOR].rolling(3).sum().values)

    df.loc[df[cn.COLUMN_NAME_PROBED_TRIAL], cn.COLUMN_NAME_CONTEXT] = np.nan

    return df


def _select_trials_for_dynamics_model_confounded(
        input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Next we remove probed trials, or trials that have an undefined `context`
    or `prior` value.
    ```
    // remove trials with no counter/prior values or probed trials
    drop if counter_3 == . | prior == . | probed_trial == 1
    ```
    """

    df = input_df.loc[
        # Negate the following selection
        ~(
            # Rows where either prior or context are missing
                (pd.isna(
                    input_df[[cn.COLUMN_NAME_PRIOR, cn.COLUMN_NAME_CONTEXT]]
                ).any(axis=1))
                # Or rows of non-probed trials
                | (input_df[cn.COLUMN_NAME_PROBED_TRIAL]))
    ]

    # Currently, (as reviewer #2 in the 2020 paper noted), `context` represents
    # sum of action-effects on trials N-3 through N-1. Therefore, we slide
    # `context` one step forward, to represent N-4 through N-2 for trial N.
    # The following is the matching code from Stata.
    # ```
    # sort part trial_num
    # by part , sort: gen temp = counter_3[_n-1] if [_n-1] != .
    # gen temp2 = temp
    # by part, sort: replace counter_3 = temp if counter_3[_n-1] != .
    # drop temp temp2
    # ```

    # Shift context values down to reflect N-4 through N-2
    df.loc[~pd.isna(df[cn.COLUMN_NAME_CONTEXT]), cn.COLUMN_NAME_CONTEXT] = (
        # On rows where context is defined
        df.loc[~pd.isna(df[cn.COLUMN_NAME_CONTEXT])].groupby(
            cn.COLUMN_NAME_UID)[
            cn.COLUMN_NAME_CONTEXT].shift().values)

    return df


def _select_trials_for_dynamics_model_amended(
        df: pd.DataFrame) -> pd.DataFrame:
    """
    Select trials for dynamics model.

    The selected trials are task-trials (i.e., not probed trials), which do not
    follow a probed trial, and have a correct response.

    Parameters
    ----------
    df: pd.DataFrame

    Returns
    -------
    pd.DataFrame

    """
    return df.loc[(df[cn.COLUMN_NAME_RESP_ACCURACY].eq(1)) & (
        df[cn.COLUMN_NAME_RESP_ACCURACY].shift().eq(1))
                  & (df[cn.COLUMN_NAME_PROBED_TRIAL].shift().eq(False)) & (
                      ~df[cn.COLUMN_NAME_PROBED_TRIAL])
                  & (df[[cn.COLUMN_NAME_PRIOR,
                         cn.COLUMN_NAME_CONTEXT]].notna().any(axis=1))]


def aggregate_for_dynamics_model(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(
            [cn.COLUMN_NAME_DATE, cn.COLUMN_NAME_CYCLE_LENGTH,
             cn.COLUMN_NAME_FEEDBACK_TYPE,
             cn.COLUMN_NAME_UID, cn.COLUMN_NAME_PRIOR,
             cn.COLUMN_NAME_CONTEXT, cn.COLUMN_NAME_FEEDBACK_MANIPULATION_TYPE],
            as_index=False
        )[cn.COLUMN_NAME_RESP_TIME].agg(cn.DYNAMICS_MODEL_AGG_FUNC)
    )
