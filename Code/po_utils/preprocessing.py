import glob
import itertools
import re
import typing
from functools import partial
from typing import Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd

from po_utils import constants as cn
from po_utils._dynamics import prep_for_dynamics_model, \
    aggregate_for_dynamics_model

__all__ = ('load_data', 'read_raw_pooled_data', 'clean_data')

OUTPUT_PATH = 'Output/{method}/{hashed}/Data/'
OUTPUT_DATA_CSV_PATH = OUTPUT_PATH + '{stage}.csv'
OUTPUT_PREPROCESSING_TXT_PATH = 'Output/{method}/{hashed_json}/preprocessing.txt'
OUTPUT_DEMOGRAPHICS_TXT_PATH = 'Output/demographics.txt'

INPUT_DATA_PATH = 'Input/e*.zip'

OUTPUT_DATA_FNAME_RAW = 'raw'
OUTPUT_DATA_FNAME_CLEAN = 'clean'
OUTPUT_DATA_FNAME_DYNAMICS_PRE_AGGREGATION = 'pre_aggregation'
OUTPUT_DATA_FNAME_DYNAMICS_POST_AGGREGATION = 'post_aggregation'
OUTPUT_DATA_FNAME_DYNAMICS_POST_AGGREGATION_WIDE = 'post_aggregation_wide'


def preprocess(method: int) -> pd.DataFrame:
    _load_data = partial(load_data, method=method)
    _clean_data = partial(clean_data, method=method)
    _prep_for_dynamics_model = partial(
        prep_for_dynamics_model, method=method)
    _aggregate_for_dynamics_model = partial(
        aggregate_for_dynamics_model)

    _pipe_data = partial(save_data_intermediate, method=method)

    raw_df = _pipe_data(_load_data(), stage=OUTPUT_DATA_FNAME_RAW)
    clean_df = _pipe_data(_clean_data(raw_df), stage=OUTPUT_DATA_FNAME_CLEAN)
    dynamics_df = _pipe_data(_prep_for_dynamics_model(clean_df),
                             stage=OUTPUT_DATA_FNAME_DYNAMICS_PRE_AGGREGATION)
    aggregated_dynamics_df = _pipe_data(
        _aggregate_for_dynamics_model(dynamics_df),
        stage=OUTPUT_DATA_FNAME_DYNAMICS_POST_AGGREGATION)

    _pipe_data(
        _pivot_for_external(aggregated_dynamics_df),
        stage=OUTPUT_DATA_FNAME_DYNAMICS_POST_AGGREGATION_WIDE)

    return aggregated_dynamics_df


def save_data_intermediate(df, method, stage, ) -> pd.DataFrame:
    """Save the current data and return it for further processing"""
    df.to_csv(OUTPUT_DATA_CSV_PATH.format(
        stage=stage, method=method, hashed=cn.HASHED_SCREENING_PARAMS))
    return df


def load_data(method) -> pd.DataFrame:
    return _mark_distance_from_probes(_sort_raw(_relabel_columns(
        _remove_columns_from_raw_data(_relabel_levels(
            read_raw_pooled_data()),
            cn.COLUMN_NAME_STUBS_TO_DROP_FROM_RAW_DATA))))


def _mark_distance_from_probes(df: pd.DataFrame) -> pd.DataFrame:
    df[cn.TRIALS_SINCE_LAST_PROBED_TRIAL] = df.groupby(
        [cn.COLUMN_NAME_UID,
         df[cn.COLUMN_NAME_PROBED_TRIAL].cumsum()]).cumcount().values
    # Remove data prior to the first probed trial
    df.loc[
        df[cn.COLUMN_NAME_TRIAL_NUMBER].le(6), cn.TRIALS_SINCE_LAST_PROBED_TRIAL] = np.nan

    return df


def _sort_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Returns the dataframe sorted by unique ID and trial number". """
    return df.assign(
        **{cn.COLUMN_NAME_DATE: _interpret_datetime(
            df[cn.COLUMN_NAME_DATE])}).sort_values(cn.COLUMN_NAME_DATE
                                                   ).sort_values(
        [cn.COLUMN_NAME_UID, cn.COLUMN_NAME_TRIAL_NUMBER])


def _interpret_datetime(s: pd.Series) -> pd.Series:
    """Casts the string-type date values to datetime. The date format could
    be one of several formats depending on the experiment.
    """

    # https://stackoverflow.com/a/70544788/8522898

    # On Pavlovia participants running around midnight are not listed as 00:mm
    # But 24:mm
    s = s.str.replace('24h', '00h').values

    for x in cn.DATE_FORMATS:
        s = pd.to_datetime(s, errors="ignore", format=f"{x}")

    if s.isna().sum() > 0:
        raise ValueError

    return s


def read_raw_pooled_data() -> pd.DataFrame:
    return pd.read_csv(_get_data_file_names()['task'])


def _relabel_levels(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(**{
        cn.COLUMN_NAME_FEEDBACK_TYPE: df[cn.COLUMN_NAME_FEEDBACK_TYPE].map(
            cn.UPDATE_FEEDBACK_LEVELS_DICT)
    })


def _relabel_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=cn.RAW_DATA_COLUMN_RENAME_DICT)


def _remove_columns_from_raw_data(df: pd.DataFrame,
                                  filter_stubs: str) -> pd.DataFrame:
    return df[df.columns.difference(
        df.filter(regex=filter_stubs).columns)]


def clean_data(df: pd.DataFrame, method: int) -> pd.DataFrame:
    df = _prep_to_clean_data(df)
    _report_preprocessing(df, method)
    save_demographics_report()
    return save_data_intermediate(
        _label_feedback_trials(_remove_invalid_data(df)),
        method, OUTPUT_DATA_FNAME_CLEAN
    )


def _remove_invalid_data(df: pd.DataFrame) -> pd.DataFrame:
    df.loc[(~df[cn.COLUMN_NAME_IS_TRIAL_VALID]),
           cn.COLUMN_NAME_RESP_TIME] = np.nan
    return df.loc[~df[cn.COLUMN_NAME_IS_POOR_PERFORMANCE_PARTICIPANT]]


def _prep_to_clean_data(df: pd.DataFrame) -> pd.DataFrame:
    return _create_accuracy_summary_columns(_label_valid_trials(
        _assign_resp_accuracy(_remove_columns_from_raw_data(
            df, cn.COLUMN_NAME_STUBS_TO_DROP_FROM_CLEAN_DATA))))


def _assign_resp_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(
        **{cn.COLUMN_NAME_IS_RESP_CORRECT: df[cn.COLUMN_NAME_RESP_ACCURACY].eq(
            1)})


def _label_valid_trials(df: pd.DataFrame) -> pd.DataFrame:
    acceptable_rt_task_trials = ~df[cn.COLUMN_NAME_PROBED_TRIAL] & (
        df[cn.COLUMN_NAME_RESP_TIME].between(
            cn.TASK_TRIAL_MIN_RT,
            df[cn.COLUMN_NAME_MAXIMAL_TASK_RT].max()))

    # Decided not to filter probed trials based on RT
    acceptable_rt_probed_trials = df[cn.COLUMN_NAME_PROBED_TRIAL] & (
        df[cn.COLUMN_NAME_RESP_TIME].between(
            cn.PROBED_TRIAL_MIN_RT,
            cn.PROBED_TRIAL_MAX_RT))

    return df.assign(
        **{cn.COLUMN_NAME_IS_TRIAL_VALID:
               (acceptable_rt_task_trials | acceptable_rt_probed_trials)
               & (df[cn.COLUMN_NAME_IS_RESP_CORRECT])
                   .values})


def _label_feedback_trials(df: pd.DataFrame) -> pd.DataFrame:
    # Find task trials with a correct response
    valid_task_trials = (~df[cn.COLUMN_NAME_PROBED_TRIAL]) & (
        df[cn.COLUMN_NAME_RESP_ACCURACY].eq(1))

    # Label valid trials (potential effect trials) based on the feedback cycle
    #  type they are in
    df.loc[:, cn.COLUMN_NAME_RSP_FEEDBACK] = df[cn.COLUMN_NAME_FEEDBACK_CYCLE].values
    df.loc[~valid_task_trials, cn.COLUMN_NAME_RSP_FEEDBACK] = np.nan

    return df


def _create_accuracy_summary_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df[cn.COLUMN_NAME_VALID_TRIALS_PROPORTION] = (
        df.groupby(cn.COLUMN_NAME_UID)[
            cn.COLUMN_NAME_IS_TRIAL_VALID].transform('mean').values)

    accuracy_vals = df.groupby(
        [cn.COLUMN_NAME_UID, cn.COLUMN_NAME_PROBED_TRIAL])[
        cn.COLUMN_NAME_IS_RESP_CORRECT].mean().reset_index(
    ).pivot(columns=cn.COLUMN_NAME_PROBED_TRIAL, index=cn.COLUMN_NAME_UID,
            values=cn.COLUMN_NAME_IS_RESP_CORRECT).to_dict()

    # For probe-less experiments we provide a dummy value instead of
    #  calculating performance on probed trials.
    if True not in accuracy_vals.keys():
        accuracy_vals[True] = dict(itertools.zip_longest(
            df[cn.COLUMN_NAME_UID].unique(), [1]))

    df = df.assign(
        **{cn.COLUMN_NAME_ACCURACY_TASK_TRIALS: df[cn.COLUMN_NAME_UID].map(
            accuracy_vals[False]),
            cn.COLUMN_NAME_ACCURACY_PROBE_TRIALS: df[cn.COLUMN_NAME_UID].map(
                accuracy_vals[True])
        }
    )

    df[cn.COLUMN_NAME_IS_POOR_PERFORMANCE_PARTICIPANT] = np.where(
        (df[cn.COLUMN_NAME_ACCURACY_TASK_TRIALS] < cn.TASK_MIN_ACCURACY)
        | (df[cn.COLUMN_NAME_ACCURACY_PROBE_TRIALS] < cn.PROBE_MIN_ACCURACY)
        | (df[
               cn.COLUMN_NAME_VALID_TRIALS_PROPORTION]
           < cn.VALID_TRIALS_MIN_PROPORTION),
        True, False
    )

    return df


def _report_preprocessing(df: pd.DataFrame, method: int) -> str:
    probe_less_experiment = ~df[
        cn.COLUMN_NAME_PROBED_TRIAL].any()  # The opposite of 'any'

    # Select only non probed trials for all continued operations
    _df = df.loc[~df[cn.COLUMN_NAME_PROBED_TRIAL]]

    incorrect_resps_percentage, missing_resps_percentage = (
        _extract_non_correct_task_trials_percentage(_df))

    (min_rt_cumulative_percentage, max_rt_cumulative_percentage
     ) = _extract_rt_limit_cumulative_percentages(_df)

    invalid_trials_percentage = _extract_invalid_trials_percentage(_df)

    (poor_task_accuracy_n_participants, poor_probe_accuracy_n_participants,
     low_proportion_of_valid_trials_n_participants,
     generally_poor_performance_participants
     ) = _extract_poor_performance_participants(_df)

    n_participants = _df[cn.COLUMN_NAME_UID].nunique()
    removed_participants_percentage = (
            100 * generally_poor_performance_participants / n_participants)

    clause = (

        # Irrelevant statement to probe-less  experiments
            ('' if probe_less_experiment else
             "The amounts specified below refer only to the portion of task "
             "trials, excluding all probe trials (16.6% of raw data). "
             )

            + (f"Note that data from a participant or a specific trial can be "
               "invalid due to more than one reason. We removed task trials with "
               f"incorrect ({incorrect_resps_percentage:.2f}%) or "
               f"missing ({missing_resps_percentage:.2f}%) "
               f"responses, "
               f"task trials with extremely fast (RT < {cn.TASK_TRIAL_MIN_RT},"
               f" {min_rt_cumulative_percentage}%) or slow RTs "
               f"(RT > {df[cn.COLUMN_NAME_MAXIMAL_TASK_RT].max():.0f}, {max_rt_cumulative_percentage}%)."
               )

            + (f" Next we removed the data from a total of "
               f"{generally_poor_performance_participants} participants "
               f"({removed_participants_percentage:.2f}% "
               f"out of {n_participants}) where their "
               f"accuracy on task trials was below < "
               f"{100 * cn.TASK_MIN_ACCURACY:.0f}% (N = "
               f"{poor_task_accuracy_n_participants}), ")

            # Skip if there are no probed trials
            + ('' if probe_less_experiment else
               f"accuracy on attentional probe trials was below < "
               f"{100 * cn.PROBE_MIN_ACCURACY:.0f}% (N = "
               f"{poor_probe_accuracy_n_participants}), ")
            + (f"or where less than "
               f"{(100 * cn.VALID_TRIALS_MIN_PROPORTION):.0f}% of the trials were"
               f" valid in terms of either RT, accuracy or both"
               f" (N = {low_proportion_of_valid_trials_n_participants})."
               f" In total, {invalid_trials_percentage:.2f}% of the task trials "
               f"were removed, by filtering whole participants' data or "
               f"individual trials. "
               + ('' if not probe_less_experiment else
                  "Note that out of the clean data, we select only trials which"
                  " immediately follow a correct-response task trial.")
               )
    )

    with open(OUTPUT_PREPROCESSING_TXT_PATH.format(method=method,
                                                   hashed_json=cn.HASHED_SCREENING_PARAMS),
              'w') as f:
        print(clause, file=f)


def _extract_poor_performance_participants(df: pd.DataFrame) -> Tuple[int, int]:
    return (*[df.loc[df[col_name] < col_min_value, cn.COLUMN_NAME_UID].nunique()
              for col_name, col_min_value in zip(
            (cn.COLUMN_NAME_ACCURACY_TASK_TRIALS,
             cn.COLUMN_NAME_ACCURACY_PROBE_TRIALS,
             cn.COLUMN_NAME_VALID_TRIALS_PROPORTION),
            (cn.TASK_MIN_ACCURACY, cn.PROBE_MIN_ACCURACY,
             cn.VALID_TRIALS_MIN_PROPORTION))],
            df.loc[df[cn.COLUMN_NAME_IS_POOR_PERFORMANCE_PARTICIPANT],
                   cn.COLUMN_NAME_UID].nunique()
            )


def _extract_rt_limit_cumulative_percentages(df: pd.DataFrame) -> npt.ArrayLike:
    rts = (df[cn.COLUMN_NAME_RESP_TIME]).copy()

    low_and_high_rt_boundaries = np.array([
        rts.le(cn.TASK_TRIAL_MIN_RT).mean(),
        rts.ge(df[cn.COLUMN_NAME_MAXIMAL_TASK_RT].max()).mean()]) * 100

    return low_and_high_rt_boundaries.round(2)


def _extract_non_correct_task_trials_percentage(df) -> float:
    return 100 * df[cn.COLUMN_NAME_RESP_ACCURACY].value_counts(
        True)[[0,
               -1]].values


def _extract_invalid_trials_percentage(df: pd.DataFrame) -> float:
    return np.round(
        100 * ((~df[cn.COLUMN_NAME_IS_TRIAL_VALID]) | (
            df[cn.COLUMN_NAME_IS_POOR_PERFORMANCE_PARTICIPANT])
               ).mean(), 2)


def save_demographics_report():
    files_path = _get_data_file_names()
    task_path, demog_path = files_path['task'], files_path['demog']
    task_df = pd.read_csv(task_path)
    demog_df = pd.read_csv(demog_path)

    numof_participants_with_task_data = task_df[cn.COLUMN_NAME_UID].nunique()
    numof_participants_with_demog_data = demog_df[cn.COLUMN_NAME_UID].nunique()

    difference_in_number_of_missing_demog_data = (
            numof_participants_with_task_data - numof_participants_with_demog_data)
    demog_missing_stub = (
        f'The demographics data for {difference_in_number_of_missing_demog_data} was not obtained.'
        if difference_in_number_of_missing_demog_data != 0 else '')

    percentage_of_females = \
        demog_df[cn.COLUMN_NAME_PARTICIPANT_SEX].value_counts(
            normalize=True).mul(
            100)['Female']
    min_age, max_age, mean_age, sd_age = demog_df[
        cn.COLUMN_NAME_PARTICIPANT_AGE].agg(
        ['min', 'max', 'mean',
         np.std]).values

    with open(OUTPUT_DEMOGRAPHICS_TXT_PATH, 'w') as f:
        print(
            (f'We recruited {numof_participants_with_task_data} participants. '
             f'{demog_missing_stub}'
             f'The participants were aged {min_age}-{max_age} (M = {mean_age:.2f}, SD = '
             f'{sd_age:.2f}), {percentage_of_females:.2f}% identified as female.'),
            file=f
        )


def _get_data_file_names() -> typing.Dict:
    data_files = glob.glob(INPUT_DATA_PATH)

    data_files_dict = dict(zip(
        map(lambda s: s.groups()[0],
            map(partial(re.search, 'e[1-9][a-z]{0,1}_(.+?).zip$'
                        ), data_files)),
        data_files
    ))

    return data_files_dict


def _pivot_for_external(df):
    """Pivot a dataframe to wide data, and return it, mainly useable for JASP.
     """
    df = df.pivot_table(values=cn.COLUMN_NAME_RESP_TIME,
                        index=[cn.COLUMN_NAME_UID, cn.COLUMN_NAME_CYCLE_LENGTH,
                               cn.COLUMN_NAME_FEEDBACK_TYPE],
                        columns=[cn.COLUMN_NAME_PRIOR,
                                 cn.COLUMN_NAME_CONTEXT]).copy()
    df.columns = df.columns.droplevel(0)
    cols = [f'{cn.COLUMN_NAME_PRIOR}_{_p}_{cn.COLUMN_NAME_CONTEXT}_{_c}'
            for _p, _c in (itertools.product([0, 1], [0, 1, 2, 3]))]
    df = df.set_axis(cols, axis=1).reset_index()

    # Convert to string as the main usage for these files is comparison with 
    # JASP, which doesn't like factor columns in ANOVAS to be integers.
    df[cn.COLUMN_NAME_CYCLE_LENGTH] = (
            df[cn.COLUMN_NAME_CYCLE_LENGTH].astype('string') + ' Trials').values

    return df
