import re
from functools import partial

import numpy as np
import pandas as pd

from po_utils import constants as c

CURRENT_EXP = 'e3'

# TODO - unpack also demographics, Adam and debrief questionnaires


def load_raw():
    df = pd.read_csv(f'Input/raw_{CURRENT_EXP}.zip')

    # remove pilot runs
    df = df.loc[df['participant'] >= 1000]

    df[c.COLUMN_NAME_UID] = df['participant'].astype(str) + ':' + df[
        'date'].astype(str)
    df[c.COLUMN_NAME_FEEDBACK_TYPE] = 'blank'

    df[c.COLUMN_NAME_FEEDBACK_MANIPULATION_TYPE] = 1

    return df


def handle_task(df):
    task_df = df.copy()

    task_df = task_df.rename(
        columns={
            'task_key_resp.rt': c.COLUMN_NAME_RAW_RESP_TIME,
            'response_accuracy': c.COLUMN_NAME_RESP_ACCURACY
        }
    )

    task_df[[c.COLUMN_NAME_RAW_RESP_COUNT, c.COLUMN_NAME_RAW_RESP_DURATION]
    ] = np.nan

    task_df[c.COLUMN_NAME_RAW_RESP_TIME] *= 1000
    task_df[c.COLUMN_NAME_FEEDBACK_CYCLE] = task_df[
        'effect_pertubration_theta'].isna()

    task_df[c.COLUMN_NAME_TASK_RESPONSE_WINDOW_DURATION] = 850
    task_df[c.COLUMN_NAME_FEEDBACK_DURATION] = 100
    task_df[c.COLUMN_NAME_MAXIMAL_TASK_RT] = (
            task_df[c.COLUMN_NAME_TASK_RESPONSE_WINDOW_DURATION]
            - task_df[c.COLUMN_NAME_FEEDBACK_DURATION])

    task_df = task_df.sort_values(
        [c.COLUMN_NAME_UID, c.COLUMN_NAME_TRIAL_NUMBER])

    task_df.loc[~task_df['trials.thisRepN'].isna()].to_csv(
        f'Input/{CURRENT_EXP}_task.zip', index=False)

    task_df.loc[~task_df['instruct_key'].isna()].to_csv(
       f'Input/{CURRENT_EXP}_training_block.zip',
        index=False)


def handle_adam(df):
    adam_df = df.copy().loc[df['adam_scale_item_slider.response'].notna()]
    adam_df = adam_df.groupby(
        [c.COLUMN_NAME_UID, 'val'])['adam_scale_item_slider.response'].mean(
    ).unstack().reset_index().rename(columns={'pos': 'Positive Scale',
                                              'neg': 'Negative Scale'})

    adam_df.to_csv(f'Input/{CURRENT_EXP}_adam.zip')


def handle_demog(df):
    stub = 'demog_que_(.*)_slider.response'
    responses = df.filter(
        regex=stub).columns.tolist()
    exp = re.compile(stub)

    demog_df = df.loc[:, [c.COLUMN_NAME_UID] + responses].dropna()

    demog_df = demog_df.rename(columns=dict(zip(responses,
                                                [s.groups()[0] for s in list(
                                                    map(partial(re.search,
                                                                exp, ),
                                                        responses))])))

    demog_df['sex'] = demog_df['sex'].map(
        dict(zip(range(1, 4), ['Male', 'Female', 'Rather not say']))).values
    demog_df['hand'] = demog_df['hand'].map(
        dict(zip(range(1, 4), ['Right', 'Left', 'Ambidextrous']))).values
    demog_df['adhd'] = demog_df['adhd'].map(
        dict(zip(range(1, 4), ["ADHD", "ADD", "None"]))).values

    demog_df.to_csv(f'Input/{CURRENT_EXP}_demog.zip')


def handle_debrief(df):
    debrief_df = df.loc[df['debrief_resp'].notna()]

    debrief_df = debrief_df.groupby(
        [c.COLUMN_NAME_UID, 'debrief_item_he', ])['debrief_resp'].first(
    ).unstack().reset_index()

    debrief_df.to_csv(f'Input/{CURRENT_EXP}_debrief.zip')


def main():
    df = load_raw()
    [f(df) for f in [handle_task, handle_adam, handle_demog, handle_debrief]]


if __name__ == '__main__':
    main()
