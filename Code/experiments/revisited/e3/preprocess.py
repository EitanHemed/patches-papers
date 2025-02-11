import numpy as np
import pandas as pd

from po_utils import constants as c

CURRENT_EXP = 'e3'

# TODO - unpack also demographics, Adam and debrief questionnaires


def unique_participant(df):
    return (df['participant'].astype(str)
            + ':' + df['date'].astype(str))


def handle_task():
    df = pd.read_csv(f'Input/raw_{CURRENT_EXP}_task.zip')

    # remove pilot runs
    df = df.loc[df['participant'] < 900]

    df[c.COLUMN_NAME_UID] = df['participant'].astype(str) + ':' + df[
        'date'].astype(str)
    df[c.COLUMN_NAME_FEEDBACK_TYPE] = 'blank'

    df[c.COLUMN_NAME_FEEDBACK_MANIPULATION_TYPE] = 0

    df = df.rename(columns={'rt': c.COLUMN_NAME_RAW_RESP_TIME,
                            'probedtrial': c.COLUMN_NAME_PROBED_TRIAL,
                            'showeffectpatch': c.COLUMN_NAME_FEEDBACK_CYCLE,
                            })

    df[c.COLUMN_NAME_PROBED_TRIAL] = df[c.COLUMN_NAME_PROBED_TRIAL
    ].map({'0': False, 'True': True}).values

    df[[c.COLUMN_NAME_RAW_RESP_COUNT, c.COLUMN_NAME_RAW_RESP_DURATION]
    ] = np.nan

    df[c.COLUMN_NAME_TASK_RESPONSE_WINDOW_DURATION] = 850
    df[c.COLUMN_NAME_FEEDBACK_DURATION] = 100
    df[c.COLUMN_NAME_MAXIMAL_TASK_RT] = (
            df[c.COLUMN_NAME_TASK_RESPONSE_WINDOW_DURATION]
            - df[c.COLUMN_NAME_FEEDBACK_DURATION])

    df = df.sort_values([c.COLUMN_NAME_UID, c.COLUMN_NAME_TRIAL_NUMBER])

    df.loc[~df['expblocksthisn'].isna()].to_csv(f'Input/{CURRENT_EXP}_task.zip',
                                                index=False)


def handle_adam_and_demog():
    df = pd.read_csv(f'Input/raw_adam_and_demo_{CURRENT_EXP}.zip')

    # remove pilot runs
    df = df.loc[df['participant'] < 900]

    df[c.COLUMN_NAME_UID] = df['participant'].astype(str) + ':' + df[
        'date'].astype(str)

    df = df.rename(columns={
        'adam_pos': 'Negative Scale', 'adam_neg': 'Positive Scale',
        'add': 'adhd'
    })

    df[c.COLUMN_NAME_UID] = unique_participant(df)

    df[[c.COLUMN_NAME_UID, 'Negative Scale', 'Positive Scale']].to_csv(
        f'Input/{CURRENT_EXP}_adam.zip', index=False)

    df[c.COLUMN_NAME_PARTICIPANT_SEX] = df[
        c.COLUMN_NAME_PARTICIPANT_SEX].replace({1: 'Male', 2: 'Female'})

    df[[c.COLUMN_NAME_UID, c.COLUMN_NAME_PARTICIPANT_SEX,
        c.COLUMN_NAME_PARTICIPANT_AGE]].to_csv(
        f'Input/{CURRENT_EXP}_demog.zip', index=False)


def main():
    handle_task()
    handle_adam_and_demog()


if __name__ == '__main__':
    main()
