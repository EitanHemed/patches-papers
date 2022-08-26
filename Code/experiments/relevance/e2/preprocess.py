import pandas as pd

from po_utils import constants as c

CURRENT_EXP = 'e2'

def unique_participant(df):
    return (df['participant'].astype(str)
            + ':' + df['date'].astype(str))


def handle_task():
    df = pd.read_csv(f'Input/raw_{CURRENT_EXP}_task.zip')
    # remove pilot runs
    df = df.loc[(df['participant'] < 900)]
    df[c.COLUMN_NAME_UID] = unique_participant(df)
    df[c.COLUMN_NAME_FEEDBACK_MANIPULATION_TYPE] = 0

    df[c.COLUMN_NAME_TASK_RESPONSE_WINDOW_DURATION] = 1000
    df[c.COLUMN_NAME_FEEDBACK_DURATION] = 150
    df[c.COLUMN_NAME_MAXIMAL_TASK_RT] = (
            df[c.COLUMN_NAME_TASK_RESPONSE_WINDOW_DURATION]
            - df[c.COLUMN_NAME_FEEDBACK_DURATION])

    df = df.sort_values(
        [c.COLUMN_NAME_UID, c.COLUMN_NAME_TRIAL_NUMBER])

    df.to_csv(f'Input/{CURRENT_EXP}_task.zip')


def handle_adam():
    df = pd.read_csv(f'Input/raw_{CURRENT_EXP}_adam.zip')
    df[c.COLUMN_NAME_UID] = unique_participant(df)
    df.groupby([c.COLUMN_NAME_UID, 'val'])['rating.response'].mean(
    ).unstack().reset_index().rename(columns={'neg': 'Negative Scale',
                                              'pos': 'Positive Scale'})
    df.to_csv(f'Input/{CURRENT_EXP}_adam.zip')


def _handle_open_ended(df: pd.DataFrame):
    df = df.loc[:, ['participant', 'date', 'que', 'resp']]
    df = df.loc[df['participant'] < 900]

    df[c.COLUMN_NAME_UID] = unique_participant(df)
    # df[['participant', 'date']].astype(str).apply(
    #     lambda s: '{} - {}'.format(*s.values), axis=1).factorize()[0]
    df = df.loc[df['resp'].notna()]
    return df.groupby(['unique_participant', 'que'])[
        'resp'].first().unstack().reset_index()


def handle_demog():
    df = _handle_open_ended(pd.read_csv(f'Input/raw_{CURRENT_EXP}_demog.zip'))
    df[c.COLUMN_NAME_PARTICIPANT_SEX] = df[
        c.COLUMN_NAME_PARTICIPANT_SEX].replace({'1': 'Male', '2': 'Female'})
    df.to_csv(f'Input/{CURRENT_EXP}_demog.zip')


def handle_debrief():
    df = _handle_open_ended(pd.read_csv(f'Input/raw_{CURRENT_EXP}_debrief.zip'))
    df.to_csv(f'Input/{CURRENT_EXP}_debrief.zip')


def handle_subjectives():
    df = pd.read_csv(f'Input/raw_{CURRENT_EXP}_subjectives.zip')
    df = df.loc[df['participant'] < 900]

    df = df[['participant', 'date', ] +
            df.filter(regex='response$').columns.tolist()]

    df = df.groupby(['date', 'participant']).mean()
    resp_columns = df.columns.tolist()
    df = df.reset_index()

    df = df.rename(
        columns=dict(zip(
            resp_columns, [i.split(".")[0].title().replace('_', ' ')
                           for i in
                           resp_columns])))

    df[c.COLUMN_NAME_UID] = unique_participant(df)

    # df[['participant', 'date']].astype(str).apply(
    #     lambda s: '{} - {}'.format(*s.values), axis=1).factorize()[0]

    df.rename(columns={'date': "date_subjectives"}, inplace=True)

    df.to_csv(f'Input/{CURRENT_EXP}_subjectives.zip')


def main():
    handle_task()
    handle_adam()
    handle_demog()
    handle_debrief()
    handle_subjectives()


if __name__ == '__main__':
    main()
