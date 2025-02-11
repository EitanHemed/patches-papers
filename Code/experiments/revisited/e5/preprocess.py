import numpy as np
import pandas as pd

from po_utils import constants as c

CURRENT_EXP = 'e5'

def load_raw_task():
    df = pd.read_csv(f'Input/raw_{CURRENT_EXP}_task.zip')

    df[c.COLUMN_NAME_UID] = (df['participant'].astype(str)
                             + ':' + df['date'].astype(str))
    df[c.COLUMN_NAME_FEEDBACK_TYPE] = 'blank'
    df[c.COLUMN_NAME_FEEDBACK_MANIPULATION_TYPE] = 1

    df = df.loc[(df['participant'].notna())
                & (df['participant'].lt(900))
                # Remove participants which their session crashed
                & (~df['participant'].isin([52, 85, 9]))
                ]

    return df


def handle_task(df):
    df = df.copy().rename(
        columns={
            'task_key_resp.rt': c.COLUMN_NAME_RAW_RESP_TIME,
            'task_key_resp.corr': c.COLUMN_NAME_RESP_ACCURACY
        }
    )

    df.loc[df['task_key_resp.keys'].isna(), c.COLUMN_NAME_RESP_ACCURACY] = -1

    df[[c.COLUMN_NAME_RAW_RESP_COUNT, c.COLUMN_NAME_RAW_RESP_DURATION]
    ] = np.nan

    df[c.COLUMN_NAME_RAW_RESP_TIME] *= 1000
    df[c.COLUMN_NAME_FEEDBACK_CYCLE] = (
        df[['effect_theta_displacement', 'effect_radius_displacement']].eq(
            0).all(axis=1))

    df[c.COLUMN_NAME_TASK_RESPONSE_WINDOW_DURATION] = 1250
    df[c.COLUMN_NAME_FEEDBACK_DURATION] = 100
    df[c.COLUMN_NAME_MAXIMAL_TASK_RT] = (
            df[c.COLUMN_NAME_TASK_RESPONSE_WINDOW_DURATION]
            - df[c.COLUMN_NAME_FEEDBACK_DURATION])

    df = df.sort_values([c.COLUMN_NAME_UID, c.COLUMN_NAME_TRIAL_NUMBER])

    df.to_csv(
        f'Input/{CURRENT_EXP}_task.zip', index=False)


def load_raw_ptq():
    ptq_df = pd.read_csv(f'Input/raw_{CURRENT_EXP}_ptq.zip')
    ptq_df = ptq_df.loc[:, ~ptq_df.columns.str.contains('^Unnamed')].rename(
        columns={"מספר נבדק": 'participant'})

    ptq_df['participant'] = ptq_df['participant'].str.extract('(\d+)').astype(
        'Int64').values
    # Remove pilot participants, or participants which their session crashed.
    ptq_df = ptq_df.loc[(ptq_df['participant'].notna())
                        & (ptq_df['participant'].lt(900))
                        & (~ptq_df['participant'].isin([52, 85, 9]))
                        ]

    return ptq_df


def handle_adam(df):
    adam_df = df[['Timestamp', 'participant',
                  'כאשר אני זז ופועל, זה מרגיש כאילו אני רובוט שמופעל מרחוק',
                  'תחושתי היא שאני בסך הכל כלי בידיו של מישהו/משהו אחר',
                  'אני שולט לחלוטין במה שאני עושה',  # 1
                  'מעשי קורים בלי שאני מתכוון לכך',  #
                  'אני הוא היוצר של מעשיי',  # 4
                  'יש לי התחושה שתוצאות פעולותי לא נובעות מהמעשים שלי',  # 5
                  'התנועות שלי הן אוטומטיות - הגוף שלי פשוט עושה אותן',  # 6
                  'היזימה של הפעולות שלי היא פתאומית ומפתיעה',  # 7
                  'הדברים שאני עושה כפופים אך ורק לרצוני החופשי',  # 8
                  'ההחלטה אם ומתי אפעל נמצאת בידי',  # 9
                  'מעשי אינם תוצאה של החלטה לעשות אותם',
                  'אני אחראי לחלוטין לכל תוצאות מעשי',
                  'ההתנהגות שלי מתוכננת על ידי מתחילתה ועד סופה'
                  ]]
    adam_df = adam_df.rename(columns=dict(zip(adam_df.columns[2:],
                                              [11, 2, 1, 3, 4, 5, 6, 7, 8, 9,
                                               10, 13, 12])))

    adam_df['Positive Scale'] = adam_df[[1, 4, 8, 9, 12, 13]].mean(axis=1)
    adam_df['Negative Scale'] = adam_df[[2, 3, 5, 6, 7, 10, 11]].mean(axis=1)

    adam_df = adam_df[['participant', 'Timestamp',
                       'Positive Scale', 'Negative Scale']]

    adam_df.to_csv(f'Input/{CURRENT_EXP}_adam.zip')


def handle_demog(df):
    df[c.COLUMN_NAME_UID] = df['participant'].astype(str) + ':' + df[
        'Timestamp']

    demog_df = df.rename(
        columns={
            'מין': c.COLUMN_NAME_PARTICIPANT_SEX,
            'גיל': c.COLUMN_NAME_PARTICIPANT_AGE,
            'ציון פסיכומטרי': 'SAT',
            "תוכנית לימודים (חוג וכו')": 'Course',
            'ממוצע תואר (0 במידה ולא סטודנט)': 'GPA',
            'יד דומיננטית': 'Dominant Hand',
            'האם אי פעם אובחנת כסובל מADD/ADHD?': 'ADHD diagnosis',
            'האם הנך סובל מעוורון צבעים?': 'Colorblindness diagnosis',
            'האם אתה סובל מליקויי ראיה?': 'Sight impairments',
            'האם השתתפת בניסוי זה עבור תשלום או קרדיט?': 'Credit participation'
        }
    )

    demog_df[c.COLUMN_NAME_PARTICIPANT_SEX] = demog_df[
        c.COLUMN_NAME_PARTICIPANT_SEX].map({'זכר': 'Male',
                                            'נקבה': 'Female'})

    demog_df['Sight impairments'] = demog_df['Sight impairments'].map(
        {'לא, הראייה שלי תקינה': 'None',
         'כן - ראייה מתוקנת באמצעות עדשות / משקפיים': 'Corrected',
         'כן - אבל הראייה שלי לא מתוקנת באמצעות עדשות / משקפיים': 'Uncorrected'
         }
    )

    demog_df['Credit participation'] = demog_df['Credit participation'].map(
        {'תשלום': 'No', 'קרדיט': 'Yes'})

    demog_df['Colorblindness diagnosis'] = (
        demog_df['Colorblindness diagnosis'].map(
            {'לא': 'No', 'כן': 'Yes'}))

    demog_df['ADHD diagnosis'] = (
        demog_df['ADHD diagnosis'].map(
            {'לא': 'No',
             'כן - ADHD': 'ADHD',
             'כן - ADD': 'ADD'}))

    demog_df['Dominant Hand'] = demog_df['Dominant Hand'].map(
        {'ימין': 'Right', 'שמאל': 'Left'}
    )

    demog_df[['ADHD diagnosis', 'Dominant Hand', c.COLUMN_NAME_UID,
              c.COLUMN_NAME_PARTICIPANT_SEX,
              c.COLUMN_NAME_PARTICIPANT_AGE]].to_csv(
        f'Input/{CURRENT_EXP}_demog.zip')


def handle_debrief(df):
    debrief_df = df.copy()[['participant', 'Timestamp',
                            'מה לדעתך הייתה מטרת הניסוי?',
                            'האם לדעתך הייתה חוקיות כלשהי שגרמה לעיגולים להפוך לבנים?',
                            'מה לדעתך הייתה מטרת המשולשים הצהובים?',
                            'האם תרצה להוסיף משהו בנוגע לניסוי?',
                            ]]

    debrief_df.to_csv(f'Input/{CURRENT_EXP}_debrief.zip')


def handle_subjectives(df):
    subjectives_df = df.copy()[['participant',
                                'Timestamp'] + df.filter(
        like='100').columns.tolist()]
    subjectives_df.to_csv(f'Input/{CURRENT_EXP}_subjectives.zip')


def main():
    handle_task(load_raw_task())
    ptq_df = load_raw_ptq()
    [f(ptq_df) for f in
     [handle_adam, handle_demog, handle_debrief, handle_subjectives]]


if __name__ == '__main__':
    main()
