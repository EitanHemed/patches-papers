from . import constants as cn, preprocessing, plotting

EXPERIMENTS = ['e1', 'e2', 'e3', 'e4']

def main():
    data = load_data()


def load_data():
    hashed = cn.HASHED_SCREENING_PARAMS
    method = 'amended'

    return {k:
        f'revisited/{k}/' + preprocessing.OUTPUT_PATH + cn.OUTPUT_DATA_FNAME_RAW.format(
        hashed, method) for k in EXPERIMENTS}

