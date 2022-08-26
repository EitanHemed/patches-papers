import json
from pathlib import Path

from . import constants as cn
from . import preprocessing, plotting, reporting, meta_analysis_utils


def run_pipeline(method: str = cn.PREPROCESSING_METHOD_AMENDED) -> None:
    """Run the project's pipeline for the experiment found in the current
        folder.

    Parameters
    ----------
    method: str
        Controls the preprocessing method used.
        If 'confounded', use method from Hemed et al., 2020 (see reference #1
        below). If 'amended', use the method from Hemed et al., 2022 (see
        reference #2 below).

    Returns
    -------
    None

    References
    ----------
    1. Hemed, E., Bakbani-Elkayam, S., Teodorescu, A. R., Yona, L., & Eitam, B.
         (2020). Evaluation of an action’s effectiveness by the motor system in
         a dynamic environment. Journal of Experimental Psychology: General,
         149(5), 935.
    2. Hemed, E., Bakbani Elkayam, S., Teodorescu, A., Yona, L., & Eitam, B.
        (2022). Evaluation of an Action’s Effectiveness by the Motor System in a
        Dynamic Environment: Amended.
    """

    assert method in [cn.PREPROCESSING_METHOD_CONFOUNDED,
                      cn.PREPROCESSING_METHOD_AMENDED]

    _prep_output_dirs(method)
    save_json(method)

    # Take only the first argument out of the analyzed results, the one which
    # includes t-test results. The 2nd is a dictionary containing the ANOVA
    # results).

    preprocessing.save_demographics_report()

    meta_analysis_utils.pool_t_test_results_for_single_experiment(
        plotting.analyze_and_plot(preprocessing.preprocess(method), method)[0],
        method)


def _prep_output_dirs(method: str) -> None:
    """    Make sure that the output files directories exist.
    """
    Path(preprocessing.OUTPUT_PATH.format(method=method,
                                          hashed=cn.HASHED_SCREENING_PARAMS)).mkdir(
        parents=True, exist_ok=True)
    Path(plotting.OUTPUT_PATH.format(method=method,
                                     hashed=cn.HASHED_SCREENING_PARAMS)).mkdir(
        parents=True, exist_ok=True)
    Path(reporting.OUTPUT_PATH.format(method=method,
                                      hashed=cn.HASHED_SCREENING_PARAMS)).mkdir(
        parents=True, exist_ok=True)


def save_json(method: str, top_dir: str = 'Output'):
    with open(f"{top_dir}/{method}/{cn.HASHED_SCREENING_PARAMS}/screening_params.json", "w",
              encoding='utf-8') as f:
        json.dump(cn.SCREENING_PARAMS, f, ensure_ascii=False, indent=4)
