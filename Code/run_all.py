"""
Run the full pipeline - preprocessing, analysis and plotting/reporting routines
for all experiments. Run using both preprocessing methods.
"""

import glob
import itertools
import os

from po_utils import constants as cn
from po_utils import run_pipeline, meta_analysis_utils
from po_utils import save_json

DIRS = glob.glob('experiments/*/*')
METHODS = (
    cn.PREPROCESSING_METHOD_AMENDED, cn.PREPROCESSING_METHOD_CONFOUNDED)


def wrap_pipeline(directory, method, root_dir):
    os.chdir(directory)
    run_pipeline(method)
    os.chdir(root_dir)


if __name__ == "__main__":
    root_dir = os.getcwd()

    [wrap_pipeline(directory, method, root_dir) for directory, method in
     itertools.product(DIRS, METHODS)]

    [meta_analysis_utils.save_summaries_from_all_experiments(
        DIRS, m) for m in METHODS]

    meta_analysis_utils.visualize_meta_analyses(meta_analysis_utils.run_meta_analyses())
    #
    # [meta_analysis_utils.draw_meta_fig(m) for m in METHODS]
    #
    # [save_json(m, 'meta_analysis') for m in METHODS]

