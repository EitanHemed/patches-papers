#!/bin/bash

# Enable strict mode.
set -euo pipefail

set +euo pipefail
source activate patches
set -euo pipefail

cd "patches-papers/Code"
jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root --NotebookApp.token=''