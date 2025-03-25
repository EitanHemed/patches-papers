#!/bin/bash

rm -rf patches-papers
git clone https://github.com/EitanHemed/patches-papers

# Enable strict mode.
set -euo pipefail

source activate patches
set -euo pipefail

cd "patches-papers/Code"
jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root --NotebookApp.token=''