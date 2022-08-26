# Patches papers

Author: [Eitan Hemed](mailto:Eitan.Hemed@gmail.com)

This repository contains the code and data associated with the following
pre-prints:

1. [Hemed, E., Bakbani Elkayam, S., Teodorescu, A., Yona, L., & Eitam, B.
   (2022). Evaluation of an Action’s Effectiveness by the Motor System in a Dynamic Environment: Amended.](Manuscripts/Revisited MS.pdf)

2. [Hemed, E., & Eitam, B. (2022). Control feedback increases response speed independently of the feedback’s goal- and task-relevance.
   ](Manuscripts/Relevance MS.pdf)

-----

## Repository's Structure

The repository structure is as follows:

1. [Manuscripts](Manuscripts) - the preprints.
2. [Code](Code) - the code and data to reproduce the analyses associated with
   the papers.

More detailed description coming soon.

## Setup

To run the pipeline you will need to install a few dependencies. The best option
is to do it on a new conda environment, as follows:

```
conda create -n po_utils_env python=3.9.12
conda activate po_utils_env
conda install -c conda-forge r-base=4 -y
cd Code
pip install .
```

Note that installing `robusta` involves setting up R on your system. The first
session in which `robusta` is imported will require R to install *many*
packages. The first time you import `robusta` the dependencies installation
process can take a few minutes on Windows, and up to 30 minutes on Linux.

## Usage

Using the project environment, run `python run_all.py`
from the `Code` directory.

## FAQ

**I want to analyze the data using something different from `robusta` (
R, SPSS, etc.). What are my options?**

Your best option is to use the data exported during any of the preprocessing
stages (e.g.,
`Code/Experiments/relevance/e1/Output/amended/13b9435ca5add3409d7fb2cbc6f836a0/Data/Data/pre_aggregation.csv`)

The wide-format dataframe found under the output data directory was used to
compare the results of the pipeline to the results given by
[JASP](https://github.com/jasp-stats/).

**How to change the screening parameters? (e.g., minimum valid response time,
proportion of allowed invalid trials**)

Edit `po_utils.constants.SCREENING_PARAMS`, before running the pipeline.

Each unique combination of screening parameters is hashed, so the output of a
set of screening parameters will be saved under the respective output
directory (e.g.,
`Code/Experiments/relevance/e1/Output/amended/13b9435ca5add3409d7fb2cbc6f836a0`)
.
