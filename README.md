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

The relevant pre-registrations are available [here](https://osf.io/b6c9a/wiki/home/).

## Setup

To run the pipeline you can either choose between running it using [docker](https://docker.com),
or using a local installation.

### Docker

First, follow this guide to install Docker on your system: https://docs.docker.com/get-docker/

After setting up Docker on your system, pull the image from Docker Hub by running the following command in the terminal:

`docker pull eitanhemed/patches-papers:latest
`

Then, run the following command to start the container, which will also launch up a Jupyter server:

`docker run -p 8888:8888 eitanhemed/patches-papers
`

Once the Jupyter server is up, you can access it by opening the following link in your browser, for example by going to
the terminal and clicking on the link: http://127.0.0.1:8888/tree, or any other link that appears in your terminal.
Your entrypoint will be a jupyter notebook, allowing you to explore the data and output, edit the project code, etc.

### Local installation

You will need to install a few dependencies. The best option
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
process can take a few minutes on Windows, and about 10-15 minutes on Linux.

## Usage

Regardless of setting up locally or via Docker, using the project environment, run `python run_all.py`
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
