# Evaluating the Impact of Model Accuracy for Optimizing Battery Energy Storage Systems

This repository contains the code that accompanies the paper "Evaluating the Impact of Model Accuracy for Optimizing Battery Energy Storage Systems".

## Table of Contents
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)

## Project Structure

The project is organized as follows:

- **`benchmark/`**: Contains the code to set up the optimization and simulation models, along with utility functions for analyzing and plotting results.
- **`benchmark/run.py`**: Script to run the MPC simulations in parallel.
- **`analysis.ipynb`**: Jupyter notebook for analyzing and displaying the results of the simulations.
- **`notebooks/efficiency.ipynb`**: Jupyter notebook for system efficiency characterization (refer to Fig. 1 of the paper).

## Setup Instructions

To set up the project environment, follow these steps:

1. **Download and install the Bonmin Solver** from the following link: [https://portal.ampl.com/user/ampl/download/coin](https://portal.ampl.com/user/ampl/download/coin).

2. **Set Up the Environment**:
   - Install the Python package manager `uv` by following the instructions here: [https://docs.astral.sh/uv/](https://docs.astral.sh/uv/).
   - Install all dependencies by running the following command in your terminal:
     ```bash
     uv sync
     ```

The `uv` lock file ensures a completely reproducible environment, for example by intalling the specific experimental version of [simses](https://github.com/martincornejo/simses-lite?rev=72181f12e50c5db681102281f5e8c7b8e35d2193) used in this study.

## Usage

To run the simulations, execute the following command in your terminal:
```bash
python benchmark/run.py
```
All scenarios are run in parallel, which significantly speeds up the simulation process. The original simulations were executed on an Intel Xeon W-2295 processor with 18 cores (36 threads).
Please note that using processors with fewer parallelization capabilities may result in longer simulation times. For optimal performance, it is recommended to run the simulations on a multi-core processor.

After running the simulations, you can analyze the results using the `analysis.ipynb` notebook.
