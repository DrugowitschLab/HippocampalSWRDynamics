# Replay Spatio-Temporal Dynamics Analysis

This repo contains the code for Krause and Drugowitsch (2022). "A large majority of 
awake hippocampal sharp-wave ripples feature spatial trajectories with momentum". Neuron. 

This repo uses [Poetry](https://python-poetry.org/docs/) to manage Python dependences.

## Code file structure

`replay_structure` - python files that contain analysis code

`scripts` - python cli scripts for running analysis

`notebooks` - jupyter notebooks for producing the figures in the paper

## High-level overview of analysis pipeline

This code was written for analyzing the dataset from Pfieffer and Foster (2013, 2015).
Please contact Pfeiffer and Foster for the dataset.

The code is split into modules that house the analysis code (stored in 
`/replay-structure`), and command line interfaces for running
the analyses (stored in `/scripts`). The analysis pipeline can roughly be broken down 
into 3 parts: 
1. Data preprocessing
2. Dynamics model analysis
3. Behavioral analysis

I will briefly describe how the files are interrelated, to hopefully help interpret the
structure of the code in this repo. 
I will describe the steps in terms of the cli used to run the analysis, referencing the 
modules that are implicated in each step.

*Note:* The "dynamics" models described in Krause and Drugowitsch (2022) are referred to 
in the code as "structure" models. Additionally, the "Gaussian" model is referred to as
"Stationary Gaussian" in the code.

### Data preprocessing

The Pfieffer and Foster (2013, 2015) dataset is initially loaded and reformatted by 
running the cli `preproccess_ratday_data.py` (imports the module `ratday.py`). 
From this initial preprocessing stage, neural data is 
extracted from SWRs, HSEs, or run snippets by running `preprocess_spikemat_data.py`
(which imports the modules `ripple_preprocessing.py`,
`highsynchronyevents.py`, or `run_snippet_preprocessing.py`, respectively, for each
type of data preprocessing).

Simulated neural data is generated using `generate_model_recovery_data.py` (imports 
`simulated_trajectories.py`, `simulated_neural_data.py` and `model_recovery.py`).

Running `reformat_data_for_structure_analysis.py` (imports `structure_analysis_input.py`)
then puts the preprocessed data (SWRs, HSEs, run snippets, and simulated SWRS) into a 
consistent format for feeding into the dynamics models.

### Dynamics model analysis


#### Dynamics models

The dynamics models are implemented in the `structure_models.py` module. The 
Diffusion and Momentum models both utilize the forward-backward algorithm (Bishop 2006), 
which is implemented in `forward_backward.py`. 

The dynamics models without parameters that require a parameter gridsearch (
the Stationary and Random model) are run using the cli `run_model.py`.
For the dynamics models with parameters (the Diffusion, Momentum, and Gaussian 
models), we used the Harvard Medical School computing cluster (called o2) to parallelize 
running the same model many times across a parameter gridsearch (code is stored in
 `scripts/o2/`). The module `structure_models_gridsearch.py` houses the code for 
running the dynamics models across a parameter gridsearch.

The cli `run_deviance_explained.py` (imports `deviance_models.py`) calculates the 
deviance explained of each model for each SWR.

For visualization purposes, the cli `get_marginals.py` (imports `marginals.py`) 
calculates the position marginals for each dynamics model.

#### Model comparison

Running the dynamics models calculates the model evidence of each model for each SWR.
Model comparison across the dynamics modes is run using the cli `run_model_comparion.py` 
(imports `model_comparison.py`). 

#### Trajectory decoding

In addition to identifying the spatio-temporal dynamics of SWRs, we also decoded the 
most likely trajectories within each SWR. Trajectories are extracted by running 
`get_trajectories.py` (imports `structure_trajectories.py`). We use the viterbi 
algorithm (Bishop, 2006) for decoding, which is implemented in `viterbi_algorithm.py`.


### Behavioral analysis

The behavioral analysis described in Figures 6 and 7 is run using
`get_descriptive_stats.py` (imports `descriptive_stats.py`) and
`run_predictive_analysis.py` (imports `predictive_analysis.py`), respectively.


### Other

#### Configuration files

The parameters used in for running the data preprocessing and dynamics models are 
stored in `config.py`.

The custom Python types for the data type (e.g. ripples, run_snippets),
session type (e.g. real recording sessions, simulated sessions),
dynamics model type (e.g. diffusion, stationary, etc.), and
likelihood type (e.g. poisson, neg_binomial) are defined in `metadata.py`.

#### Implementations of previous methods

The comparison of our dynamics models to the method described 
in Stella et al. (2019), described in Figure 5, is run using `run_diffusion_constant.py` 
(imports `diffusion_constant.py`).

We also implemented the decoding method from Pfeiffer and Foster (2013) for
visualization purposes, which is run using `run_pf_analysis.py` 
(imports `pf_analysis.py`).