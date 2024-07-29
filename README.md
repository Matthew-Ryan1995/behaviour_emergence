# behaviour_emergence

This git repo will reproduce the results for the paper *Behaviour emergence in a BaD modelling framework*.  The purpose of this work is to investigate:

1. Early approximations of the prevalence of self protective behaviours in the presence of an infectious disease, and
2. The effects of self protective behaviour on the outbreak probability and final size of an epidemic.

This is a compartmental stochastic model for a covid-like illness with self protective behaviour accounted for using behaviour science theory and is simulated using the `gillespy2` python package.  Other modules needed are in `requirements.txt` file.


# Contents

*Model code*: Python scripts to fit the CTMC and ODE versions of the compartmental model. 

- code/BaD.py
- code/bad_ctmc.py

*Model parameters*: Default parameters and simulation parameters.

- code/model_parameters.json
- code/00_create_parameters_set.py

*Results 1*: create simulations for behaviour approximations and generate plots.

- code/01_baseline_simulations.py
- 04_results_1_estimating_behaviour.py

*Results 2*: Effects of varying initial behaviour and infectiousness of disease.

- code/03_Bstar_by_R0_simulations.py
- code/05_results_2_collect_data.py
- code/06_results_2_plots.py

*Results 3*: Interventions on epidemiological outcomes.

- code/02_intervention_simulations.py
- code/07_results_3_collect_data.py
- code/08_results_3_plots.py

*Supplement*: Code to investigate within and between odds ratio, and perform parameter sweep of $\omega_3$.

- code/09_within_OR_simulations.py
- code/10_within_OR_collect_data.py
- code/11_within_OR_plots.py
- code/12_between_OR_simulations.py
- code/13_between_OR_collect_data.py
- code/14_between_OR_plots.py
- code/w3_sweep/*

*HPC Versions*: version of code to run all simulations on a SLURM HPC.

- HPC_versions/*
