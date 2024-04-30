#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 14:24:16 2024

@author: rya200
"""
# %% Libraries
from bad_ctmc import *
import os
import json
import pickle

# %%

parent_directory = "../data/simulations"

try:
    os.mkdir(parent_directory)
except:
    print()

child_directory = parent_directory + "/baseline"

try:
    os.mkdir(child_directory)
except:
    print()

child_directory_model = parent_directory + "/model"

try:
    os.mkdir(child_directory_model)
except:
    print()

with open("../data/simulation_parameters.json", "r") as f:
    simulation_parameters = json.load(f)
f.close()

save_file = child_directory + \
    f"/baseline_simulations_seed_{simulation_parameters['seed']}_tend_{simulation_parameters['t_end']}_trajectories_{simulation_parameters['num_trajectory']}.json"
save_file_model = child_directory_model + \
    f"/baseline_simulations_seed_{simulation_parameters['seed']}_tend_{simulation_parameters['t_end']}_trajectories_{simulation_parameters['num_trajectory']}.json"

# %%

model = bad_ctmc(param_vals=simulation_parameters["params"],
                 P=simulation_parameters["P"],
                 I0=simulation_parameters["I0"],
                 B0=simulation_parameters["B0"],
                 t_end=simulation_parameters["t_end"])

results = model.run(number_of_trajectories=simulation_parameters["num_trajectory"],
                    seed=simulation_parameters["seed"])

with open(save_file, "w") as f:
    json.dump(results.to_json(), f)
f.close()
# %%
with open(save_file_model, "wb") as f:
    pickle.dump(model, f)
f.close()
