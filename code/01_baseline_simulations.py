#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 14:24:16 2024

Baseline simulations, looking at results 1.

@author: Matt Ryan
"""
# %% Libraries
from bad_ctmc import *
import os
import json
import gzip
import pickle
from datetime import datetime
start_time = datetime.now()

# %%
phantom_counter = 0
parent_directory = "../data/simulations"

try:
    os.mkdir(parent_directory)
except:
    phantom_counter += 1

child_directory = parent_directory + "/baseline"

try:
    os.mkdir(child_directory)
except:
    phantom_counter += 1

child_directory_model = parent_directory + "/model"

try:
    os.mkdir(child_directory_model)
except:
    phantom_counter += 1

with open("../data/simulation_parameters.json", "r") as f:
    simulation_parameters = json.load(f)
f.close()

simulation_parameters["num_trajectory"] = 10000

save_file = child_directory + \
    f"/baseline_simulations_p_{round(simulation_parameters['params']['inf_B_efficacy'],2)}_c_{round(simulation_parameters['params']['susc_B_efficacy'],2)}_OR_{simulation_parameters['OR']}_seed_{simulation_parameters['seed']}_tend_{simulation_parameters['t_end']}_trajectories_{simulation_parameters['num_trajectory']}.gz"
save_file_model = child_directory_model + \
    f"/baseline_simulations_p_{round(simulation_parameters['params']['inf_B_efficacy'],2)}_c_{round(simulation_parameters['params']['susc_B_efficacy'],2)}_OR_{simulation_parameters['OR']}_seed_{simulation_parameters['seed']}_tend_{simulation_parameters['t_end']}_trajectories_{simulation_parameters['num_trajectory']}.json"

# %%

model = bad_ctmc(param_vals=simulation_parameters["params"],
                 P=simulation_parameters["P"],
                 I0=simulation_parameters["I0"],
                 B0=simulation_parameters["B0"],
                 t_end=simulation_parameters["t_end"])

results = model.run(number_of_trajectories=simulation_parameters["num_trajectory"],
                    seed=simulation_parameters["seed"],
                    solver=gillespy2.solvers.TauHybridCSolver)

# with open(save_file, "w") as f:
#     json.dump(results.to_json(), f)
# f.close()

with gzip.open(save_file, "wb") as f:
    f.write(compress_data(results.to_json()))
f.close()

# %%
with open(save_file_model, "wb") as f:
    pickle.dump(model, f)
f.close()

print(f"Time taken: {datetime.now()-start_time}")
