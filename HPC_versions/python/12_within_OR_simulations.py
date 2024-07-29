#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 14:46:18 2024

@author: rya200
"""
# %% Libraries
from bad_ctmc import *
import os
import json
import gzip
import pickle
import multiprocessing
import numpy as np
from datetime import datetime

os.chdir(os.getcwd() + "/code")


start_time = datetime.now()

mc_cores = os.environ["SLURM_NTASKS"]
mc_cores = int(mc_cores)
array_val = os.environ["SLURM_ARRAY_TASK_ID"]
array_val = int(array_val)

print("Starting job ", array_val)

# %%
phantom_counter = 0
parent_directory = "../data/simulations"

try:
    os.mkdir(parent_directory)
except:
    phantom_counter += 1

child_directory = parent_directory + "/within_OR"

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

# save_file = child_directory + \
#     f"/baseline_simulations_seed_{simulation_parameters['seed']}_tend_{simulation_parameters['t_end']}_trajectories_{simulation_parameters['num_trajectory']}.json"

# %%


def run_simulation_with_event(tmp,
                              simulation_parameters=simulation_parameters,
                              child_directory=child_directory):

    # if strength==1:
    #     return "Done"
    OR = round(tmp[0], 3)
    p = round(tmp[1][0], 3)
    c = round(tmp[1][1], 3)

    save_name = f"/OR_{OR}_p_{p}_seed_{simulation_parameters['seed']}_tend_{simulation_parameters['t_end']}_trajectories_{simulation_parameters['num_trajectory']}.gz"

    save_file = child_directory + save_name
    save_file_model = child_directory_model + save_name

    if os.path.exists(save_file):
        return "Done"

    # Make Bstar the steady state
    tmp_params = dict(simulation_parameters["params"])
    tmp_params["inf_B_efficacy"] = p
    tmp_params["susc_B_efficacy"] = c

    model = bad_ctmc(param_vals=tmp_params,
                     P=simulation_parameters["P"],
                     I0=simulation_parameters["I0"],
                     B0=simulation_parameters["B0"],
                     t_end=simulation_parameters["t_end"])

    results = model.run(number_of_trajectories=500,  # simulation_parameters["num_trajectory"],
                        seed=simulation_parameters["seed"],
                        solver=gillespy2.solvers.TauHybridCSolver)

    # with open(save_file, "w") as f:
    #     json.dump(results.to_json(), f)
    # f.close()
    with gzip.open(save_file, "wb") as f:
        f.write(compress_data(results.to_json()))
    f.close()

    with open(save_file_model, "wb") as f:
        pickle.dump(model, f)
    f.close()

    return "Done"


# %%
if __name__ == '__main__':

    params = simulation_parameters["params"]

    gamma = 1/params["infectious_period"]
    beta = params["transmission"]

    pi = beta/(beta + gamma)

    k = [0.1, 0.2, 0.3, 0.6]
    p = np.arange(0, 1, step=0.02)
    p_save = []

    c_min = 0.
    c = []

    idxx = []

    for ii, OR in enumerate(k):
        A = OR/(1-(1-OR)*pi) - 1

        c_tmp = 1-((gamma * (A+1)) / ((1-p)*(gamma - beta * A)))

        idx = next(i for i, cc in enumerate(c_tmp) if cc < c_min)
        idxx.append(idx)

        c.append(c_tmp[:idx])
        p_save.append(p[:idx])

    int_params = [(x, y) for i, x in enumerate(k)
                  for y in zip(p_save[i], c[i])]

    step_size = 50

    start_index = ((array_val-1)*step_size)
    end_index = array_val * step_size
    if end_index > len(int_params):
        end_index = len(int_params)

    subset_params = int_params[start_index:end_index]

    with multiprocessing.Pool(mc_cores) as p:
        p.map(run_simulation_with_event, subset_params)

    print(f"Time taken: {datetime.now()-start_time}")
