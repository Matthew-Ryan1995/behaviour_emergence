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
import pickle
import multiprocessing
import numpy as np
from datetime import datetime

start_time = datetime.now()
# %%
phantom_counter = 0
parent_directory = "../data/simulations"

try:
    os.mkdir(parent_directory)
except:
    phantom_counter += 1

child_directory = parent_directory + "/Bstar_by_R0"

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
    Bstar = round(tmp[0], 3)
    R0 = round(tmp[1], 3)

    save_name = f"/Bstar_{Bstar}_R0_{R0}_seed_{simulation_parameters['seed']}_tend_{simulation_parameters['t_end']}_trajectories_{simulation_parameters['num_trajectory']}.json"

    save_file = child_directory + save_name
    save_file_model = child_directory_model + save_name

    if os.path.exists(save_file):
        return "Done"

    B0 = int(Bstar * simulation_parameters["P"]) - simulation_parameters["I0"]

    # Make Bstar the steady state
    tmp_params = dict(simulation_parameters["params"])
    w1 = get_w1(Bstar, tmp_params)
    tmp_params["B_social"] = w1

    tmp_params["transmission"] = R0/tmp_params["infectious_period"]

    model = bad_ctmc(param_vals=tmp_params,
                     P=simulation_parameters["P"],
                     I0=simulation_parameters["I0"],
                     B0=B0,
                     t_end=simulation_parameters["t_end"])

    results = model.run(number_of_trajectories=simulation_parameters["num_trajectory"],
                        seed=simulation_parameters["seed"])

    with open(save_file, "w") as f:
        json.dump(results.to_json(), f)
    f.close()

    with open(save_file_model, "wb") as f:
        pickle.dump(model, f)
    f.close()

    return "Done"


if __name__ == '__main__':

    Bstar_min = 0.01
    Bstar_max = 0.98
    Bstar_step = 0.05

    Bstar = np.arange(start=Bstar_min,
                      stop=Bstar_max,  # Code freaks out when we move past 0.98
                      step=Bstar_step)

    R0_min = 0.25
    R0_max = 10.25
    R0_step = 0.1

    R0 = np.arange(start=R0_min,
                   stop=R0_max,
                   step=R0_step)

    R0 = np.append(R0, 3.28)

    int_params = [(x, y) for x in Bstar for y in R0]
    with multiprocessing.Pool(6) as p:
        p.map(run_simulation_with_event, int_params)

    print(f"Time taken: {datetime.now()-start_time}")
