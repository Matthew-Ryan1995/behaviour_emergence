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

# %%

parent_directory = "../data/simulations"

try:
    os.mkdir(parent_directory)
except:
    print()

child_directory = parent_directory + "/vary_Bstar"

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

# save_file = child_directory + \
#     f"/baseline_simulations_seed_{simulation_parameters['seed']}_tend_{simulation_parameters['t_end']}_trajectories_{simulation_parameters['num_trajectory']}.json"

# %%


def run_simulation_with_event(Bstar,
                              simulation_parameters=simulation_parameters,
                              child_directory=child_directory):

    # if strength==1:
    #     return "Done"
    Bstar = round(Bstar, 3)

    save_name = f"/Bstar_{Bstar}_seed_{simulation_parameters['seed']}_tend_{simulation_parameters['t_end']}_trajectories_{simulation_parameters['num_trajectory']}.json"

    save_file = child_directory + save_name
    save_file_model = child_directory_model + save_name

    if os.path.exists(save_file):
        return "Done"

    B0 = int(Bstar * simulation_parameters["P"]) - simulation_parameters["I0"]

    model = bad_ctmc(param_vals=simulation_parameters["params"],
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

    Bstar_min = 0.05
    Bstar_max = 1
    Bstar_step = 0.05

    Bstar = np.arange(start=Bstar_min, stop=Bstar_max +
                      Bstar_step, step=Bstar_step)

    with multiprocessing.Pool(6) as p:
        p.map(run_simulation_with_event, Bstar)
