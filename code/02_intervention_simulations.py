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
from datetime import datetime

start_time = datetime.now()

# %%
phantom_counter = 0
parent_directory = "../data/simulations"

try:
    os.mkdir(parent_directory)
except:
    phantom_counter += 1

child_directory = parent_directory + "/interventions"

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


def run_simulation_with_event(inter_scenario,
                              simulation_parameters=simulation_parameters,
                              child_directory=child_directory):
    target = inter_scenario[0]
    day = inter_scenario[1]
    strength = inter_scenario[2]

    # if strength==1:
    #     return "Done"

    save_name = f"/target_{target}_day_{day}_strength_{strength}_seed_{simulation_parameters['seed']}_tend_{simulation_parameters['t_end']}_trajectories_{simulation_parameters['num_trajectory']}.json"

    save_file = child_directory + save_name
    save_file_model = child_directory_model + save_name

    if os.path.exists(save_file):
        return "Done"

    event = {
        "strength": strength,
        "target": target,
        "day": day
    }

    model = bad_ctmc(param_vals=simulation_parameters["params"],
                     P=simulation_parameters["P"],
                     I0=simulation_parameters["I0"],
                     B0=simulation_parameters["B0"],
                     t_end=simulation_parameters["t_end"],
                     event=event)

    results = model.run(number_of_trajectories=simulation_parameters["num_trajectory"],
                        seed=simulation_parameters["seed"])

    with open(save_file, "w") as f:
        json.dump(results.to_json(), f)
    f.close()

    with open(save_file_model, "wb") as f:
        pickle.dump(model, f)
    f.close()

    return "Done"


with open("../data/intervention_parameters.json", "r") as f:
    intervention_params = json.load(f)
f.close()

if __name__ == '__main__':
    with multiprocessing.Pool(6) as p:
        p.map(run_simulation_with_event, intervention_params)

    print(f"Time taken: {datetime.now()-start_time}")
