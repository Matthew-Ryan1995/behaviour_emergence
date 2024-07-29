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

    save_name = f"/target_{target}_day_{day}_strength_{strength}_seed_{simulation_parameters['seed']}_tend_{simulation_parameters['t_end']}_trajectories_{simulation_parameters['num_trajectory']}.gz"

    save_file = child_directory + save_name
    save_file_model = child_directory_model + save_name

    if os.path.exists(save_file):
        return "Done"

    if strength == 1:
        event = {}
    else:
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


with open("../data/intervention_parameters.json", "r") as f:
    intervention_params = json.load(f)
f.close()

step_size = 50

start_index = ((array_val-1)*step_size)
end_index = array_val * step_size
if end_index > len(intervention_params):
    end_index = len(intervention_params)

subset_params = intervention_params[start_index:end_index]

if __name__ == '__main__':
    with multiprocessing.Pool(mc_cores) as p:
        p.map(run_simulation_with_event, subset_params)

    print(f"Time taken: {datetime.now()-start_time}")
# %%
# array_num = 1
# step_size = 500

#
