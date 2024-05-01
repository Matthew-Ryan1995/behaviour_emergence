#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 12:50:00 2024

Create the parameter set and the initial conditions for all subsequent simulations


@author: rya200
"""
# %% Libraries
from BaD import load_param_defaults
import json
import numpy as np

# %%

simulation_parameters = dict()

R0 = 3
R0B = 0.9
infectious_period = 5


# Load and alter parameters
params = load_param_defaults()

params["immune_period"] = 0  # No waning immunity

params["infectious_period"] = infectious_period
params["transmission"] = R0/params["infectious_period"]


params["B_fear"] = params["B_fear"]/params["infectious_period"]
params["B_const"] = params["B_const"] / params["infectious_period"]

params["N_social"] = params["N_social"] / params["infectious_period"]
params["N_const"] = params["N_const"] / params["infectious_period"]

params["B_social"] = (R0B * (params["N_social"] + params["N_const"]))


# Set up initial conditions
P = 10000  # population size
I0 = 1  # Initial infected
B0 = 1  # Initial Behaviour
num_trajectory = 100

# Number of simulation days heuristically chosen
t_end = 200

# Seed for simulations

seed = 20240430

simulation_parameters["params"] = params
simulation_parameters["P"] = P
simulation_parameters["I0"] = I0
simulation_parameters["B0"] = B0
simulation_parameters["num_trajectory"] = num_trajectory
simulation_parameters["t_end"] = t_end
simulation_parameters["seed"] = seed

with open("../data/simulation_parameters.json", "w") as f:
    json.dump(simulation_parameters, f)
f.close()

# %% Intervention parameters

int_start = 0
int_stop = 5
int_step = 0.5  # Change to get finer grain

strength = np.arange(start=int_start, stop=int_stop + int_step, step=int_step)
target = ["w1", "w2", "w3"]
day = [5, 10, 15]

params = [(x, y, z) for x in target for y in day for z in strength]

with open("../data/intervention_parameters.json", "w") as f:
    json.dump(params, f)
f.close()
