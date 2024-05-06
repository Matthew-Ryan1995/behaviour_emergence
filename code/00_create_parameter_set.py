#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 12:50:00 2024

Create the parameter set and the initial conditions for all subsequent simulations

Epi params taken from https://doi.org/10.1186/s40001-023-01047-0

Behaviour params taken from Ryan2024

w1 chosen such that B* = 0.03, value taken from https://covid19.healthdata.org/global for Aust. mask wearing


@author: rya200
"""
# %% Libraries
from BaD import load_param_defaults
from bad_ctmc import get_w1
import json
import numpy as np

# %%

simulation_parameters = dict()

R0 = 3.28
# R0B = 0.9
B_star_min = 0.001
infectious_period = 7


# Load and alter parameters
params = load_param_defaults()

params["immune_period"] = 0  # No waning immunity

params["infectious_period"] = infectious_period
params["transmission"] = R0/params["infectious_period"]


params["B_fear"] = params["B_fear"]/params["infectious_period"]
# params["B_const"] = params["B_const"] / params["infectious_period"]

params["N_social"] = params["N_social"] / params["infectious_period"]
params["N_const"] = params["N_const"] / params["infectious_period"]

params["B_const"] = B_star_min * params["N_const"] / (1-B_star_min)

# params["B_social"] = (R0B * (params["N_social"] + params["N_const"]))

Bstar = 0.03
w1 = get_w1(Bstar, params)
params["B_social"] = w1


# Set up initial conditions
P = 5000  # population size, chosen for speed of simulations
I0 = 1  # Initial infected
B0 = 1  # Initial Behaviour
num_trajectory = 100

# Number of simulation days heuristically chosen
# Most epidemics are completed by this time (number of infected individuals goes to 0)
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
int_step = 0.2  # Change to get finer grain

strength = np.arange(start=int_start, stop=int_stop + int_step, step=int_step)
target = ["w1", "w2", "w3"]
day = [5, 10, 15]

int_params = [(x, y, z) for x in target for y in day for z in strength]

with open("../data/intervention_parameters.json", "w") as f:
    json.dump(int_params, f)
f.close()
