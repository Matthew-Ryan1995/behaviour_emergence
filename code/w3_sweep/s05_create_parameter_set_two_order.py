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
from bad_ctmc import get_w1, get_w3
import json
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

params = {"ytick.color": "black",
          "xtick.color": "black",
          "axes.labelcolor": "black",
          "axes.edgecolor": "black",
          # "text.usetex": True,
          "font.family": "serif"}
plt.rcParams.update(params)
plt.rcParams['mathtext.fontset'] = 'dejavuserif'

dpi = 600
font_size = 16

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

params["B_const"] = get_w3(Bstar_min=B_star_min, params=params) * 100

# params["B_social"] = (R0B * (params["N_social"] + params["N_const"]))

Bstar = 0.13
w1 = get_w1(Bstar, params)
params["B_social"] = w1


# Calculate p and c

params["inf_B_efficacy"] = 0.75

gamma = 1/params["infectious_period"]
beta = params["transmission"]

pi = beta/(beta + gamma)

k = 0.22
A = k/(1-(1-k)*pi) - 1

c_min = 0.

# p = params["inf_B_efficacy"]
# p = np.arange(0, 1, step=0.01)
p = 0.6  # Around 0.6 gives the optimal results on reduced final size

c = round(1-((gamma * (A+1)) / ((1-p)*(gamma - beta * A))), 2)

# idx = next(i for i, cc in enumerate(c) if cc < c_min)

# idx = int(idx/2)

params["susc_B_efficacy"] = c  # [idx]
params["inf_B_efficacy"] = p  # [idx]

# Set up initial conditions
P = 5000  # population size, chosen for speed of simulations
I0 = 1  # Initial infected
B0 = 1  # Initial Behaviour
num_trajectory = 500

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
simulation_parameters["OR"] = k

with open("data/simulation_parameters_w3_by_100.json", "w") as f:
    json.dump(simulation_parameters, f)
f.close()

# %% Intervention parameters

int_start = 0
int_stop = 5
int_step = 0.5  # Change to get finer grain

strength = np.arange(start=int_start, stop=int_stop +
                     int_step, step=int_step).round(2)
target = ["w1", "w2", "w3"]
day = [5, 10, 15]

int_params = [(x, y, z) for x in target for y in day for z in strength]

with open("data/intervention_parameters_w3_by_100.json", "w") as f:
    json.dump(int_params, f)
f.close()

# %%

# beta = params["transmission"]
# gamma = 1/params["infectious_period"]

# pi = beta/(beta + gamma)
# plt.figure()

# kk = np.array([0.1, 0.2, 0.3, 0.6])
# # kk = np.array([0.22])
# p = np.arange(0, 1, step=0.01)
# # p = 0.9
# c = []
# idxes = []
# for i, k in enumerate(kk):
#     k = k.round(2)
#     A = k/(1-(1-k)*pi) - 1

#     c.append(1-((gamma * (A+1)) / ((1-p)*(gamma - beta * A))))

#     idx = next(ii for ii, cc in enumerate(c[i]) if cc < 0) + 1
#     idxes.append(idx)

#     plt.plot(p[:idx], c[i][:idx], label="OR: " + str(k))

# plt.xlabel("Infectious efficacy (p)", fontsize=font_size)
# plt.ylabel("Susceptible efficacy (c)", fontsize=font_size)

# plt.xticks(fontsize=font_size)
# plt.yticks(fontsize=font_size)

# plt.ylim(0, 1)
# plt.xlim(0, 1)
# # plt.plot([0.9, 0.9], [0, 1], ":k")
# # plt.plot([0.5, 0.5], [0, 1], ":k")
# plt.legend(loc=(1.01, 0.25), fontsize=font_size)
# plt.savefig("../figs/OR_vary_p_c.png", dpi=dpi,  bbox_inches="tight")
# plt.show()
