#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 16:19:56 2024

@author: rya200
"""
# %% Libraries
import json
import gillespy2
import os
from BaD import *
import matplotlib.pyplot as plt

# %% Get and laod data
file_path = "../data/simulations/baseline/"

filenames = next(os.walk(file_path), (None, None, []))[2]  # [] if no file

with open("../data/simulation_parameters.json", "r") as f:
    simulation_parameters = json.load(f)
f.close()

with open(file_path + filenames[0], "r") as f:
    results_json = json.load(f)
f.close()

results = gillespy2.core.jsonify.Jsonify.from_json(results_json)
# %% Set up parameters

P = simulation_parameters["P"]
I0 = simulation_parameters["I0"]
B0 = simulation_parameters["B0"]
num_trajectory = len(results)

Sn = P - I0 - B0

IC = np.array([Sn, B0, I0, 0, 0, 0]) / P
t_start, t_end = [0, simulation_parameters["t_end"]]

params = simulation_parameters["params"]

# Define and calcualte the ODE approximation
M = bad(**params)
M.run(IC=IC, t_start=t_start, t_end=t_end)

# %% Figure 1: Behaviour estimates against simulations
plt.figure()
for idx in range(num_trajectory):
    trajectory = results[idx]
    B = (trajectory["Sb"] + trajectory["Ib"] + trajectory["Rb"]) / P
    plt.plot(trajectory["time"], B, color="blue", alpha=0.2)

# Plot/demonstrate early time approximations
exp_approx = early_behaviour_dynamics(M)
tt = [i for i in range(len(exp_approx)) if exp_approx[i] < 1]
plt.plot(range(tt[-1] + 1), exp_approx[tt],
         linestyle="dashed", color="black")

cubic_approx = early_behaviour_dynamics(M, method="poly", M=3)
tt = [i for i in range(len(cubic_approx)) if cubic_approx[i] < 1]
plt.plot(range(tt[-1] + 1), cubic_approx[tt],
         linestyle="dotted", color="black")

plt.xlabel("time")
plt.ylabel("Proportion")
plt.legend(["Behaviour"])
plt.title("Dashed line - exponential\nDotted line - Cubic")
plt.show()

# todo: Add save

# %% Figure 2: Snapshot at day 10

snapshot_day = 10 - 1

B_snapshot = []
I_snapshot = []
for idx in range(num_trajectory):
    trajectory = results[idx]
    B = (trajectory["Sb"] + trajectory["Ib"] + trajectory["Rb"]) / P
    B_snapshot.append(B[snapshot_day])
    I = (trajectory["Ib"] + trajectory["In"]) / P
    I_snapshot.append(I[snapshot_day])


# Plot/demonstrate early time approximations
exp_approx = early_behaviour_dynamics(M)
cubic_approx = early_behaviour_dynamics(M, method="poly", M=3)


plt.figure()
plt.hist(B_snapshot, bins=len(B_snapshot))
plt.plot([exp_approx[snapshot_day], exp_approx[snapshot_day]],
         [1, 1], "x", color="red")
plt.plot([cubic_approx[snapshot_day], cubic_approx[snapshot_day]],
         [1, 1], "x", color="black")
plt.xlabel("Proportion doing behaviour")
plt.ylabel("Frequency")
plt.show()


plt.figure()
plt.hist(I_snapshot, bins=len(I_snapshot))
plt.xlabel("Proportion infected")
plt.ylabel("Frequency")
plt.show()
