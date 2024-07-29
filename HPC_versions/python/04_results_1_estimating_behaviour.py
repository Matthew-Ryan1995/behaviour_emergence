#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 16:19:56 2024

@author: rya200
"""
# %% Libraries
import json
import gzip
import gillespy2
import os
from BaD import *
import matplotlib.pyplot as plt

os.chdir(os.getcwd() + "/code")


params = {"ytick.color": "black",
          "xtick.color": "black",
          "axes.labelcolor": "black",
          "axes.edgecolor": "black",
          # "text.usetex": True,
          "font.family": "serif"}
plt.rcParams.update(params)
plt.rcParams['mathtext.fontset'] = 'dejavuserif'

# %% Get and laod data
file_path = "../data/simulations/baseline/"

filenames = next(os.walk(file_path), (None, None, []))[2]  # [] if no file

with open("../data/simulation_parameters.json", "r") as f:
    simulation_parameters = json.load(f)
f.close()

# %%

num_trajectories = 1000
OR = 0.22


target_file = f"trajectories_{num_trajectories}"

filenames = [f for idx, f in enumerate(filenames) if target_file in f]

target_file = f"OR_{OR}"

filenames = [f for idx, f in enumerate(filenames) if target_file in f]

if len(filenames) > 1:
    # p = 0.75
    c = 0.45
    target_file = f"c_{c}"

    filenames = [f for idx, f in enumerate(filenames) if target_file in f]


with gzip.open(file_path + filenames[0], "rb") as f:
    results_json_compressed = f.read()
    results_json = gzip.decompress(results_json_compressed)
    # results_json = results_json_compressed.decode("utf-8")
    # results_json = json.loads(results_json)
f.close()

# with gzip.open("../text.gz", "rb") as f:
#     json_bytes = f.read()

# json_str = json_bytes.decode('utf-8')
# tmp = json.loads(json_str)


results = gillespy2.core.jsonify.Jsonify.from_json(results_json)

save_name = filenames[0]
save_name = save_name.replace("baseline_simulatims_", "")
save_name = save_name.replace(".json", "")
save_name = save_name.replace(".gz", "")

# %% Plot parameters

dpi = 600
font_size = 16

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
    I = (trajectory["In"] + trajectory["Ib"]) / P
    plt.plot(trajectory["time"], B, color="blue", alpha=0.2)
    plt.plot(trajectory["time"], I, color="red", alpha=0.2)

# Plot/demonstrate early time approximations
exp_approx = early_behaviour_dynamics(M)
tt = [i for i in range(len(exp_approx)) if exp_approx[i] < 1]
plt.plot(range(tt[-1] + 1), exp_approx[tt],
         linestyle="dashed", color="black", linewidth=3)

cubic_approx = early_behaviour_dynamics(M, method="poly", M=3)
tt = [i for i in range(len(cubic_approx)) if cubic_approx[i] < 1]
plt.plot(range(tt[-1] + 1), cubic_approx[tt],
         linestyle="dotted", color="black",  linewidth=3)

no_I_approx = early_behaviour_dynamics(M, method="none")
tt = [i for i in range(len(no_I_approx)) if no_I_approx[i] < 1]
plt.plot(range(tt[-1] + 1), no_I_approx[tt],
         linestyle="dashdot", color="black",  linewidth=3)

plt.xlabel("Time (days)", fontsize=font_size)
plt.ylabel("Prevalence", fontsize=font_size)
plt.xticks(fontsize=font_size)
plt.yticks(fontsize=font_size)
# plt.legend(["Behaviour", "Infection"])
# plt.title("Dashed line - exponential\nDotted line - Cubic")
plt.savefig(f"../figs/results1_{save_name}_timeseries.png",
            dpi=dpi,
            bbox_inches="tight")
plt.close()

# todo: Add save

# %% Figure 2: Snapshot at day 10

snapshot_day = 10 - 1
y_coord = 40

marker_size = 10
linewidth = 3

B_snapshot = []
I_snapshot = []
B_median = []
# I_final = []
for idx in range(num_trajectory):
    trajectory = results[idx]
    B = (trajectory["Sb"] + trajectory["Ib"] + trajectory["Rb"]) / P
    B_snapshot.append(B[snapshot_day])
    I = (trajectory["I_total"]) / P
    I_snapshot.append(I[snapshot_day])
    # I_final.append(I[-1])
    if I[-1] > 20/P:
        B_median.append(B[snapshot_day])


# Plot/demonstrate early time approximations
exp_approx = early_behaviour_dynamics(M)
cubic_approx = early_behaviour_dynamics(M, method="poly", M=3)


plt.figure()
plt.hist(B_snapshot, bins=30,
         edgecolor="black", color="white")

plt.plot([np.median(B_median), np.median(B_median)], [0, 80], "blue",
         linewidth=linewidth)

plt.plot([exp_approx[snapshot_day], exp_approx[snapshot_day]],
         [y_coord, y_coord], "x", color="black", markersize=marker_size)
plt.plot([exp_approx[snapshot_day], exp_approx[snapshot_day]],
         [0, y_coord], linestyle="dashed", color="black", linewidth=linewidth,
         label="exponential")

plt.plot([cubic_approx[snapshot_day], cubic_approx[snapshot_day]],
         [y_coord, y_coord], "D", color="black",  markersize=marker_size)
plt.plot([cubic_approx[snapshot_day], cubic_approx[snapshot_day]],
         [0, y_coord], linestyle="dotted", color="black", linewidth=linewidth,
         label="cubic")

plt.plot([no_I_approx[snapshot_day], no_I_approx[snapshot_day]],
         [y_coord, y_coord], "*", color="black",  markersize=marker_size)
plt.plot([no_I_approx[snapshot_day], no_I_approx[snapshot_day]],
         [0, y_coord], linestyle="dashdot", color="black", linewidth=linewidth,
         label="no infection")


plt.xlabel(
    f"Prevalence of behaviour on day {snapshot_day + 1}", fontsize=font_size)
plt.ylabel("Frequency", fontsize=font_size)
plt.xticks(fontsize=font_size)
plt.yticks(fontsize=font_size)

plt.legend()

plt.savefig(f"../figs/results1_{save_name}_behaviour_snapshot.png",
            dpi=dpi,
            bbox_inches="tight")
plt.close()

# plt.hist(I_final, bins = 300)

# %% Figure S1: Behaviour estimates against simulations
plt.figure()
for idx in range(num_trajectory):
    trajectory = results[idx]
    B = (trajectory["Sb"] + trajectory["Ib"] + trajectory["Rb"]) / P
    I = (trajectory["In"] + trajectory["Ib"]) / P
    I_tot = trajectory["I_total"]
    if I_tot[-1] > 20:
        plt.plot(trajectory["time"], B, color="grey", alpha=0.2)
        plt.plot(trajectory["time"], I, color="black", alpha=0.2)


linear_approx = early_behaviour_dynamics(M, method="poly", M=1)
tt = [i for i in range(len(linear_approx)) if linear_approx[i] < 1]
plt.plot(range(tt[-1] + 1), linear_approx[tt],
         label="linear", linewidth=3)

quadratic_approx = early_behaviour_dynamics(M, method="poly", M=2)
tt = [i for i in range(len(quadratic_approx)) if quadratic_approx[i] < 1]
plt.plot(range(tt[-1] + 1), quadratic_approx[tt],
         label="quadratic",  linewidth=3)

cubic_approx = early_behaviour_dynamics(M, method="poly", M=3)
tt = [i for i in range(len(cubic_approx)) if cubic_approx[i] < 1]
plt.plot(range(tt[-1] + 1), cubic_approx[tt],
         label="cubic",  linewidth=3)

quartic_approx = early_behaviour_dynamics(M, method="poly", M=4)
tt = [i for i in range(len(quartic_approx)) if quartic_approx[i] < 1]
plt.plot(range(tt[-1] + 1), quartic_approx[tt],
         label="quartic",  linewidth=3)


plt.xlabel("Time (days)", fontsize=font_size)
plt.ylabel("Prevalence", fontsize=font_size)
plt.xticks(fontsize=font_size)
plt.yticks(fontsize=font_size)
plt.legend()
# plt.legend(["Behaviour", "Infection"])
# plt.title("Dashed line - exponential\nDotted line - Cubic")
plt.savefig(f"../figs/supp_results1_{save_name}_timeseries.png",
            dpi=dpi,
            bbox_inches="tight")
plt.close()

# todo: Add save

# %% Figure S2: Snapshot at day 10

snapshot_day = 10 - 1
y_coord = 40

marker_size = 10
linewidth = 3

B_snapshot = []
I_snapshot = []
for idx in range(num_trajectory):
    trajectory = results[idx]
    B = (trajectory["Sb"] + trajectory["Ib"] + trajectory["Rb"]) / P
    B_snapshot.append(B[snapshot_day])
    I = (trajectory["I_total"]) / P
    I_snapshot.append(I[snapshot_day])
    if I[-1] > 20/P:
        B_median.append(B[snapshot_day])


# Plot/demonstrate early time approximations
linear_approx = early_behaviour_dynamics(M, method="poly", M=1)
quadratic_approx = early_behaviour_dynamics(M, method="poly", M=2)
cubic_approx = early_behaviour_dynamics(M, method="poly", M=3)
quartic_approx = early_behaviour_dynamics(M, method="poly", M=4)


plt.figure(figsize=(8, 6))
plt.hist(B_snapshot, bins=30,
         edgecolor="black", color="white")

# plt.plot([exp_approx[snapshot_day], exp_approx[snapshot_day]],
#          [y_coord, y_coord], "x", color="black", markersize=marker_size)
# plt.plot([exp_approx[snapshot_day], exp_approx[snapshot_day]],
#          [0, y_coord], linestyle="dashed", color="black", linewidth=linewidth,
#          label="exponential")

plt.plot([np.median(B_median), np.median(B_median)], [0, 80], "blue",
         linewidth=linewidth)


plt.plot([linear_approx[snapshot_day], linear_approx[snapshot_day]],
         [0, y_coord], linewidth=linewidth,
         label="linear")
plt.plot([linear_approx[snapshot_day], linear_approx[snapshot_day]],
         [y_coord, y_coord], "o", color="black",  markersize=marker_size)


plt.plot([quadratic_approx[snapshot_day], quadratic_approx[snapshot_day]],
         [0, y_coord], linewidth=linewidth,
         label="quadratic")
plt.plot([quadratic_approx[snapshot_day], quadratic_approx[snapshot_day]],
         [y_coord, y_coord], "o", color="black",  markersize=marker_size)


plt.plot([cubic_approx[snapshot_day], cubic_approx[snapshot_day]],
         [0, y_coord], linewidth=linewidth,
         label="cubic")
plt.plot([cubic_approx[snapshot_day], cubic_approx[snapshot_day]],
         [y_coord, y_coord], "o", color="black",  markersize=marker_size)


plt.plot([quartic_approx[snapshot_day], quartic_approx[snapshot_day]],
         [0, y_coord], linewidth=linewidth,
         label="quartic")
plt.plot([quartic_approx[snapshot_day], quartic_approx[snapshot_day]],
         [y_coord, y_coord], "o", color="black",  markersize=marker_size)

# plt.plot([no_I_approx[snapshot_day], no_I_approx[snapshot_day]],
#          [y_coord, y_coord], "*", color="black",  markersize=marker_size)
# plt.plot([no_I_approx[snapshot_day], no_I_approx[snapshot_day]],
#          [0, y_coord], linestyle="dashdot", color="black", linewidth=linewidth,
#          label="no infection")

plt.plot([np.median(B_median), np.median(B_median)], [0, 80], "grey",
         linewidth=linewidth)

plt.xlabel(
    f"Prevalence of behaviour on day {snapshot_day + 1}", fontsize=font_size)
plt.ylabel("Frequency", fontsize=font_size)
plt.xticks(fontsize=font_size)
plt.yticks(fontsize=font_size)

plt.legend()

plt.savefig(f"../figs/supp_results1_{save_name}_behaviour_snapshot.png",
            dpi=dpi,
            bbox_inches="tight")
plt.close()
