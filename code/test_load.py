#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 15:00:03 2024

@author: rya200
"""
import json
import gillespy2
import pickle
import os
import matplotlib.pyplot as plt

baseline_path = "../data/simulations/baseline/"
baseline_files = next(os.walk(baseline_path), (None, None, []))[2]

# Remove unwanted files
if ".DS_Store" in baseline_files:
    baseline_files.remove(".DS_Store")

with open(baseline_path + baseline_files[0], "r") as f:
    test_load = json.load(f)

test_load_dict = gillespy2.core.jsonify.Jsonify.from_json(test_load)

# with open("../data/simulations/model/baseline_simulations_seed_20240430_tend_200_trajectories_10.json", "rb") as f:
#     tmp = pickle.load(f)
# f.close()

# %%
plt.figure()
for idx in range(len(test_load_dict)):
    trajectory = test_load_dict[idx]
    plt.plot(trajectory["time"], trajectory["In"] + trajectory["Ib"])
    plt.plot(trajectory["time"], trajectory["I_total"])
plt.show()

# %% Distribution at day 5

I_snapshot = []

for idx in range(len(test_load_dict)):
    trajectory = test_load_dict[idx]
    I = trajectory["In"] + trajectory["Ib"]
    I_snapshot.append(I[4])

plt.figure()
plt.hist(I_snapshot)
plt.show()
