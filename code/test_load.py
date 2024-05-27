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
from BaD import *
import pandas as pd
import ast
import gzip

baseline_path = "../data/simulations/baseline/"
baseline_files = next(os.walk(baseline_path), (None, None, []))[2]

# Remove unwanted files
if ".DS_Store" in baseline_files:
    baseline_files.remove(".DS_Store")

with open(baseline_path + baseline_files[0], "r") as f:
    test_load = json.load(f)


def compress_data(data):
    """
    https://gist.github.com/LouisAmon/4bd79b8ab80d3851601f3f9016300ac4

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.

    Returns
    -------
    compressed : TYPE
        DESCRIPTION.

    """
    # Convert to JSON
    json_data = json.dumps(data, indent=2)
    # Convert to bytes
    encoded = json_data.encode('utf-8')
    # Compress
    compressed = gzip.compress(encoded)
    return compressed


with open("../text.gz", "wb") as f:
    f.write(compress_data(test_load))
f.close()

with gzip.open("../data/simulations/baseline/baseline_simulations_p_0.6_c_0.45_OR_0.22_seed_20240430_tend_200_trajectories_100.gz",
               "rb") as f:
    json_bytes = f.read()

json_str = json_bytes.decode('utf-8')
tmp = json.loads(json_str)


test_load_dict = gillespy2.core.jsonify.Jsonify.from_json(tmp)

# %%
# with open(baseline_path + baseline_files[0], "r") as f:
#     test_json = json.load(f)
#     test_load = pd.read_json(test_json)
# f.close()

# test_load.to_csv("../test.csv")

# tmp = test_load.to_json()
# tmp_dict = json.loads(tmp)
# test_load_dict = gillespy2.core.jsonify.Jsonify.from_dict(tmp_dict)

# with open("../data/simulations/model/baseline_simulations_seed_20240430_tend_200_trajectories_10.json", "rb") as f:
#     tmp = pickle.load(f)
# f.close()

# # %%
# plt.figure()
# for idx in range(len(test_load_dict)):
#     trajectory = test_load_dict[idx]
#     plt.plot(trajectory["time"], trajectory["In"] + trajectory["Ib"])
#     plt.plot(trajectory["time"], trajectory["I_total"])
# plt.show()

# # %% Distribution at day 5

# I_snapshot = []

# for idx in range(len(test_load_dict)):
#     trajectory = test_load_dict[idx]
#     I = trajectory["In"] + trajectory["Ib"]
#     I_snapshot.append(I[4])

# plt.figure()
# plt.hist(I_snapshot)
# plt.show()

# params = load_param_defaults()

# R0 = 4

# params["B_social"] = 0.
# params["B_const"] = 0
# params["B_fear"] = 4

# params["immune_period"] = 0.1

# params["inf_B_efficacy"] = 0
# params["susc_B_efficacy"] = 0

# params["transmission"] = R0 / params["infectious_period"]

# P = 1
# I0 = 1e-6
# B0 = 1e-6

# IC = [P - I0 - B0, B0, I0, 0, 0, 0]

# t_start, t_end = [0, 300]

# M = bad(**params)

# M.run(IC=IC, t_start=t_start, t_end=t_end)

# # %%
# plt.figure()
# plt.plot(M.t_range, M.get_I(), label="Infections")
# plt.plot(M.t_range, M.get_B(), label="Behaviour")
# plt.legend()
# plt.show()

# print(M.get_I()[-1])
