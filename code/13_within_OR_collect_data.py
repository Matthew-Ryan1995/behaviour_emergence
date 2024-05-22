#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 08:17:47 2024

@author: rya200
"""
# %% libraries

import json
import gillespy2
import os
from BaD import *
from bad_ctmc import *
import pandas as pd
import numpy as np
import re
from scipy.integrate import simpson

# %% Get filepaths

file_path = "../data/simulations/within_OR/"

filenames = next(os.walk(file_path), (None, None, []))[2]  # [] if no file

with open("../data/simulation_parameters.json", "r") as f:
    simulation_parameters = json.load(f)
f.close()

# Remove unwanted files
if ".DS_Store" in filenames:
    filenames.remove(".DS_Store")

# %% Helper functions


def find_median_curve(dlr, outbreaks, P):
    B_med = np.zeros(len(dlr[0]["time"]))
    cc = 0

    for idx in range(len(dlr)):
        if outbreaks[idx] > 0:
            cc += 1
            trajectory = dlr[idx]
            B = (trajectory["Sb"] + trajectory["Ib"] + trajectory["Rb"]) / P
            B_med = np.column_stack((B_med, B))
    if cc > 0:
        B_med = B_med[:, 1:]
        B_med = np.median(B_med, axis=1)

    return B_med


def load_bstar_data(file, file_path="../data/simulations/within_OR/",
                    P=simulation_parameters["P"], params=simulation_parameters):
    # Load data
    with open(file_path + file, "r") as f:
        results_json = json.load(f)
    f.close()
    results = gillespy2.core.jsonify.Jsonify.from_json(results_json)

    # Probability of outbreak
    outbreak_occurs = np.array([get_outbreak(x, P=P) for x in results])

    outbreak_probability = outbreak_occurs.sum()/outbreak_occurs.size

    outbreak_probability_std = np.sqrt(
        outbreak_probability * (1-outbreak_probability) / outbreak_occurs.size)

    # Final size
    fs = np.array([x["I_total"][-1] for x in results])/P

    fs_mean = fs.mean()

    fs_std = fs.std()/np.sqrt(fs.size)

    # FS when an outbreak occurs
    fs_conditional = np.array(
        [x["I_total"][-1] for idx, x in enumerate(results) if outbreak_occurs[idx] > 0])/P
    if fs_conditional.size > 0:
        fs_conditional_mean = fs_conditional.mean()
        fs_conditional_std = fs_conditional.std()/np.sqrt(fs_conditional.size)
    else:
        fs_conditional_mean = 0
        fs_conditional_std = 0

    # BStar parameter
    extracted_params = re.findall("0.[0-9]*",  file)
    OR = np.float32(extracted_params[0])
    p = np.float32(extracted_params[1])

    # Get errors
    B = find_median_curve(results, outbreak_occurs, P=P)

    tmp_params = dict(params["params"])

    I0 = params["I0"]
    B0 = params["B0"]

    N0 = P - B0

    M = bad(**tmp_params)

    Sn = P - I0 - B0
    IC = np.array([Sn, B0, I0, 0, 0, 0]) / P
    t_start, t_end = [0, params["t_end"]]
    M.run(IC=IC, t_start=t_start, t_end=t_end)

    exp_approx = early_behaviour_dynamics(M)

    poly_approx = early_behaviour_dynamics(M, method="poly", M=3)

    error_days = [5, 10, 15]
    exp_errors = []
    poly_errors = []
    for i, d in enumerate(error_days):
        tt = np.arange(0, d, step=1)
        exp_errors.append(simpson((exp_approx[tt] - B[tt])**2, x=tt))
        poly_errors.append(simpson((poly_approx[tt] - B[tt])**2, x=tt))

    # Create answers

    ans = {
        "OR": OR,
        "p": p,
        "pHat": outbreak_probability,
        "pHat_std": outbreak_probability_std,
        "FS_avg": fs_mean,
        "FS_std": fs_std,
        "FS_conditional": fs_conditional_mean,
        "FS_conditional_std": fs_conditional_std
    }

    for i, d in enumerate(error_days):
        ans[f"exp_error_{d}"] = exp_errors[i]
        ans[f"poly_error_{d}"] = poly_errors[i]
    return ans


# %% Load and save data
df = list(map(load_bstar_data, filenames))
df = pd.DataFrame(df)

df = df.sort_values(by=["OR", "p"])

df.to_csv("../data/df_within_OR.csv")

# %%

# P = 10000
# I0 = 1
# B0 = 1

# N0 = P - B0

# M = bad(**params)

# Sn = P - I0 - B0
# IC = np.array([Sn, B0, I0, 0, 0, 0]) / P
# t_start, t_end = [0, 100]
# M.run(IC=IC, t_start=t_start, t_end=t_end)

# res = []

# tmp = early_behaviour_dynamics(M)
# tt = [i for i in range(len(tmp)) if tmp[i] < 1]

# for idx in range(num_trajectory):
#     trajectory = results[idx]
#     B = (trajectory["Sb"] + trajectory["Ib"] + trajectory["Rb"])/P
#     res.append(simpson((tmp[tt] - B[tt])**2, x=tt))

# print(np.mean(res))
# res2 = []

# tmp2 = early_behaviour_dynamics(M, method="poly", M=3)
# tt2 = [i for i in range(len(tmp2)) if tmp2[i] < 1]


# for idx in range(num_trajectory):
#     trajectory = results[idx]
#     B = (trajectory["Sb"] + trajectory["Ib"] + trajectory["Rb"])/P
#     res2.append(simpson((tmp2[tt] - B[tt])**2, x=tt))
# print(np.mean(res2))
