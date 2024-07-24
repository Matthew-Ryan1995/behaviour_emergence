#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 08:17:47 2024

@author: rya200
"""
# %% libraries

import json
import gzip
import gillespy2
import os
from BaD import *
from bad_ctmc import *
import pandas as pd
import numpy as np
import re

# %% Get filepaths

file_path = "../data/simulations/interventions/"

filenames = next(os.walk(file_path), (None, None, []))[2]  # [] if no file

with open("../data/simulation_parameters.json", "r") as f:
    simulation_parameters = json.load(f)
f.close()

# Remove unwanted files
if ".DS_Store" in filenames:
    filenames.remove(".DS_Store")

baseline_path = "../data/simulations/baseline/"
baseline_files = next(os.walk(baseline_path), (None, None, []))[2]

# Remove unwanted files
if ".DS_Store" in baseline_files:
    baseline_files.remove(".DS_Store")

# %%
num_trajectories = 1000

target_file = f"trajectories_{num_trajectories}"

filenames = [f for idx, f in enumerate(filenames) if target_file in f]
baseline_files = [f for idx, f in enumerate(
    baseline_files) if target_file in f]

# filenames = filenames[ids]

# %% Helper functions


def load_intervention_data(file, file_path="../data/simulations/interventions/",
                           P=simulation_parameters["P"]):
    # Load data
    # with open(file_path + file, "r") as f:
    #     results_json = json.load(f)
    # f.close()
    with gzip.open(file_path + file, "rb") as f:
        results_json_compressed = f.read()
        results_json = gzip.decompress(results_json_compressed)
    f.close()

    results = gillespy2.core.jsonify.Jsonify.from_json(results_json)

    # Probability of outbreak
    outbreak_occurs = np.array([get_outbreak(x, P=P) for x in results])

    n = outbreak_occurs.size

    outbreak_probability = outbreak_occurs.sum()/n

    outbreak_probability_std = np.sqrt(
        outbreak_probability * (1-outbreak_probability) / n)

    # Final size
    fs = np.array([x["I_total"][-1] for x in results]) / P

    fs_mean = fs.mean()

    fs_std = fs.std()/np.sqrt(n)

    # FS when an outbreak occurs
    fs_conditional = np.array(
        [x["I_total"][-1] for idx, x in enumerate(results) if outbreak_occurs[idx] > 0]) / P
    fs_conditional_mean = fs_conditional.mean()

    fs_conditional_std = fs_conditional.std()/np.sqrt(fs_conditional.size)

    # Parameter
    target = re.findall("w[1-3]",  file)[0]

    day = re.findall("day_[0-5]*",  file)[0]
    day = int(re.findall("[0-5]+",  day)[0])

    strength = re.findall("strength_[0-9].[0-9]*",  file)[0]
    strength = np.float32(re.findall("[0-9].[0-9]+",  strength)[0])

    ans = {
        "target": target,
        "day": day,
        "strength": strength,
        "num_trajectories": n,
        "pHat": outbreak_probability,
        "pHat_std": outbreak_probability_std,
        "FS_avg": fs_mean,
        "FS_std": fs_std,
        "FS_conditional": fs_conditional_mean,
        "FS_conditional_std": fs_conditional_std
    }
    return ans


def load_baseline_data(file, file_path="../data/simulations/baseline/",
                       P=simulation_parameters["P"]):
    # Load data
    # with open(file_path + file, "r") as f:
    #     results_json = json.load(f)
    # f.close()
    with gzip.open(file_path + file, "rb") as f:
        results_json_compressed = f.read()
        results_json = gzip.decompress(results_json_compressed)
    f.close()

    results = gillespy2.core.jsonify.Jsonify.from_json(results_json)

    # Probability of outbreak
    outbreak_occurs = np.array([get_outbreak(x, P=P) for x in results])

    n = outbreak_occurs.size

    outbreak_probability = outbreak_occurs.sum()/n

    outbreak_probability_std = np.sqrt(
        outbreak_probability * (1-outbreak_probability) / n)

    # Final size
    fs = np.array([x["I_total"][-1] for x in results])/P

    fs_mean = fs.mean()

    fs_std = fs.std()/np.sqrt(n)

    # FS when an outbreak occurs
    fs_conditional = np.array(
        [x["I_total"][-1] for idx, x in enumerate(results) if outbreak_occurs[idx] > 0])/P
    fs_conditional_mean = fs_conditional.mean()

    fs_conditional_std = fs_conditional.std()/np.sqrt(fs_conditional.size)

    ans = {
        "type": "baseline",
        "num_trajectories": n,
        "pHat": outbreak_probability,
        "pHat_std": outbreak_probability_std,
        "FS_avg": fs_mean,
        "FS_std": fs_std,
        "FS_conditional": fs_conditional_mean,
        "FS_conditional_std": fs_conditional_std
    }
    return ans


# %% Load and save data
df = list(map(load_intervention_data, filenames))
df = pd.DataFrame(df)

df = df.sort_values(by=["target", "day", "strength"])

df.to_csv("../data/df_results3_intervention.csv")
# %%
df_base = list(map(load_baseline_data, baseline_files))
df_base = pd.DataFrame(df_base)

df_base.to_csv("../data/df_results3_baseline.csv")

# %%
# file_path = "../data/simulations/interventions/"
# for file in filenames:
#     try:
#         with open(file_path + file, "r") as f:
#             results_json = json.load(f)
#         f.close()
#     except:
#         print(file)
