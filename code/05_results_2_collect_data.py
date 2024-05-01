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

# %% Get filepaths

file_path = "../data/simulations/vary_Bstar/"

filenames = next(os.walk(file_path), (None, None, []))[2]  # [] if no file

with open("../data/simulation_parameters.json", "r") as f:
    simulation_parameters = json.load(f)
f.close()

# Remove unwanted files
if ".DS_Store" in filenames:
    filenames.remove(".DS_Store")

# %% Helper functions


def load_bstar_data(file, file_path="../data/simulations/vary_Bstar/",
                    P=simulation_parameters["P"]):
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
    fs = np.array([x["I_total"][-1] for x in results])

    fs_mean = fs.mean()

    fs_std = fs.std()/np.sqrt(fs.size)

    # FS when an outbreak occurs
    fs_conditional = np.array(
        [x["I_total"][-1] for idx, x in enumerate(results) if outbreak_occurs[idx] > 0])
    fs_conditional_mean = fs_conditional.mean()

    fs_conditional_std = fs_conditional.std()/np.sqrt(fs_conditional.size)

    # BStar parameter
    b_star = np.float32(re.findall("[0-1].[0-9]*",  file)[0])

    ans = {
        "Bstar": b_star,
        "pHat": outbreak_probability,
        "pHat_std": outbreak_probability_std,
        "FS_avg": fs_mean,
        "FS_std": fs_std,
        "FS_conditional": fs_conditional_mean,
        "FS_conditional_std": fs_conditional_std
    }
    return ans


# %% Load and save data
df = list(map(load_bstar_data, filenames))
df = pd.DataFrame(df)

df.to_csv("../data/df_results2_varyBstar.csv")
