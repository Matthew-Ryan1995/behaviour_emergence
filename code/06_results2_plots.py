#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 09:08:48 2024

@author: rya200
"""
# %% Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# %% Load data
file_path = "../data/df_results2_varyBstar.csv"
df = pd.read_csv(file_path, index_col=0)

df["pHat_lwr"] = df["pHat"] - 1.96 * df["pHat_std"]
df["pHat_upr"] = df["pHat"] + 1.96 * df["pHat_std"]

df["FS_lwr"] = df["FS_avg"] - 1.96 * df["FS_std"]
df["FS_upr"] = df["FS_avg"] + 1.96 * df["FS_std"]

df["FS_conditional_lwr"] = df["FS_conditional"] - \
    1.96 * df["FS_conditional_std"]
df["FS_conditional_upr"] = df["FS_conditional"] + \
    1.96 * df["FS_conditional_std"]

# %% Figure 1: Outbreak Probability over time


fig = plt.figure()
l1 = sns.lineplot(data=df, x="Bstar", y="pHat")
l2 = sns.lineplot(data=df, x="Bstar", y="pHat_lwr",
                  linestyle="--", alpha=0.1, color="blue")
l3 = sns.lineplot(data=df, x="Bstar", y="pHat_upr",
                  linestyle="--", alpha=0.1, color="blue")

line = l3.get_lines()
plt.fill_between(line[0].get_xdata(), line[1].get_ydata(),
                 line[2].get_ydata(), color='blue', alpha=.1)

plt.xlabel("Infection-free steady state of beahviour")
plt.ylabel("Probability of major outbreak")
plt.show()

# %% Figure 2: Final size

fig = plt.figure()
l1 = sns.lineplot(data=df, x="Bstar", y="FS_avg", color="red")
l2 = sns.lineplot(data=df, x="Bstar", y="FS_lwr",
                  linestyle="--", alpha=0.1, color="red")
l3 = sns.lineplot(data=df, x="Bstar", y="FS_upr",
                  linestyle="--", alpha=0.1, color="red")

line = l3.get_lines()
plt.fill_between(line[0].get_xdata(), line[1].get_ydata(),
                 line[2].get_ydata(), color='red', alpha=.1)

plt.xlabel("Infection-free steady state of beahviour")
plt.ylabel("Average final size of simulations")
plt.show()

# %% Figure 3: Conditional Final size

fig = plt.figure()
l1 = sns.lineplot(data=df, x="Bstar", y="FS_conditional", color="red")
l2 = sns.lineplot(data=df, x="Bstar", y="FS_conditional_lwr",
                  linestyle="--", alpha=0.1, color="red")
l3 = sns.lineplot(data=df, x="Bstar", y="FS_conditional_upr",
                  linestyle="--", alpha=0.1, color="red")

line = l3.get_lines()
plt.fill_between(line[0].get_xdata(), line[1].get_ydata(),
                 line[2].get_ydata(), color='red', alpha=.1)

plt.xlabel("Infection-free steady state of beahviour")
plt.ylabel("Conditional Average final size of simulations")
plt.show()
