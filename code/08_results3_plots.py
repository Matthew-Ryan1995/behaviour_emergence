#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 10:21:59 2024

@author: rya200
"""
# %% Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# %%

file_path = "../data/df_results3_intervention.csv"
df = pd.read_csv(file_path, index_col=0)

file_path_baseline = "../data/df_results3_baseline.csv"
df_baseline = pd.read_csv(file_path_baseline, index_col=0)

# %% Create new vars

df["odds_ratio"] = (df["pHat"]/(1-df["pHat"])) / \
    (df_baseline["pHat"]/(1-df_baseline["pHat"])).iloc[0]

df["odds_ratio_se"] = np.sqrt((1/df["num_trajectories"]) * (1/(df["pHat"]*(
    1-df["pHat"])) + 1/(df_baseline["pHat"]*(1-df_baseline["pHat"])).iloc[0]))

df["odds_ratio_lwr"] = np.exp(np.log(df["odds_ratio"]) - 1.96*df["odds_ratio"])
df["odds_ratio_upr"] = np.exp(np.log(df["odds_ratio"]) + 1.96*df["odds_ratio"])

df["FS_diff"] = df_baseline["FS_avg"].iloc[0] - df["FS_avg"]
df["FS_diff_se"] = np.sqrt(df_baseline["FS_std"].iloc[0]**2 + df["FS_std"]**2)

df["FS_diff_lwr"] = df["FS_diff"] - 1.96 * df["FS_diff_se"]
df["FS_diff_upr"] = df["FS_diff"] + 1.96 * df["FS_diff_se"]

df["FS_conditional_diff"] = df_baseline["FS_conditional"].iloc[0] - \
    df["FS_conditional"]
df["FS_conditional_diff_se"] = np.sqrt(
    df_baseline["FS_conditional_std"].iloc[0]**2 + df["FS_conditional_std"]**2)

df["FS_conditional_diff_lwr"] = df["FS_conditional_diff"] - \
    1.96 * df["FS_conditional_diff_se"]
df["FS_conditional_diff_upr"] = df["FS_conditional_diff"] + \
    1.96 * df["FS_conditional_diff_se"]

# %% Figure 1: Odds ratio
# todo: All days, all Interventions

target = "w3"
day = 5

plot_data = df[(df["target"] == target) & (df["day"] == day)]
plt.figure()

l1 = sns.lineplot(data=plot_data, x="strength", y="odds_ratio")
l2 = sns.lineplot(data=plot_data, x="strength", y="odds_ratio_lwr",
                  linestyle="--", alpha=0.1, color="blue")
# l3 = sns.lineplot(data=plot_data, x="strength", y="odds_ratio_upr",
#                   linestyle="--", alpha=0.1, color="blue")

# line = l3.get_lines()
# plt.fill_between(line[0].get_xdata(), line[1].get_ydata(),
#                   line[2].get_ydata(), color='blue', alpha=.1)

plt.plot([0, 5], [1, 1], "k:")

plt.xlabel("Strength of intervention")
plt.ylabel("Odds ratio of major outbreak")
plt.show()

# %% Figure 1: Log-Odds ratio
# todo: All days, all Interventions

target = "w1"
day = 5

plot_data["lo"] = np.log(plot_data["odds_ratio"])
plot_data["lo_lwr"] = np.log(plot_data["odds_ratio_lwr"])
plot_data["lo_upr"] = np.log(plot_data["odds_ratio_upr"])

plt.figure()

l1 = sns.lineplot(data=plot_data, x="strength", y="lo")
l2 = sns.lineplot(data=plot_data, x="strength", y="lo_lwr",
                  linestyle="--", alpha=0.1, color="blue")
l3 = sns.lineplot(data=plot_data, x="strength", y="lo_upr",
                  linestyle="--", alpha=0.1, color="blue")

line = l3.get_lines()
plt.fill_between(line[0].get_xdata(), line[1].get_ydata(),
                 line[2].get_ydata(), color='blue', alpha=.1)

plt.plot([0, 5], [0, 0], "k:")

plt.xlabel("Strength of intervention")
plt.ylabel("Log-Odds ratio of major outbreak")
plt.show()

# %% Figure 2: Infections saved

plt.figure()

l1 = sns.lineplot(data=plot_data, x="strength", y="FS_diff", color="red")
l2 = sns.lineplot(data=plot_data, x="strength", y="FS_diff_lwr",
                  linestyle="--", alpha=0.1, color="red")
l3 = sns.lineplot(data=plot_data, x="strength", y="FS_diff_upr",
                  linestyle="--", alpha=0.1, color="red")

line = l3.get_lines()
plt.fill_between(line[0].get_xdata(), line[1].get_ydata(),
                 line[2].get_ydata(), color='red', alpha=.1)


plt.plot([-0.05, 5.05], [0, 0], "k:")

plt.xlabel("Strength of intervention")
plt.ylabel("Infections saved")
plt.show()

# %% Figure 2: Conditional Infections saved

plt.figure()

l1 = sns.lineplot(data=plot_data, x="strength",
                  y="FS_conditional_diff", color="red")
l2 = sns.lineplot(data=plot_data, x="strength", y="FS_conditional_diff_lwr",
                  linestyle="--", alpha=0.1, color="red")
l3 = sns.lineplot(data=plot_data, x="strength", y="FS_conditional_diff_upr",
                  linestyle="--", alpha=0.1, color="red")

line = l3.get_lines()
plt.fill_between(line[0].get_xdata(), line[1].get_ydata(),
                 line[2].get_ydata(), color='red', alpha=.1)

plt.plot([-0.05, 5.05], [0, 0], "k:")

plt.xlabel("Strength of intervention")
plt.ylabel("Conditional Infections saved")
plt.show()
