#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 10:21:59 2024


Appendix w3 sweep plots by 2 orders of magnitude

@author: Matt Ryan
"""
# %% Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# import seaborn as sns
from scipy.interpolate import make_smoothing_spline

params = {"ytick.color": "black",
          "xtick.color": "black",
          "axes.labelcolor": "black",
          "axes.edgecolor": "black",
          # "text.usetex": True,
          "font.family": "serif"}
plt.rcParams.update(params)
plt.rcParams['mathtext.fontset'] = 'dejavuserif'

# %%

file_path = "data/df_results3_intervention_w3_by_100.csv"
df = pd.read_csv(file_path, index_col=0)

file_path_baseline = "data/df_results3_baseline_w3_by_100.csv"
df_baseline = pd.read_csv(file_path_baseline, index_col=0)

# %% Plot parameters

dpi = 600
font_size = 16

# %% Create new vars

df["log_odds_ratio"] = np.log((df["pHat"]/(1-df["pHat"])) /
                              (df_baseline["pHat"]/(1-df_baseline["pHat"])).iloc[0])

df["log_odds_ratio_se"] = np.sqrt((1/df["num_trajectories"]) * (1/(df["pHat"]*(
    1-df["pHat"])) + 1/(df_baseline["pHat"]*(1-df_baseline["pHat"])).iloc[0]))

df["log_odds_ratio_lwr"] = df["log_odds_ratio"] - 1.96*df["log_odds_ratio_se"]
df["log_odds_ratio_upr"] = df["log_odds_ratio"] + 1.96*df["log_odds_ratio_se"]

# df["log_odds_ratio"] = np.exp(df["log_odds_ratio"])
# df["log_odds_ratio_lwr"] = np.exp(df["log_odds_ratio_lwr"])
# df["log_odds_ratio_upr"] = np.exp(df["log_odds_ratio_upr"])

df["FS_diff"] = (df_baseline["FS_avg"].iloc[0] - df["FS_avg"]
                 ) / df_baseline["FS_avg"].iloc[0]
# df["FS_diff_se"] = np.sqrt(df_baseline["FS_std"].iloc[0]**2 + df["FS_std"]**2)
# SE calculated using Eqn 10 of https://link.springer.com/content/pdf/10.3758/BF03201412.pdf
df["FS_diff_se"] = np.sqrt(
    df["FS_std"]**2 + (df["FS_avg"] / df_baseline["FS_avg"].iloc[0]
                       ) ** 2 * df_baseline["FS_std"].iloc[0]**2
) / df_baseline["FS_avg"].iloc[0]

df["FS_diff_lwr"] = df["FS_diff"] - 1.96 * df["FS_diff_se"]
df["FS_diff_upr"] = df["FS_diff"] + 1.96 * df["FS_diff_se"]

df["FS_conditional_diff"] = (df_baseline["FS_conditional"].iloc[0] - df["FS_conditional"]
                             ) / df_baseline["FS_conditional"].iloc[0]

# df["FS_conditional_diff_se"] = np.sqrt(
#     df_baseline["FS_conditional_std"].iloc[0]**2 + df["FS_conditional_std"]**2)

df["FS_conditional_diff_se"] = np.sqrt(
    df["FS_conditional_std"]**2 + (df["FS_conditional"] / df_baseline["FS_conditional"].iloc[0]
                                   ) ** 2 * df_baseline["FS_conditional_std"].iloc[0]**2
) / df_baseline["FS_conditional"].iloc[0]

df["FS_conditional_diff_lwr"] = df["FS_conditional_diff"] - \
    1.96 * df["FS_conditional_diff_se"]
df["FS_conditional_diff_upr"] = df["FS_conditional_diff"] + \
    1.96 * df["FS_conditional_diff_se"]

# %% Figure 1: Odds ratio
# todo: All days, all Interventions

target = "w1"
day = 5

plot_data = df[(df["target"] == target) & (df["day"] == day)]
# %% Function it


def create_plot(df, target, day, f, ymin=-1, ymax=1, subplot_idx=1, plot_type="log_odds"):
    plot_data = df[(df["target"] == target) & (df["day"] == day)]

    xmin = df["strength"].min() - 1e-1
    xmax = df["strength"].max() + 1e-1

    if plot_type == "log_odds":
        strength_range = np.linspace(start=plot_data["strength"].min(),
                                     stop=plot_data["strength"].max(),
                                     num=300)
        spl = make_smoothing_spline(np.array(plot_data["strength"]),
                                    np.array(plot_data["log_odds_ratio"]))
        spl_lwr = make_smoothing_spline(np.array(plot_data["strength"]),
                                        np.array(plot_data["log_odds_ratio_lwr"]))
        spl_upr = make_smoothing_spline(np.array(plot_data["strength"]),
                                        np.array(plot_data["log_odds_ratio_upr"]))
        log_odds_smooth = spl(strength_range)
        log_odds_lwr_smooth = spl_lwr(strength_range)
        log_odds_upr_smooth = spl_upr(strength_range)

        axarr = f.add_subplot(3, 3, subplot_idx)

        plt.scatter(plot_data["strength"],
                    plot_data["log_odds_ratio"], color="grey", marker=".")
        plt.plot(strength_range, log_odds_smooth, color="blue")
        plt.fill_between(strength_range, log_odds_lwr_smooth,
                         log_odds_upr_smooth, color='blue', alpha=.1)

        plt.plot([0, 5], [0, 0], "k:")
        plt.xlim(xmin, xmax)
        plt.xticks([0, 1, 2, 3, 4, 5], fontsize=font_size)
        plt.ylim(ymin, ymax)
        plt.yticks(fontsize=font_size)

    if plot_type == "FS":

        # ylim = max(abs(ymin), abs(ymax))
        strength_range = np.linspace(start=plot_data["strength"].min(),
                                     stop=plot_data["strength"].max(),
                                     num=300)

        spl = make_smoothing_spline(np.array(plot_data["strength"]),
                                    np.array(plot_data["FS_diff"]))
        spl_lwr = make_smoothing_spline(np.array(plot_data["strength"]),
                                        np.array(plot_data["FS_diff_lwr"]))
        spl_upr = make_smoothing_spline(np.array(plot_data["strength"]),
                                        np.array(plot_data["FS_diff_upr"]))
        FS_smooth = spl(strength_range) * 100
        FS_lwr_smooth = spl_lwr(strength_range) * 100
        FS_upr_smooth = spl_upr(strength_range) * 100

        axarr = f.add_subplot(3, 3, subplot_idx)

        plt.scatter(plot_data["strength"], plot_data["FS_diff"] * 100,
                    color="grey", marker=".")
        plt.plot(strength_range, FS_smooth, color="red")
        plt.fill_between(strength_range, FS_lwr_smooth,
                         FS_upr_smooth, color='red', alpha=.1)

        plt.plot([-0.05, 5.05], [0, 0], "k:")

        plt.xlim(xmin, xmax)
        plt.xticks([0, 1, 2, 3, 4, 5], fontsize=font_size)
        plt.ylim(ymin*100, ymax*100)
        # plt.yticks(np.array([-0.2, 0, 0.2, 0.4])*100)
        plt.yticks([-50,  0, 50, 100], fontsize=font_size)

    if plot_type == "FS_conditional":

        strength_range = np.linspace(start=plot_data["strength"].min(),
                                     stop=plot_data["strength"].max(),
                                     num=300)

        spl = make_smoothing_spline(np.array(plot_data["strength"]),
                                    np.array(plot_data["FS_conditional_diff"]))
        spl_lwr = make_smoothing_spline(np.array(plot_data["strength"]),
                                        np.array(plot_data["FS_conditional_diff_lwr"]))
        spl_upr = make_smoothing_spline(np.array(plot_data["strength"]),
                                        np.array(plot_data["FS_conditional_diff_upr"]))
        FS_smooth = spl(strength_range) * 100
        FS_lwr_smooth = spl_lwr(strength_range) * 100
        FS_upr_smooth = spl_upr(strength_range) * 100

        axarr = f.add_subplot(3, 3, subplot_idx)

        # plt.scatter(plot_data["strength"],
        #             plot_data["FS_conditional_diff"] * 100, color="grey", marker=".")
        plt.plot(strength_range, FS_smooth, color="red")
        plt.fill_between(strength_range, FS_lwr_smooth,
                         FS_upr_smooth, color='red', alpha=.1)

        plt.plot([-0.05, 5.05], [0, 0], "k:")

        plt.xlim(xmin, xmax)
        plt.xticks([0, 1, 2, 3, 4, 5], fontsize=font_size)
        plt.ylim(ymin*100, ymax*100)
        # plt.yticks(np.array([-0.2, 0, 0.2, 0.4])*100)
        plt.yticks([-50,  0, 50, 100], fontsize=font_size)


# log-odds
fig = plt.figure()

ymin = df["log_odds_ratio_lwr"].min()
ymax = df["log_odds_ratio_upr"].max()

ylim = max(abs(ymin), abs(ymax))

days = [5, 10, 15]
target_count = 0
targets = ["w1", "w2", "w3"]
targets_display = ["$\omega_1$", "$\omega_2$", "$\omega_3$"]
counter = 1

for d in days:
    for t in targets:
        ax = create_plot(df, t, d,
                         ymin=-ylim, ymax=ylim,
                         f=fig, subplot_idx=counter)
        counter += 1

        if d == 5:
            plt.title(targets_display[target_count], fontsize=font_size)
            target_count += 1

        if t == "w3":
            plt.text(5.5, (ylim + (-ylim))/2,  f"day {d}", fontsize=font_size)

        if (d == 10) & (t == "w1"):
            plt.ylabel("Log-Odds ratio of outbreak", fontsize=font_size)
        if (d == 15) & (t == "w2"):
            plt.xlabel("Strength of intervention", fontsize=font_size)


plt.tight_layout()
plt.savefig("figs/results3_log_odds_w3_by_100.png",
            dpi=dpi,
            bbox_inches="tight")
plt.close()

# %%
# FS
fig = plt.figure()
ymin = df["FS_diff"].min() - 0.1
ymax = df["FS_diff"].max() + 0.1

counter = 1
target_count = 0

for d in days:
    for t in targets:
        ax = create_plot(df, t, d, f=fig,
                         ymin=ymin, ymax=ymax,
                         subplot_idx=counter, plot_type="FS")
        counter += 1

        if d == 5:
            plt.title(targets_display[target_count], fontsize=font_size)
            target_count += 1

        if t == "w3":
            plt.text(5.5, (ymin + ymax)/2, f"day {d}", fontsize=font_size)

        if (d == 10) & (t == "w1"):
            plt.ylabel("Infections saved (%)", fontsize=font_size)
        if (d == 15) & (t == "w2"):
            plt.xlabel("Strength of intervention", fontsize=font_size)


plt.tight_layout()
plt.savefig("figs/results3_infections_saved_w3_by_100.png",
            dpi=dpi,
            bbox_inches="tight")
plt.close()

# %%
# FS Conditional
fig = plt.figure()
ymin = df["FS_conditional_diff_lwr"].min()
ymax = df["FS_conditional_diff_upr"].max()

counter = 1
target_count = 0

for d in days:
    for t in targets:
        ax = create_plot(df, t, d,
                         ymin=ymin, ymax=ymax,
                         f=fig, subplot_idx=counter,
                         plot_type="FS_conditional")
        counter += 1

        if d == 5:
            plt.title(targets_display[target_count], fontsize=font_size)
            target_count += 1

        if t == "w3":
            plt.text(5.5, (ymin + ymax)/2, f"day {d}", fontsize=font_size)

        if (d == 10) & (t == "w1"):
            plt.ylabel("Infections saved", fontsize=font_size)
        if (d == 15) & (t == "w2"):
            plt.xlabel("Strength of intervention", fontsize=font_size)


plt.tight_layout()
plt.savefig("figs/results3_infections_saved_conditional_w3_by_100.png",
            dpi=dpi,
            bbox_inches="tight")
plt.close()

# fig = plt.figure()
# ax1 = fig.add_subplot(2, 2, 1)
# ax1 = create_plot(df, "w1", 5)
# ax2 = fig.add_subplot(2, 2, 2)
# ax2 = create_plot(df, "w1", 10)
# plt.show()
# %%
