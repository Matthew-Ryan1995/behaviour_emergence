#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 09:08:48 2024

todo: Fit a smoothed line to these, not a pointwise.  This will allow for the error in point simulations

@author: rya200
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

# %% Plot parameters

dpi = 600
font_size = 16

# %% Figure 1: Outbreak Probability over time
# df = df.sort_values(by="Bstar")
B_range = np.linspace(start=df["Bstar"].min(), stop=df["Bstar"].max(), num=300)
spl = make_smoothing_spline(np.array(df["Bstar"]),
                            np.array(df["pHat"]))
spl_lwr = make_smoothing_spline(np.array(df["Bstar"]),
                                np.array(df["pHat_lwr"]))
spl_upr = make_smoothing_spline(np.array(df["Bstar"]),
                                np.array(df["pHat_upr"]))
pHat_smooth = spl(B_range)
pHat_lwr_smooth = spl_lwr(B_range)
pHat_upr_smooth = spl_upr(B_range)


fig = plt.figure()
# l1 = sns.lineplot(data=df, x="Bstar", y="pHat")
# l2 = sns.lineplot(data=df, x="Bstar", y="pHat_lwr",
#                   linestyle="--", alpha=0.1, color="blue")
# l3 = sns.lineplot(data=df, x="Bstar", y="pHat_upr",
#                   linestyle="--", alpha=0.1, color="blue")

# line = l3.get_lines()
# plt.fill_between(line[0].get_xdata(), line[1].get_ydata(),
#                  line[2].get_ydata(), color='blue', alpha=.1)
plt.scatter(df["Bstar"], df["pHat"], color="grey", marker=".")
plt.plot(B_range, pHat_smooth, color="blue")
plt.fill_between(B_range, pHat_lwr_smooth,
                 pHat_upr_smooth, color='blue', alpha=.1)

plt.xlabel("Infection-free steady state of behaviour ($B^*$)",
           fontsize=font_size)
plt.ylabel("Probability of outbreak", fontsize=font_size)
plt.xticks(fontsize=font_size)
plt.yticks(fontsize=font_size)

plt.savefig("../figs/results2_outbreak_prob_by_Bstar.png",
            dpi=dpi,
            bbox_inches="tight")
plt.close()

tmp = next(idx for idx in range(300) if B_range[idx] > 0.5)
print("Outbreak prob when behaviour is 0%")
print(
    f"{pHat_smooth[0].round(3)}, 95% CI [{pHat_lwr_smooth[0].round(3)}, {pHat_upr_smooth[0].round(3)}]")
print("Outbreak prob when behaviour is 50%")
print(
    f"{pHat_smooth[tmp].round(3)}, 95% CI [{pHat_lwr_smooth[tmp].round(3)}, {pHat_upr_smooth[tmp].round(3)}]")

# %% Figure 2: Final size

B_range = np.linspace(start=df["Bstar"].min(), stop=df["Bstar"].max(), num=300)
spl = make_smoothing_spline(np.array(df["Bstar"]),
                            np.array(df["FS_avg"]))
spl_lwr = make_smoothing_spline(np.array(df["Bstar"]),
                                np.array(df["FS_lwr"]))
spl_upr = make_smoothing_spline(np.array(df["Bstar"]),
                                np.array(df["FS_upr"]))
FS_smooth = spl(B_range) * 100
FS_lwr_smooth = spl_lwr(B_range) * 100
FS_upr_smooth = spl_upr(B_range) * 100


fig = plt.figure()
# l1 = sns.lineplot(data=df, x="Bstar", y="FS_avg", color="red")
# l2 = sns.lineplot(data=df, x="Bstar", y="FS_lwr",
#                   linestyle="--", alpha=0.1, color="red")
# l3 = sns.lineplot(data=df, x="Bstar", y="FS_upr",
#                   linestyle="--", alpha=0.1, color="red")

# line = l3.get_lines()
# plt.fill_between(line[0].get_xdata(), line[1].get_ydata(),
#                  line[2].get_ydata(), color='red', alpha=.1)

plt.scatter(df["Bstar"], df["FS_avg"]*100, color="grey", marker=".")
plt.plot(B_range, FS_smooth, color="red")
plt.fill_between(B_range, FS_lwr_smooth,
                 FS_upr_smooth, color='red', alpha=.1)

plt.xlabel("Infection-free steady state of behaviour", fontsize=font_size)
plt.ylabel("Expected final size (%)", fontsize=font_size)
plt.xticks(fontsize=font_size)
plt.yticks(fontsize=font_size)

plt.savefig("../figs/results2_FS_by_Bstar.png",
            dpi=dpi,
            bbox_inches="tight")
plt.close()

# tmp = next(idx for idx in range(300) if B_range[idx] > 0.5)
# print("Outbreak prob when behaviour is 0%")
# print(
#     f"{FS_smooth[0].round(1)}, 95% CI [{FS_lwr_smooth[0].round(1)}, {FS_upr_smooth[0].round(1)}]")
# print("Outbreak prob when behaviour is 50%")
# print(
#     f"{FS_smooth[tmp].round(1)}, 95% CI [{FS_lwr_smooth[tmp].round(1)}, {FS_upr_smooth[tmp].round(1)}]")


# %% Figure 3: Conditional Final size


B_range = np.linspace(start=df["Bstar"].min(), stop=df["Bstar"].max(), num=300)
spl = make_smoothing_spline(np.array(df["Bstar"]),
                            np.array(df["FS_conditional"]))
spl_lwr = make_smoothing_spline(np.array(df["Bstar"]),
                                np.array(df["FS_conditional_lwr"]))
spl_upr = make_smoothing_spline(np.array(df["Bstar"]),
                                np.array(df["FS_conditional_upr"]))
FS_smooth = spl(B_range) * 100
FS_lwr_smooth = spl_lwr(B_range) * 100
FS_upr_smooth = spl_upr(B_range) * 100


fig = plt.figure()
# l1 = sns.lineplot(data=df, x="Bstar", y="FS_conditional", color="red")
# l2 = sns.lineplot(data=df, x="Bstar", y="FS_conditional_lwr",
#                   linestyle="--", alpha=0.1, color="red")
# l3 = sns.lineplot(data=df, x="Bstar", y="FS_conditional_upr",
#                   linestyle="--", alpha=0.1, color="red")

# line = l3.get_lines()
# plt.fill_between(line[0].get_xdata(), line[1].get_ydata(),
#                  line[2].get_ydata(), color='red', alpha=.1)

plt.scatter(df["Bstar"], df["FS_conditional"] * 100, color="grey", marker=".")
plt.plot(B_range, FS_smooth, color="red")
plt.fill_between(B_range, FS_lwr_smooth,
                 FS_upr_smooth, color='red', alpha=.1)

plt.xlabel("Infection-free steady state of behaviour ($B^*$)",
           fontsize=font_size)
plt.ylabel("Expected final size (%)", fontsize=font_size)
plt.xticks(fontsize=font_size)
plt.yticks(fontsize=font_size)

plt.savefig("../figs/results2_conditional_FS_by_Bstar.png",
            dpi=dpi,
            bbox_inches="tight")
plt.close()


tmp = next(idx for idx in range(300) if B_range[idx] > 0.5)
print("FS when behaviour is 0%")
print(
    f"{FS_smooth[0].round(1)}, 95% CI [{FS_lwr_smooth[0].round(1)}, {FS_upr_smooth[0].round(1)}]")
print("FS when behaviour is 50%")
print(
    f"{FS_smooth[tmp].round(1)}, 95% CI [{FS_lwr_smooth[tmp].round(1)}, {FS_upr_smooth[tmp].round(1)}]")


# %%

# tmp = df.sort_values(by="Bstar")
# tmp["new"] = tmp["FS_conditional"] * tmp["pHat"]

# fig = plt.figure()
# l1 = sns.lineplot(data=tmp, x="Bstar", y="new", color="red")
# l1 = sns.lineplot(data=tmp, x="Bstar", y="FS_avg", color="blue")
# plt.show()
