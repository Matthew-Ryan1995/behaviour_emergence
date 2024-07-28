#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 13:01:58 2024

Appendix: Between odds ratio plots

@author: Matt Ryan
"""
# %% Libraries
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
import matplotlib.ticker as tkr
import matplotlib.cm as cm

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
file_path = "../data/df_between_OR.csv"
df = pd.read_csv(file_path, index_col=0)

df["pHat_lwr"] = df["pHat"] - 1.96 * df["pHat_std"]
df["pHat_upr"] = df["pHat"] + 1.96 * df["pHat_std"]

df["FS_lwr"] = df["FS_avg"] - 1.96 * df["FS_std"]
df["FS_upr"] = df["FS_avg"] + 1.96 * df["FS_std"]

df["FS_conditional_lwr"] = df["FS_conditional"] - \
    1.96 * df["FS_conditional_std"]
df["FS_conditional_upr"] = df["FS_conditional"] + \
    1.96 * df["FS_conditional_std"]

# %%
dpi = 600
font_size = 16

# %%
tmp = df[["OR", "p", "pHat", "pHat_lwr", "pHat_upr"]]

plt.figure()


B_range = np.linspace(
    start=tmp["OR"].min(), stop=tmp["OR"].max(), num=300)

spl = make_smoothing_spline(np.array(tmp["OR"]),
                            np.array(tmp["pHat"]))
spl_lwr = make_smoothing_spline(np.array(tmp["OR"]),
                                np.array(tmp["pHat_lwr"]))
spl_upr = make_smoothing_spline(np.array(tmp["OR"]),
                                np.array(tmp["pHat_upr"]))
pHat_smooth = spl(B_range)
pHat_lwr_smooth = spl_lwr(B_range)
pHat_upr_smooth = spl_upr(B_range)

# plt.plot(tmp_df["p"], tmp_df["pHat"], )
plt.scatter(tmp["OR"], tmp["pHat"], marker=".",
            color="blue",
            alpha=.1)
plt.plot(B_range, pHat_smooth, color="blue",)
plt.fill_between(B_range,
                 pHat_lwr_smooth,
                 pHat_upr_smooth,
                 color="blue",
                 alpha=.1)


plt.xlabel("Odds ratio (k)", fontsize=font_size)
plt.ylabel("Outbreak probability", fontsize=font_size)


# plt.yticks([0.25, 0.5, 0.75], fontsize=font_size)
# plt.text(-1.34, 0.95, "1.00", fontsize=font_size)
# plt.text(-1.34, 0., "0.00", fontsize=font_size)

plt.xticks(fontsize=font_size)
plt.yticks(fontsize=font_size)

plt.savefig("../figs/between_OR_outbreak_prob.png",
            dpi=dpi, bbox_inches="tight")
plt.close()

# %%
tmp = df[["OR", "p", "FS_conditional",
          "FS_conditional_lwr", "FS_conditional_upr"]]

plt.figure()


B_range = np.linspace(
    start=tmp["OR"].min(), stop=tmp["OR"].max(), num=300)

spl = make_smoothing_spline(np.array(tmp["OR"]),
                            np.array(tmp["FS_conditional"]))
spl_lwr = make_smoothing_spline(np.array(tmp["OR"]),
                                np.array(tmp["FS_conditional_lwr"]))
spl_upr = make_smoothing_spline(np.array(tmp["OR"]),
                                np.array(tmp["FS_conditional_upr"]))
pHat_smooth = spl(B_range)
pHat_lwr_smooth = spl_lwr(B_range)
pHat_upr_smooth = spl_upr(B_range)

plt.scatter(tmp["OR"], tmp["FS_conditional"], color="red", marker=".",
            alpha=.1)
plt.plot(B_range, pHat_smooth, color="red")
plt.fill_between(B_range, pHat_lwr_smooth,
                 pHat_upr_smooth, color="red", alpha=.1)


plt.xlabel("Odds ratio (k)", fontsize=font_size)
plt.ylabel("Final size", fontsize=font_size)


# plt.yticks([0.25, 0.5, 0.75], fontsize=font_size)
# plt.text(-1.34, 0.95, "1.00", fontsize=font_size)
# plt.text(-1.34, 0., "0.00", fontsize=font_size)

plt.xticks(fontsize=font_size)
plt.yticks(fontsize=font_size)

plt.savefig("../figs/between_OR_final_size.png", dpi=dpi, bbox_inches="tight")
plt.close()


# %%

error_type = ["exp", "poly"]
day = [5, 10, 15]

tmp_df_prime = df

vmin = np.log(
    tmp_df_prime[[f"{e}_error_{d}" for e in error_type for d in day]].min().min())
vmax = np.log(
    tmp_df_prime[[f"{e}_error_{d}" for e in error_type for d in day]].max().max())

fig, axes = plt.subplots(nrows=3, ncols=2, sharex=True, sharey=True)


R0_max = tmp_df_prime["OR"].max()

c = 0
for d in day:
    for e in error_type:
        if e == "poly":
            E = "cubic"
        else:
            E = "exponential"
        ax = axes.flat[c]
        tmp = tmp_df_prime[["OR", "p", f"{e}_error_{d}"]]

        tmp.loc[:, (f"{e}_error_{d}")] = np.log(tmp.loc[:, (f"{e}_error_{d}")])

        B_range = np.linspace(
            start=tmp["OR"].min(), stop=tmp["OR"].max(), num=300)

        spl = make_smoothing_spline(np.array(tmp["OR"]),
                                    np.array(tmp[f"{e}_error_{d}"]))
        pHat_smooth = spl(B_range)

        # plt.plot(tmp_df["p"], tmp_df["pHat"], )
        color = next(ax._get_lines.prop_cycler)['color']
        ax.scatter(tmp["OR"], tmp[f"{e}_error_{d}"], color=color, marker=".",
                   alpha=.1)
        ax.plot(B_range, pHat_smooth, color=color)

        ax.tick_params(axis='both', which='major', labelsize=font_size)
        ax.set_xticks([0, 0.5, 1])
        if d == 5:
            ax.set_title(f"{E}", fontsize=font_size)
        if e == "poly":
            y_max = tmp[f"{e}_error_{d}"].max()
            y_min = tmp[f"{e}_error_{d}"].min()
            ax.text(1 + 0.05, -15, f"day {d}", fontsize=font_size)

        if (e == "poly") & (d == 15):
            ax.set_xlabel("Odds ratio (k)", x=-0.1, fontsize=font_size)
        if (e == "exp") & (d == 10):
            ax.set_ylabel("Log-Error", fontsize=font_size)
        # plt.show()
        c += 1
fig.savefig("../figs/between_OR_error.png", dpi=dpi, bbox_inches="tight")
plt.close(fig)


# %%
# for d in day:
#     for e in error_type:
#         if e == "poly":
#             E = "cubic"
#         else:
#             E = "exponential"

#         plt.figure()
#         R0_max = tmp_df_prime["OR"].max()

#         tmp = tmp_df_prime[["OR", "p", f"{e}_error_{d}"]]

#         tmp.loc[:, (f"{e}_error_{d}")] = np.log(tmp.loc[:, (f"{e}_error_{d}")])

#         y_max = tmp[f"{e}_error_{d}"].max()
#         y_min = tmp[f"{e}_error_{d}"].min()

#         ax = plt.gca()
#         B_range = np.linspace(
#             start=tmp["OR"].min(), stop=tmp["OR"].max(), num=300)

#         spl = make_smoothing_spline(np.array(tmp["OR"]),
#                                     np.array(tmp[f"{e}_error_{d}"]))
#         pHat_smooth = spl(B_range)

#         # plt.plot(tmp_df["p"], tmp_df["pHat"], )
#         color = next(ax._get_lines.prop_cycler)['color']
#         plt.scatter(tmp["OR"], tmp[f"{e}_error_{d}"], color=color, marker=".",
#                     alpha=.1)
#         plt.plot(B_range, pHat_smooth, color=color)

#         plt.tick_params(axis='both', which='major', labelsize=font_size)
#         plt.title(f"{E}", fontsize=font_size)
#         plt.text(R0_max + 0.05, (y_max + y_min) /
#                  2, f"day {d}", fontsize=font_size)

#         plt.xlabel("OR", fontsize=font_size)
#         plt.ylabel("$Log-Error$", fontsize=font_size)
#         # plt.show()
#         plt.legend(loc=(1, 0.1))
#         plt.savefig(
#             f"../figs/between_OR_error_{e}_day{d}.png", dpi=dpi, bbox_inches="tight")
#         plt.close()
