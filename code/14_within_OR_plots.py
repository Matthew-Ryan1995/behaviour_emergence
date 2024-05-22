#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 13:01:58 2024

@author: rya200
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
file_path = "../data/df_within_OR.csv"
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
# tmp_pivot = tmp.pivot(index="OR", columns="p", values="pHat")

# z = np.array(tmp_pivot)
# X = tmp_pivot.columns.values
# Y = tmp_pivot.index.values

# xx, yy = np.meshgrid(X, Y)

plt.figure()
# im = plt.contourf(xx, yy, z,  cmap=plt.cm.Blues)
# # ctr = plt.contour(xx, yy, z,
# #                   # levels = lvls,
# #                   # cmap=plt.cm.Blues,
# #                   colors="black",
# #                   alpha=0.5)
# cbar = plt.colorbar(im, format=tkr.PercentFormatter(xmax=1, decimals=1))
# cbar.ax.tick_params(labelsize=font_size)
# # cbar_lvls = ctr.levels[1:-1]
# # cbar.add_lines(ctr)
# # cbar.set_ticks(cbar_lvls)
# plt.xlim(xx.min(), xx.max())
# plt.ylim(yy.min(), yy.max())


ax = plt.gca()
for OR in tmp["OR"].unique():
    tmp_df = tmp[tmp["OR"] == OR]
    B_range = np.linspace(
        start=tmp_df["p"].min(), stop=tmp_df["p"].max(), num=300)

    spl = make_smoothing_spline(np.array(tmp_df["p"]),
                                np.array(tmp_df["pHat"]))
    spl_lwr = make_smoothing_spline(np.array(tmp_df["p"]),
                                    np.array(tmp_df["pHat_lwr"]))
    spl_upr = make_smoothing_spline(np.array(tmp_df["p"]),
                                    np.array(tmp_df["pHat_upr"]))
    pHat_smooth = spl(B_range)
    pHat_lwr_smooth = spl_lwr(B_range)
    pHat_upr_smooth = spl_upr(B_range)

    # plt.plot(tmp_df["p"], tmp_df["pHat"], )
    color = next(ax._get_lines.prop_cycler)['color']
    plt.scatter(tmp_df["p"], tmp_df["pHat"], color=color, marker=".",
                alpha=.1)
    plt.plot(B_range, pHat_smooth, color=color, label="OR: " + str(OR))
    plt.fill_between(B_range, pHat_lwr_smooth,
                     pHat_upr_smooth, color=color, alpha=.1)


plt.xlabel("p", fontsize=font_size)
plt.ylabel("outbreak prob", fontsize=font_size)


# plt.yticks([0.25, 0.5, 0.75], fontsize=font_size)
# plt.text(-1.34, 0.95, "1.00", fontsize=font_size)
# plt.text(-1.34, 0., "0.00", fontsize=font_size)

plt.xticks(fontsize=font_size)
plt.legend(loc=(1, 0.5))

plt.savefig("../figs/within_OR_outbreak_prob.png",
            dpi=dpi, bbox_inches="tight")
plt.close()

# %%
tmp = df[["OR", "p", "FS_conditional",
          "FS_conditional_lwr", "FS_conditional_upr"]]
# tmp_pivot = tmp.pivot(index="OR", columns="p", values="pHat")

# z = np.array(tmp_pivot)
# X = tmp_pivot.columns.values
# Y = tmp_pivot.index.values

# xx, yy = np.meshgrid(X, Y)

plt.figure()
# im = plt.contourf(xx, yy, z,  cmap=plt.cm.Blues)
# # ctr = plt.contour(xx, yy, z,
# #                   # levels = lvls,
# #                   # cmap=plt.cm.Blues,
# #                   colors="black",
# #                   alpha=0.5)
# cbar = plt.colorbar(im, format=tkr.PercentFormatter(xmax=1, decimals=1))
# cbar.ax.tick_params(labelsize=font_size)
# # cbar_lvls = ctr.levels[1:-1]
# # cbar.add_lines(ctr)
# # cbar.set_ticks(cbar_lvls)
# plt.xlim(xx.min(), xx.max())
# plt.ylim(yy.min(), yy.max())


ax = plt.gca()
for OR in tmp["OR"].unique():
    tmp_df = tmp[tmp["OR"] == OR]
    B_range = np.linspace(
        start=tmp_df["p"].min(), stop=tmp_df["p"].max(), num=300)

    spl = make_smoothing_spline(np.array(tmp_df["p"]),
                                np.array(tmp_df["FS_conditional"]))
    spl_lwr = make_smoothing_spline(np.array(tmp_df["p"]),
                                    np.array(tmp_df["FS_conditional_lwr"]))
    spl_upr = make_smoothing_spline(np.array(tmp_df["p"]),
                                    np.array(tmp_df["FS_conditional_upr"]))
    pHat_smooth = spl(B_range)
    pHat_lwr_smooth = spl_lwr(B_range)
    pHat_upr_smooth = spl_upr(B_range)

    # plt.plot(tmp_df["p"], tmp_df["pHat"], )
    color = next(ax._get_lines.prop_cycler)['color']
    plt.scatter(tmp_df["p"], tmp_df["FS_conditional"], color=color, marker=".",
                alpha=.1)
    plt.plot(B_range, pHat_smooth, color=color, label="OR: " + str(OR))
    plt.fill_between(B_range, pHat_lwr_smooth,
                     pHat_upr_smooth, color=color, alpha=.1)


plt.xlabel("p", fontsize=font_size)
plt.ylabel("Final size", fontsize=font_size)


# plt.yticks([0.25, 0.5, 0.75], fontsize=font_size)
# plt.text(-1.34, 0.95, "1.00", fontsize=font_size)
# plt.text(-1.34, 0., "0.00", fontsize=font_size)

plt.xticks(fontsize=font_size)
plt.legend(loc=(1, 0.5))

plt.savefig("../figs/within_OR_final_size.png", dpi=dpi, bbox_inches="tight")
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


R0_max = tmp_df_prime["p"].max()

c = 0
for d in day:
    for e in error_type:
        if e == "poly":
            E = "cubic"
        else:
            E = "exponential"
        ax = axes.flat[c]
        tmp = tmp_df_prime[["OR", "p", f"{e}_error_{d}"]]

        for OR in tmp["OR"].unique():
            tmp_df = tmp[tmp["OR"] == OR]
            B_range = np.linspace(
                start=tmp_df["p"].min(), stop=tmp_df["p"].max(), num=300)

            tmp_df.loc[:, (f"{e}_error_{d}")] = np.log(
                tmp_df.loc[:, (f"{e}_error_{d}")])

            spl = make_smoothing_spline(np.array(tmp_df["p"]),
                                        np.array(tmp_df[f"{e}_error_{d}"]))
            pHat_smooth = spl(B_range)

            # plt.plot(tmp_df["p"], tmp_df["pHat"], )
            color = next(ax._get_lines.prop_cycler)['color']
            ax.scatter(tmp_df["p"], tmp_df[f"{e}_error_{d}"], color=color, marker=".",
                       alpha=.1)
            ax.plot(B_range, pHat_smooth, color=color, label="OR: " + str(OR))

        ax.tick_params(axis='both', which='major', labelsize=font_size)
        ax.set_xticks([0, 0.5, 1])
        if d == 5:
            ax.set_title(f"{E}", fontsize=font_size)
        if e == "poly":
            ax.text(1 + 0.05, -15, f"day {d}", fontsize=font_size)

        if (e == "poly") & (d == 15):
            ax.set_xlabel("$p$", x=-0.1, fontsize=font_size)
        if (e == "exp") & (d == 10):
            ax.set_ylabel("$Log-Error$", fontsize=font_size)
        # plt.show()
        c += 1
plt.legend(loc=(1.1, -0.9))
fig.savefig("../figs/within_OR_error.png", dpi=dpi, bbox_inches="tight")
plt.close(fig)


# %%
for d in day:
    for e in error_type:
        if e == "poly":
            E = "cubic"
        else:
            E = "exponential"

        plt.figure()
        R0_max = tmp_df_prime["p"].max()

        tmp = tmp_df_prime[["OR", "p", f"{e}_error_{d}"]]

        tmp.loc[:, (f"{e}_error_{d}")] = np.log(tmp.loc[:, (f"{e}_error_{d}")])

        y_max = tmp[f"{e}_error_{d}"].max()
        y_min = tmp[f"{e}_error_{d}"].min()

        ax = plt.gca()
        for OR in tmp["OR"].unique():
            tmp_df = tmp[tmp["OR"] == OR]
            B_range = np.linspace(
                start=tmp_df["p"].min(), stop=tmp_df["p"].max(), num=300)

            spl = make_smoothing_spline(np.array(tmp_df["p"]),
                                        np.array(tmp_df[f"{e}_error_{d}"]))
            pHat_smooth = spl(B_range)

            # plt.plot(tmp_df["p"], tmp_df["pHat"], )
            color = next(ax._get_lines.prop_cycler)['color']
            plt.scatter(tmp_df["p"], tmp_df[f"{e}_error_{d}"], color=color, marker=".",
                        alpha=.1)
            plt.plot(B_range, pHat_smooth, color=color, label="OR: " + str(OR))

        plt.tick_params(axis='both', which='major', labelsize=font_size)
        plt.title(f"{E}", fontsize=font_size)
        plt.text(R0_max + 0.05, (y_max + y_min) /
                 2, f"day {d}", fontsize=font_size)

        plt.xlabel("$p$", x=-0.1, fontsize=font_size)
        plt.ylabel("$Log-Error$", fontsize=font_size)
        # plt.show()
        plt.legend(loc=(1, 0.1))
        plt.savefig(
            f"../figs/within_OR_error_{e}_day{d}.png", dpi=dpi, bbox_inches="tight")
        plt.close()
