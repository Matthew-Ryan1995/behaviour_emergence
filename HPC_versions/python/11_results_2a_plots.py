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
import os

import numpy as np
# import seaborn as sns
from scipy.interpolate import make_smoothing_spline
# from scipy.interpolate import RegularGridInterpolator
#

os.chdir(os.getcwd() + "/code")

params = {"ytick.color": "black",
          "xtick.color": "black",
          "axes.labelcolor": "black",
          "axes.edgecolor": "black",
          # "text.usetex": True,
          "font.family": "serif"}
plt.rcParams.update(params)
plt.rcParams['mathtext.fontset'] = 'dejavuserif'

# %% Load data
file_path = "../data/df_results2_Bstar_by_R0.csv"
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

# tmp = df[["Bstar", "R0", "pHat"]]
# tmp_pivot = tmp.pivot(index="Bstar", columns="R0", values="pHat")

# z = np.array(tmp_pivot)
# X = tmp_pivot.columns.values
# Y = tmp_pivot.index.values

# xx, yy = np.meshgrid(X, Y, indexing="ij")

# f = RegularGridInterpolator((xx, yy), z, method='cubic')
# xnew = np.arange(X.min(), X.max(), .001)
# ynew = np.arange(Y.min(), Y.max(), .001)
# data1 = f(xnew, ynew)
# Xn, Yn = np.meshgrid(xnew, ynew)
# plt.figure()
# plt.contourf(Xn, Yn, data1, cmap=plt.cm.Blues)
# plt.show()

# %%
tmp = df[["Bstar", "R0", "pHat"]]
tmp_pivot = tmp.pivot(index="Bstar", columns="R0", values="pHat")

z = np.array(tmp_pivot)
X = tmp_pivot.columns.values
Y = tmp_pivot.index.values

xx, yy = np.meshgrid(X, Y)

plt.figure()
im = plt.contourf(xx, yy, z,  cmap=plt.cm.Blues)
# ctr = plt.contour(xx, yy, z,
#                   # levels = lvls,
#                   # cmap=plt.cm.Blues,
#                   colors="black",
#                   alpha=0.5)
cbar = plt.colorbar(im, format=tkr.PercentFormatter(xmax=1, decimals=1))
cbar.ax.tick_params(labelsize=font_size)
# cbar_lvls = ctr.levels[1:-1]
# cbar.add_lines(ctr)
# cbar.set_ticks(cbar_lvls)

plt.plot([3.28, 3.28], [0, 1], "black")

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())


plt.xlabel("$\mathscr{R}_0^D$", fontsize=font_size)
plt.ylabel("$B^*$", fontsize=font_size)


plt.yticks([0.25, 0.5, 0.75], fontsize=font_size)
plt.text(-1.34, 0.95, "1.00", fontsize=font_size)
plt.text(-1.34, 0., "0.00", fontsize=font_size)

plt.xticks(fontsize=font_size)

plt.savefig("../figs/results2a_outbreak_prob.png",
            dpi=dpi, bbox_inches="tight")
plt.close()

# %%
tmp = df[["Bstar", "R0", "FS_conditional"]]
tmp_pivot = tmp.pivot(index="Bstar", columns="R0", values="FS_conditional")

z = np.array(tmp_pivot)
X = tmp_pivot.columns.values
Y = tmp_pivot.index.values

xx, yy = np.meshgrid(X, Y)

plt.figure()
im = plt.contourf(xx, yy, z,  cmap=plt.cm.Reds)
# ctr = plt.contour(xx, yy, z,
#                   # levels = lvls,
#                   # cmap=plt.cm.Blues,
#                   colors="black",
#                   alpha=0.5)
cbar = plt.colorbar(im, format=tkr.PercentFormatter(xmax=1, decimals=1))
# cbar_lvls = ctr.levels[1:-1]
# cbar.add_lines(ctr)
# cbar.set_ticks(cbar_lvls)

plt.plot([3.28, 3.28], [0, 1], "black")

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())

cbar.ax.tick_params(labelsize=font_size)


plt.xlabel("$\mathscr{R}_0^D$", fontsize=font_size)
plt.ylabel("$B^*$", fontsize=font_size)


plt.yticks([0.25, 0.5, 0.75], fontsize=font_size)
plt.text(-1.34, 0.95, "1.00", fontsize=font_size)
plt.text(-1.34, 0., "0.00", fontsize=font_size)

plt.xticks(fontsize=font_size)

plt.savefig("../figs/results2a_final_size.png", dpi=dpi, bbox_inches="tight")
plt.close()


# %%

error_type = ["exp", "poly"]
day = [5, 10, 15]

tmp_df = df[(df["Bstar"] <= 0.55) & (df["R0"] >= 1)]

vmin = np.log(
    tmp_df[[f"{e}_error_{d}" for e in error_type for d in day]].min().min())
vmax = np.log(
    tmp_df[[f"{e}_error_{d}" for e in error_type for d in day]].max().max())

fig, axes = plt.subplots(nrows=3, ncols=2, sharex=True, sharey=True)

normalizer = clrs.Normalize(0, 4)
im = cm.ScalarMappable(norm=normalizer)

R0_max = tmp_df["R0"].max()

c = 0
for d in day:
    for e in error_type:
        if e == "poly":
            E = "cubic"
        else:
            E = "exponential"
        ax = axes.flat[c]
        tmp = tmp_df[["Bstar", "R0", f"{e}_error_{d}"]]
        # tmp = tmp[(tmp["Bstar"] < 0.5) & (tmp["R0"] >= 1)]
        tmp_pivot = tmp.pivot(index="Bstar", columns="R0",
                              values=f"{e}_error_{d}")

        z = np.log(np.array(tmp_pivot))
        # z = z/z.max()
        X = tmp_pivot.columns.values
        Y = tmp_pivot.index.values

        xx, yy = np.meshgrid(X, Y)

        plt.figure()
        # im = plt.imshow(z,
        #                 origin='lower',
        #                 extent=[xx.min(), xx.max(), yy.min(), yy.max()],
        #                 aspect="auto")
        im_sub = ax.contourf(xx, yy, z, vmin=vmin, vmax=vmax, norm=normalizer)
        # ctr = plt.contour(xx, yy, z,
        #                   # levels = lvls,
        #                   # cmap=plt.cm.Blues,
        #                   colors="black",
        #                   alpha=0.5)
        # cbar = ax.colorbar(im)
        # im.set_clim(vmin, vmax)
        # cbar_lvls = ctr.levels[1:-1]
        # cbar.add_lines(ctr)
        # cbar.set_ticks(cbar_lvls)

        ax.plot([3.28, 3.28], [0, 0.5], "black")

        ax.set_xlim(xx.min(), xx.max())
        ax.set_xticks([1, 3, 5, 7, 9])

        ax.set_ylim(yy.min(), yy.max())
        ax.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5])

        ax.tick_params(axis='both', which='major', labelsize=font_size)
        if d == 5:
            ax.set_title(f"{E}", fontsize=font_size)
        if e == "poly":
            ax.text(R0_max + 0.5, 0.25, f"day {d}", fontsize=font_size)

        if (e == "poly") & (d == 15):
            ax.set_xlabel("$\mathscr{R}_0^D$", x=-0.1, fontsize=font_size)
        if (e == "exp") & (d == 10):
            ax.set_ylabel("$B^*$", fontsize=font_size)
        # plt.show()
        c += 1

cbar = fig.colorbar(im, ax=axes.ravel().tolist(), pad=0.22)
ticks = np.exp(cbar.get_ticks())
cbar.ax.tick_params(labelsize=font_size)
# cbar.ax.set_yticklabels(ticks)
# cbar.set_ticks(ticks)
# cbar.ax.yaxis.set_major_locator(tkr.LogLocator())
# cbar.ax.yaxis.set_major_formatter(tkr.LogFormatter())
# plt.show()
fig.savefig("../figs/results2a_error.png", dpi=dpi, bbox_inches="tight")
plt.close(fig)
