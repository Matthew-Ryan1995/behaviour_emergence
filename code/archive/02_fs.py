#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 11:01:14 2024

@author: rya200
"""
from BaD import *
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

# %%

params = load_param_defaults()

R0D = 1.5
R0B = 0.5

params["transmission"] = R0D
params["immune_period"] = 0
params["B_social"] = R0B * (params["N_social"] + params["N_const"])
params["B_fear"] = 16

M = bad(**params)

# %%

P = 1
Ib0, Rb0, Rn0 = np.zeros(3)
Sb0 = 1e-1  # 1 in a million seeded with behaviour
In0 = 1e-6  # 1 in a million seeded with disease
# Ib0, Rb0, Rn0 = np.zeros(3)
# Sb0 = 1-0.6951793156273507  # 1 in a million seeded with behaviour
# In0 = Ib0 = 1e-6  # 1 in a million seeded with disease

Sn0 = P - Sb0 - Ib0 - Rb0 - In0 - Rn0

PP = np.array([Sn0, Sb0, In0, Ib0, Rn0, Rb0])

M.run(IC=PP, t_start=0, t_end=100)

plt.figure()
plt.plot(M.get_I(), label="I")
plt.plot(M.get_S(), label="S")
plt.plot(M.get_B(), label="B")
plt.legend()
plt.show()

# %%
true_fs = P - M.get_S()[-1]


def fs_N(x):
    ans = x - np.exp(-R0D * (1-x))
    return ans


def fs_B(x):
    ans = x - np.exp(-R0D * (1-x) *
                     params["susc_B_efficacy"] * params["inf_B_efficacy"])
    return ans


pi_N = P - fsolve(func=fs_N, x0=0.5)[0]
pi_B = P - fsolve(func=fs_B, x0=0.5)[0]

print(f"lower bound is {pi_B}")
print(f"obs value us {true_fs}")
print(f"Upper bound is {pi_N}")

# %%


def find_z(z):
    ans = true_fs - (z*pi_B + (1-z) * pi_N)
    return ans


z_val = fsolve(find_z, x0=0)[0]

M.endemic_behaviour()

N_star = M.Nstar
B_star = 1-N_star

omega_star = B_star * M.B_social + M.B_const
alpha_star = N_star * M.N_social + M.N_const

gamma = 1/M.infectious_period

omega_hat = (omega_star + gamma)/(omega_star + alpha_star + gamma)
alpha_hat = (alpha_star + gamma)/(omega_star + alpha_star + gamma)

# %%


def niave_fs(x):
    ans = np.zeros(2)

    N = PP[[0, 2, 4]].sum()
    B = 1-N

    ans[0] = x[0] - PP[0] * \
        np.exp(-R0D * (N-x[0])-(1-params["inf_B_efficacy"]) * R0D * (B - x[1]))
    ans[1] = x[1] - PP[1] * np.exp(-(1-params["susc_B_efficacy"]) * R0D * (N-x[0])-(
        1-params["susc_B_efficacy"])*(1-params["inf_B_efficacy"]) * R0D * (B - x[1]))
    return ans


niave_f = fsolve(func=niave_fs, x0=[0, 0])

print(((1-niave_f)*PP[[0, 1]]).sum())
