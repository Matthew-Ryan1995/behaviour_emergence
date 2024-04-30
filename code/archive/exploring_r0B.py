#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 09:58:04 2024

@author: rya200
"""
from BaD import *
import matplotlib.pyplot as plt

# %%

params = load_param_defaults()
params["transmision"] = 0
params["B_const"] = 0

R0b = [0., 0.5, 1, 1.1, 1.8, 2.5]

denom = params["N_const"] + params["N_social"]

# %%
P = 1

In = 0
Ib = 0
Sb = 1e-6
Rn = 0
Rb = 0

Sn = P - In - Ib - Sb - Rn - Rb

PP = [Sn, Sb, In, Ib, Rn, Rb]

t_start = 0
t_end = 50

# %%

plt.figure()

for idx in range(len(R0b)):
    params["B_social"] = R0b[idx] * denom

    M = bad(**params)
    M.run(IC=PP, t_start=t_start, t_end=t_end)

    plt.plot(M.t_range, M.get_B(), label=str(R0b[idx]))
    plt.ylim(0, 1)

plt.legend()
plt.show()
