#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 08:44:09 2024

@author: rya200
"""

# %% Packages

import matplotlib.pyplot as plt
import gillespy2
import numpy as np
from scipy.integrate import solve_ivp
from datetime import datetime
start_time = datetime.now()

# %% Functions


def behaviour_sis(parameter_values=None, t_end=20, P=100, B_0=1):

    model = gillespy2.Model(name="b_sis")

    if parameter_values is not None:
        params = parameter_values
    else:
        params = {"B_social": 2, "B_const": 1, "N_social": 2, "N_const": 1}

    # Params
    w1 = gillespy2.Parameter(name="w1", expression=params["B_social"])
    w3 = gillespy2.Parameter(name="w3", expression=params["B_const"])
    a1 = gillespy2.Parameter(name="a1", expression=params["N_social"])
    a2 = gillespy2.Parameter(name="a2", expression=params["N_const"])

    model.add_parameter([w1, w3, a1, a2])

    # States
    N = gillespy2.Species(name="N", initial_value=P-B_0)
    B = gillespy2.Species(name="B", initial_value=B_0)

    model.add_species([N, B])

    # Reactions
    b_social = gillespy2.Reaction(name="b_social",
                                  reactants={N: 1, B: 1},
                                  products={B: 2},
                                  # rate=w1,
                                  propensity_function="w1 * N * B / (N + B - 1)"
                                  )
    b_const = gillespy2.Reaction(name="b_const",
                                 reactants={N: 1},
                                 products={B: 1},
                                 rate=w3)
    n_social = gillespy2.Reaction(name="n_social",
                                  reactants={N: 1, B: 1},
                                  products={N: 2},
                                  # rate=a1,
                                  propensity_function="a1 * N * B / (N + B - 1)"
                                  )
    n_const = gillespy2.Reaction(name="n_const",
                                 reactants={B: 1},
                                 products={N: 1},
                                 rate=a2)

    model.add_reaction([b_social, b_const, n_social, n_const])

    tspan = gillespy2.TimeSpan.linspace(t=t_end, num_points=t_end + 1)
    model.timespan(tspan)
    return model


num_trajectory = 200

t_end = 100


w1 = 0.1
w3 = 0.10
a1 = 1.2
a2 = 0.5

P = 10000
B = 1

N = P - B
IC = [N, B]


params = {"B_social": w1, "B_const": w3, "N_social": a1, "N_const": a2}

model = behaviour_sis(parameter_values=params, t_end=t_end, P=P, B_0=B)
results = model.run(number_of_trajectories=num_trajectory)

# %%

results.plot()

# %%


def odes(t, PP):
    Y = np.zeros(2)

    Y[0] = -(w1 * PP[1] / (P-1) + w3) * PP[0] + \
        (a1 * PP[0] / (P - 1) + a2) * PP[1]
    Y[1] = +(w1 * PP[1] / (P-1) + w3) * PP[0] - \
        (a1 * PP[0] / (P - 1) + a2) * PP[1]
    return Y


t_start = 0

t_span = np.arange(t_start, t_end + 1, step=1, )

res = solve_ivp(fun=odes, t_span=[t_start, t_end], y0=IC, t_eval=t_span)

dat = res.y.T

# %%

plt.figure()
plt.plot(dat[:, 1], label="B")
plt.plot(dat[:, 0], label="N")
plt.legend()
plt.show()

print(f"Time taken: {datetime.now() - start_time}")
