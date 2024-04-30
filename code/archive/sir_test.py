#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 09:16:36 2024

@author: rya200
"""

import gillespy2
import numpy as np
import matplotlib.pyplot as plt

# %%

P = 1e5
I0 = 1

gamma = 1/5
beta = 2 * gamma


def sir_stoch(beta, gamma, P, I0, t_end=30):

    model = gillespy2.Model(name="SIR")

    beta = gillespy2.Parameter(name="beta", expression=beta)
    sigma = gillespy2.Parameter(name="sigma", expression=gamma * 2)
    gamma = gillespy2.Parameter(name="gamma", expression=gamma)

    model.add_parameter([beta, gamma, sigma])

    S = gillespy2.Species(name="S", initial_value=P-I0)
    E = gillespy2.Species(name="E", initial_value=0)
    I = gillespy2.Species(name="I", initial_value=I0)
    R = gillespy2.Species(name="R", initial_value=0)

    model.add_species([S, I, R, E])

    infect = gillespy2.Reaction(name="infect",
                                reactants={S: 1, I: 1},
                                products={E: 1, I: 1},
                                propensity_function="beta * S * I / (S + I + R)")
    exposed_infectious = gillespy2.Reaction(name="exposed_infectious",
                                            reactants={E: 1},
                                            products={I: 1},
                                            rate=sigma)
    recover = gillespy2.Reaction(name="recover",
                                 reactants={I: 1},
                                 products={R: 1},
                                 rate=gamma)

    model.add_reaction([infect, exposed_infectious, recover])

    tspan = gillespy2.TimeSpan.linspace(t=t_end, num_points=t_end + 1)
    model.timespan(tspan)
    return model
# def sir_stoch(beta, gamma, P, I0, t_end=30):

#     model = gillespy2.Model(name="SIR")

#     beta = gillespy2.Parameter(name="beta", expression=beta)
#     gamma = gillespy2.Parameter(name="gamma", expression=gamma)

#     model.add_parameter([beta, gamma])

#     S = gillespy2.Species(name="S", initial_value=P-I0)
#     I = gillespy2.Species(name="I", initial_value=I0)
#     R = gillespy2.Species(name="R", initial_value=0)

#     model.add_species([S, I, R])

#     infect = gillespy2.Reaction(name="infect",
#                                 reactants={S: 1, I: 1},
#                                 products={I: 2},
#                                 propensity_function="beta * S * I / (S + I + R)")
#     recover = gillespy2.Reaction(name="recover",
#                                  reactants={I: 1},
#                                  products={R: 1},
#                                  rate=gamma)

#     model.add_reaction([infect, recover])

#     tspan = gillespy2.TimeSpan.linspace(t=t_end, num_points=t_end + 1)
#     model.timespan(tspan)
#     return model


num_trajectories = 100
t_end = 300

M = sir_stoch(beta=beta, gamma=gamma, P=P, I0=I0, t_end=t_end)
results = M.run(number_of_trajectories=num_trajectories)

# %%
plt.figure()
for idx in range(num_trajectories):
    trajectory = results[idx]
    plt.plot(trajectory["time"], trajectory["I"]/P, color="red", alpha=0.2)

lam = (-(beta + gamma) + np.sqrt((beta - gamma)**2 + 4 * beta * beta)) / 2
f = (I0/P) * np.exp(lam * trajectory["time"])
tt2 = [i for i in range(len(f)) if f[i] < 1]
plt.plot(tt2, f[tt2],
         color="grey")
plt.xlabel("time")
plt.ylabel("Count")
plt.legend(["Infections"])
plt.show()
