#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 11:40:52 2024

@author: rya200
"""


import gillespy2

# %%


def SIR(parameter_values=None, t_end=20, N=100, I_0=1):

    model = gillespy2.Model(name="SIRS")

    if parameter_values is not None:
        params = parameter_values
    else:
        params = {"transmission": 2, "recovery_rate": 1, "immune_rate": 1/40}

    # Params
    beta = gillespy2.Parameter(
        name="beta", expression=params["transmission"]/(N-1))  # If N is constant, this works.  But if N is not constant, this does not work
    # beta = gillespy2.Parameter(
    #     name="beta", expression=params["transmission"])
    gamma = gillespy2.Parameter(
        name="gamma", expression=params["recovery_rate"])
    nu = gillespy2.Parameter(
        name="nu", expression=params["immune_rate"])

    model.add_parameter([beta, gamma, nu])

    # States
    S = gillespy2.Species(name="Susceptible", initial_value=N-I_0)
    I = gillespy2.Species(name="Infectious", initial_value=I_0)
    R = gillespy2.Species(name="Recovered", initial_value=0)

    model.add_species([S, I, R])

    beta2 = gillespy2.Parameter(
        name="b2", expression="beta/(self.Susceptible + Infectious + Recovered  - 1)")

    model.add_parameter([beta2])

    # Reactions
    s_to_i = gillespy2.Reaction(name="infect",
                                reactants={S: 1, I: 1},
                                products={I: 2},
                                rate=beta
                                )
    i_to_r = gillespy2.Reaction(name="recover",
                                reactants={I: 1},
                                products={R: 1},
                                rate=gamma)
    r_to_s = gillespy2.Reaction(name="wane_immunity",
                                reactants={R: 1},
                                products={S: 1},
                                rate=nu)

    model.add_reaction([s_to_i, i_to_r, r_to_s])

    tspan = gillespy2.TimeSpan.linspace(t=t_end, num_points=t_end + 1)
    model.timespan(tspan)
    return model


num_trajectory = 200

t_end = 200


N = 100
I = 1

beta = 2
gamma = 1
nu = 1/40


params = {"transmission": beta, "recovery_rate": gamma, "immune_rate": nu}

model = SIR(parameter_values=params, t_end=t_end, N=N, I_0=I)
results = model.run(number_of_trajectories=num_trajectory)


results.plot()
