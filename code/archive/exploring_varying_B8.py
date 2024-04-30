#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 13:44:31 2024

@author: rya200
"""
from BaD import *
import numpy as np
from scipy.optimize import fsolve
import gillespy2
from datetime import datetime
import matplotlib.pyplot as plt


# %%

params = load_param_defaults()

# %%


def get_w1(Bstar, params):
    a1 = params["N_social"]
    w3 = params["B_const"]

    k = a1 + w3 + params["N_const"]

    def solve_w1(x):
        ans = ((1-2*Bstar)**2 - 1) * x**2 + (2*(2*Bstar*a1 - k)*(1-2*Bstar) -
                                             (4*w3 - 2*k)) * x + ((2*Bstar*a1 - k)**2 - (k**2 - 4*a1*w3))
        return ans

    w1 = fsolve(solve_w1, x0=[0, 1, 2])
    return w1.max()


def bad_ctmc(param_vals, P=100, I0=1, B0=1, t_end=100):

    model = gillespy2.Model()

    # Parameters
    gamma = gillespy2.Parameter(
        name="gamma", expression=1/param_vals["infectious_period"])
    beta_nn = gillespy2.Parameter(
        name="beta_nn", expression=param_vals["transmission"] / P)
    beta_nb = gillespy2.Parameter(name="beta_nb", expression=(
        1 - param_vals["susc_B_efficacy"]) * param_vals["transmission"] / P)
    beta_bn = gillespy2.Parameter(name="beta_bn", expression=(
        1 - param_vals["inf_B_efficacy"]) * param_vals["transmission"] / P)
    beta_bb = gillespy2.Parameter(name="beta_bb", expression=(
        1 - param_vals["inf_B_efficacy"]) * (1 - param_vals["susc_B_efficacy"]) * param_vals["transmission"] / P)
    # p = gillespy2.Parameter(name="p", expression=params["inf_B_efficacy"])
    # c = gillespy2.Parameter(name="c", expression=params["susc_B_efficacy"])
    w1 = gillespy2.Parameter(name="w1", expression=param_vals["B_social"] / P)
    w2 = gillespy2.Parameter(name="w2", expression=param_vals["B_fear"] / P)
    w3 = gillespy2.Parameter(name="w3", expression=param_vals["B_const"])
    a1 = gillespy2.Parameter(name="a1", expression=param_vals["N_social"] / P)
    a2 = gillespy2.Parameter(name="a2", expression=param_vals["N_const"])

    model.add_parameter([gamma, beta_nn, beta_nb, beta_bn, beta_bb,
                         w1, w2, w3, a1, a2])

    # Species
    Sn = gillespy2.Species(name="Sn", initial_value=P - I0 - B0)
    In = gillespy2.Species(name="In", initial_value=I0)
    Rn = gillespy2.Species(name="Rn", initial_value=0)
    Sb = gillespy2.Species(name="Sb", initial_value=B0)
    Ib = gillespy2.Species(name="Ib", initial_value=0)
    Rb = gillespy2.Species(name="Rb", initial_value=0)

    model.add_species([Sn, Sb, In, Ib, Rn, Rb])

    # Reactions

    # Sn
    sn_to_sb_social = gillespy2.Reaction(name="sn_to_sb_social",
                                         reactants={Sn: 1}, products={Sb: 1},
                                         propensity_function="w1 * (Sb + Ib + Rb) * Sn")
    sn_to_sb_fear = gillespy2.Reaction(name="sn_to_sb_fear",
                                       reactants={Sn: 1}, products={Sb: 1},
                                       propensity_function="w2 * (In + Ib) * Sn")
    sn_to_sb_const = gillespy2.Reaction(name="sn_to_sb_const",
                                        reactants={Sn: 1}, products={Sb: 1},
                                        rate=w3)

    # Sb
    sb_to_sn_social = gillespy2.Reaction(name="sb_to_sn_social",
                                         reactants={Sb: 1}, products={Sn: 1},
                                         propensity_function="a1 * (Sn + In + Rn) * Sb")
    sb_to_sn_const = gillespy2.Reaction(name="sb_to_sn_const",
                                        reactants={Sb: 1}, products={Sn: 1},
                                        rate=a2)

    model.add_reaction([sn_to_sb_social, sn_to_sb_fear, sn_to_sb_const,
                        sb_to_sn_social, sb_to_sn_const])

    # In
    in_infect_sn = gillespy2.Reaction(name="in_infect_sn",
                                      reactants={Sn: 1, In: 1}, products={In: 2},
                                      rate=beta_nn)
    in_infect_sb = gillespy2.Reaction(name="in_infect_sb",
                                      reactants={Sb: 1, In: 1}, products={In: 1, Ib: 1},
                                      rate=beta_nb)

    in_recover = gillespy2.Reaction(name="in_recover",
                                    reactants={In: 1}, products={Rn: 1},
                                    rate=gamma)

    in_to_ib_social = gillespy2.Reaction(name="in_to_ib_social",
                                         reactants={In: 1}, products={Ib: 1},
                                         propensity_function="w1 * (Sb + Ib + Rb) * In")
    in_to_ib_fear = gillespy2.Reaction(name="in_to_ib_fear",
                                       reactants={In: 1}, products={Ib: 1},
                                       propensity_function="w2 * (In + Ib) * In")
    in_to_ib_const = gillespy2.Reaction(name="in_to_ib_const",
                                        reactants={In: 1}, products={Ib: 1},
                                        rate=w3)

    # Ib
    ib_infect_sn = gillespy2.Reaction(name="ib_infect_sn",
                                      reactants={Sn: 1, Ib: 1}, products={In: 1, Ib: 1},
                                      rate=beta_bn)
    ib_infect_sb = gillespy2.Reaction(name="ib_infect_sb",
                                      reactants={Sb: 1, Ib: 1}, products={Ib: 2},
                                      rate=beta_bb)

    ib_recover = gillespy2.Reaction(name="ib_recover",
                                    reactants={Ib: 1}, products={Rb: 1},
                                    rate=gamma)

    ib_to_in_social = gillespy2.Reaction(name="ib_to_in_social",
                                         reactants={Ib: 1}, products={In: 1},
                                         propensity_function="a1 * (Sn + In + Rn) * Ib")
    ib_to_in_const = gillespy2.Reaction(name="ib_to_in_const",
                                        reactants={Ib: 1}, products={In: 1},
                                        rate=a2)

    model.add_reaction([in_infect_sn, in_infect_sb, in_recover,
                        in_to_ib_social, in_to_ib_fear, in_to_ib_const,
                        ib_infect_sn, ib_infect_sb, ib_recover,
                        ib_to_in_social, ib_to_in_const
                        ])

    # Rn

    rn_to_rb_social = gillespy2.Reaction(name="rn_to_rb_social",
                                         reactants={Rn: 1}, products={Rb: 1},
                                         propensity_function="w1 * (Sb + Ib + Rb) * Rn")
    rn_to_rb_fear = gillespy2.Reaction(name="rn_to_rb_fear",
                                       reactants={Rn: 1}, products={Rb: 1},
                                       propensity_function="w2 * (In + Ib) * Rn")
    rn_to_rb_const = gillespy2.Reaction(name="rn_to_rb_const",
                                        reactants={Rn: 1}, products={Rb: 1},
                                        rate=w3)

    # Rb

    rb_to_rn_social = gillespy2.Reaction(name="rb_to_rn_social",
                                         reactants={Rb: 1}, products={Rn: 1},
                                         propensity_function="a1 * (Sn + In + Rn) * Rb")
    rb_to_rn_const = gillespy2.Reaction(name="rb_to_rn_const",
                                        reactants={Rb: 1}, products={Rn: 1},
                                        rate=a2)

    model.add_reaction([rn_to_rb_social, rn_to_rb_fear, rn_to_rb_const,
                        rb_to_rn_social, rb_to_rn_const])

    tspan = gillespy2.TimeSpan.linspace(t=t_end, num_points=t_end + 1)
    model.timespan(tspan)
    return model


def get_outbreak(dlr):
    I = dlr["In"] + dlr["Ib"]

    ans = 0
    if I.max() > int(0.001*P):
        ans = 1

    return ans


def get_outbreak_probabilities(params):
    B_star_vec = np.arange(0, 1, step=0.05)

    params_new = dict(params)
    outbreak_probs = []
    lwr = []
    upr = []

    for b in B_star_vec:

        params_new["B_social"] = get_w1(b, params_new)

        num_trajectory = 100

        B0 = int(b * P)

        # t_end = 300
        model = bad_ctmc(param_vals=params_new, P=P, I0=I0, B0=B0, t_end=t_end)
        results = model.run(number_of_trajectories=num_trajectory)

        tmp = list(map(get_outbreak, results))
        tmp = np.array(tmp)

        outbreak_probabilty = tmp.sum()/num_trajectory

        moe = 1.96 * np.sqrt(outbreak_probabilty *
                             (1-outbreak_probabilty) / num_trajectory)
        lwr.append(outbreak_probabilty - moe)
        upr.append(outbreak_probabilty + moe)

        outbreak_probs.append(outbreak_probabilty)

    return B_star_vec, outbreak_probs, lwr, upr


# %%

params["infectious_period"] = 5
params["transmission"] = 3/params["infectious_period"]
params["immune_period"] = 0

params["B_fear"] = 8/params["infectious_period"]
params["N_social"] = 0.4 / params["infectious_period"]
params["B_const"] = 0.01 / params["infectious_period"]
params["N_const"] = 0.6 / params["infectious_period"]

params["B_social"] = (
    1.7 * (params["N_social"] + params["N_const"]))
# params["B_social"] = 0.08

# params["inf_B_efficacy"] = 0
# params["susc_B_efficacy"] = 0
# beta = params["transmission"]

# gamma = 1/params["infectious_period"]
# p = params["inf_B_efficacy"]
# c = params["susc_B_efficacy"]
# w1 = params["B_social"]
# w2 = params["B_fear"]
# w3 = params["B_const"]
# a1 = params["N_social"]
# a2 = params["N_const"]

Bstar = 0.02
params["B_social"] = get_w1(Bstar, params)

P = 10000
I0 = 1
B0 = 1

N0 = P - B0

M = bad(**params)

Sn = P - I0 - B0
IC = np.array([Sn, B0, I0, 0, 0, 0]) / P
t_start, t_end = [0, 100]
M.run(IC=IC, t_start=t_start, t_end=t_end)

M.endemic_behaviour(I_eval=0)
print(1-M.Nstar)

# %%

B_test, prob_test, lwr, upr = get_outbreak_probabilities(params)

# %%
plt.figure()
plt.fill_between(B_test, lwr, upr, alpha=0.1, color="blue")
plt.plot(B_test, prob_test, color="black")
plt.xlabel("Bstar")
plt.ylabel("Outbreak Probability")
plt.show()


# %%
# Bstar = 0.9
# a1 = params["N_social"]
# a2 = params["N_const"]
# w3 = params["B_const"]

# k = a1 + w3 + params["N_const"]


# def solve_w1(x):
#     ans = (((1-2*Bstar)**2 - 1) * x**2 + (2*(2*Bstar*a1 - k)*(1-2*Bstar) -
#                                           (4*w3 - 2*k)) * x + ((2*Bstar*a1 - k)**2 - (k**2 - 4*a1*w3)))
#     return ans

# # def solve_w1(x):
# #     ans = x - ((1-Bstar)*a1 + a2) / (1-Bstar)
# #     return ans


# w1 = fsolve(solve_w1, x0=[-5, 5])
