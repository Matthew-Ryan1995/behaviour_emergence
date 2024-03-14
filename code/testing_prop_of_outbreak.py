#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 11:45:15 2024

My equations aren't quite correct.  Need to account for the social and possibly the fear as well.
I shouldn't need to account for fear, cause this is all about early onset.

@author: rya200
"""
from BaD import *
import numpy as np
from scipy.optimize import fsolve
import gillespy2
from datetime import datetime

start_time = datetime.now()

# %%

params = load_param_defaults()

# %%
# beta = params["transmission"]
# beta = 2

params["transmission"] = 2
params["immune_period"] = 1e6
params["B_fear"] = 20
beta = params["transmission"]

gamma = 1/params["infectious_period"]
p = params["inf_B_efficacy"]
c = params["susc_B_efficacy"]
w1 = params["B_social"]
w2 = params["B_fear"]
w3 = params["B_const"]
a1 = params["N_social"]
a2 = params["N_const"]


def eqns_to_solve(x):

    q1 = x[0]
    q2 = x[1]

    p1N = beta/(beta + gamma + w3)
    p3N = gamma/(beta + gamma + w3)
    p4N = w3/(beta + gamma + w3)

    p1B = ((1-p) * beta) / ((1-p) * beta + gamma + a1 + a2)
    p3B = gamma / ((1-p) * beta + gamma + a1 + a2)
    p4B = (a1 + a2) / ((1-p) * beta + gamma + a1 + a2)

    ans = np.zeros(2)

    ans[0] = q1 - (q1**2 * p1N + p3N + q2 * p4N)
    ans[1] = q2 - (q1 * q2 * p1B + p3B + q1 * p4B)

    return ans


no_major_in_niave = fsolve(func=eqns_to_solve, x0=np.array([0, 0]))

prob_outbreak_in_niave = 1 - no_major_in_niave

# %%


def bad_ctmc(param_vals, P=100, I0=1, B0=1, t_end=100):

    model = gillespy2.Model()

    # Parameters
    gamma = gillespy2.Parameter(
        name="gamma", expression=1/param_vals["infectious_period"])
    nu = gillespy2.Parameter(name="nu", expression=1 /
                             param_vals["immune_period"])
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

    model.add_parameter([gamma, nu, beta_nn, beta_nb, beta_bn, beta_bb,
                         w1, w2, w3, a1, a2])

    # Species
    Sn = gillespy2.Species(name="Sn", initial_value=P - I0 - B0)
    In = gillespy2.Species(name="In", initial_value=I0)
    Rn = gillespy2.Species(name="Rn", initial_value=0)
    Sb = gillespy2.Species(name="Sb", initial_value=0)
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
                        ib_to_in_social, ib_to_in_const])

    # Rn
    rn_wane = gillespy2.Reaction(name="rn_wane",
                                 reactants={Rn: 1}, products={Sn: 1},
                                 rate=nu)

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
    rb_wane = gillespy2.Reaction(name="rb_wane",
                                 reactants={Rb: 1}, products={Sb: 1},
                                 rate=nu)

    rb_to_rn_social = gillespy2.Reaction(name="rb_to_rn_social",
                                         reactants={Rb: 1}, products={Rn: 1},
                                         propensity_function="a1 * (Sn + In + Rn) * Rb")
    rb_to_rn_const = gillespy2.Reaction(name="rb_to_rn_const",
                                        reactants={Rb: 1}, products={Rn: 1},
                                        rate=a2)

    model.add_reaction([rn_wane, rn_to_rb_social, rn_to_rb_fear, rn_to_rb_const,
                        rb_wane, rb_to_rn_social, rb_to_rn_const])

    tspan = gillespy2.TimeSpan.linspace(t=t_end, num_points=t_end + 1)
    model.timespan(tspan)
    return model

# %%


num_trajectory = 500

t_end = 60


P = 10000
I0 = 1
B0 = 1


model = bad_ctmc(param_vals=params, P=P, I0=I0, B0=B0, t_end=t_end)
results = model.run(number_of_trajectories=num_trajectory)

# %%
results.plot()

# %%

count = 0
for index in range(0, num_trajectory):
    trajectory = results[index]
    if (trajectory["Sn"][-1] + trajectory["Sb"][-1]) > (P-10):
        count += 1

print(count/num_trajectory)

print(f"Time taken: {datetime.now()-start_time}")

print(no_major_in_niave)
