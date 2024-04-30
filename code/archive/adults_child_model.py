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
params["transmission"] = 2

# %%
# beta = params["transmission"]
# beta = 2

params["transmission"] = 2
# params["immune_period"] = 1e6
# params["B_social"] = 0
# params["B_fear"] = 0
# params["N_social"] = 0
# params["B_const"] = 0
# params["N_const"] = 0
# beta = params["transmission"]

# gamma = 1/params["infectious_period"]
# p = params["inf_B_efficacy"]
# c = params["susc_B_efficacy"]
# w1 = params["B_social"]
# w2 = params["B_fear"]
# w3 = params["B_const"]
# a1 = params["N_social"]
# a2 = params["N_const"]

# %%

# N - Child
# B - Adult

P = 10000
I0 = 1
B0 = P/2

N0 = P - B0

beta = params["transmission"]
p = params["inf_B_efficacy"]
c = params["susc_B_efficacy"]
gamma = 1/params["infectious_period"]

beta_cc = beta
beta_ac = (1-p) * beta
beta_ca = (1-c) * beta
beta_aa = (1-c) * (1-p) * beta


def eqns_to_solve(x):

    q1 = x[0]
    q2 = x[1]

    child_denom = beta_cc + beta_ca + gamma
    adult_denom = beta_ac + beta_aa + gamma

    cc_val = beta_cc / child_denom
    ca_val = beta_ca/child_denom
    c_gamma = gamma/child_denom

    ac_val = beta_ac/adult_denom
    aa_val = beta_aa/adult_denom
    a_gamma = gamma/adult_denom

    ans = np.zeros(2)

    ans[0] = q1 - (cc_val * q1**2 + ca_val * q1*q2 + c_gamma)
    ans[1] = q2 - (ac_val * q1*q2 + aa_val * q2**2 + a_gamma)

    return ans


no_major_in_niave = fsolve(func=eqns_to_solve, x0=np.array([0, 0]))

prob_outbreak_in_niave = 1 - no_major_in_niave

tmp_params = {
    "beta_cc": beta_cc,
    "beta_ac": beta_ac,
    "beta_ca": beta_ca,
    "beta_aa": beta_aa,
    "gamma": gamma
}

# %%


def bad_ctmc(param_vals, P=100, I0=1, B0=1, t_end=100):

    N_c = P-B0
    N_a = B0

    model = gillespy2.Model()

    # Parameters
    gamma = gillespy2.Parameter(
        name="gamma", expression=param_vals["gamma"])
    beta_cc = gillespy2.Parameter(
        name="beta_cc", expression=param_vals["beta_cc"] / N_c)
    beta_ac = gillespy2.Parameter(
        name="beta_ac", expression=param_vals["beta_ac"] / N_c)
    beta_ca = gillespy2.Parameter(
        name="beta_ca", expression=param_vals["beta_ca"] / N_a)
    beta_aa = gillespy2.Parameter(
        name="beta_aa", expression=param_vals["beta_aa"] / N_a)

    model.add_parameter([gamma, beta_cc, beta_ac, beta_ca, beta_aa])

    # Species
    Sn = gillespy2.Species(name="Sc", initial_value=N_c)
    In = gillespy2.Species(name="Ic", initial_value=0)
    Rn = gillespy2.Species(name="Rc", initial_value=0)
    Sb = gillespy2.Species(name="Sa", initial_value=N_a-I0)
    Ib = gillespy2.Species(name="Ia", initial_value=I0)
    Rb = gillespy2.Species(name="Ra", initial_value=0)

    model.add_species([Sn, Sb, In, Ib, Rn, Rb])

    # Reactions

    # In
    in_infect_sn = gillespy2.Reaction(name="in_infect_sn",
                                      reactants={Sn: 1, In: 1}, products={In: 2},
                                      rate=beta_cc)
    in_infect_sb = gillespy2.Reaction(name="in_infect_sb",
                                      reactants={Sb: 1, In: 1}, products={In: 1, Ib: 1},
                                      rate=beta_ca)

    in_recover = gillespy2.Reaction(name="in_recover",
                                    reactants={In: 1}, products={Rn: 1},
                                    rate=gamma)

    # Ib
    ib_infect_sn = gillespy2.Reaction(name="ib_infect_sn",
                                      reactants={Sn: 1, Ib: 1}, products={In: 1, Ib: 1},
                                      rate=beta_ac)
    ib_infect_sb = gillespy2.Reaction(name="ib_infect_sb",
                                      reactants={Sb: 1, Ib: 1}, products={Ib: 2},
                                      rate=beta_aa)

    ib_recover = gillespy2.Reaction(name="ib_recover",
                                    reactants={Ib: 1}, products={Rb: 1},
                                    rate=gamma)

    model.add_reaction([in_infect_sn, in_infect_sb, in_recover,
                        ib_infect_sn, ib_infect_sb, ib_recover])

    tspan = gillespy2.TimeSpan.linspace(t=t_end, num_points=t_end + 1)
    model.timespan(tspan)
    return model

# %%


num_trajectory = 500

t_end = 60


model = bad_ctmc(param_vals=tmp_params, P=P, I0=I0, B0=B0, t_end=t_end)
results = model.run(number_of_trajectories=num_trajectory)

# %%
results.plot()

# %%

count = 0
for index in range(0, num_trajectory):
    trajectory = results[index]
    if (trajectory["Sc"][-1] + trajectory["Sa"][-1]) > (P-10):
        count += 1

print(count/num_trajectory)

print(f"Time taken: {datetime.now()-start_time}")

print(no_major_in_niave)
