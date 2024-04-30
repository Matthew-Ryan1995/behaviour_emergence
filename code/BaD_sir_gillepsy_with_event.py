#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 11:45:15 2024

Why is event making it so slow? Hybrid solver.  Small speed up by forcing to be discrete

Should I be comparing distance to output across a common time or across all time?

@author: rya200
"""
import json
from scipy.integrate import simpson
from BaD import *
import numpy as np
from scipy.optimize import fsolve
import gillespy2
from datetime import datetime
import matplotlib.pyplot as plt


# %%

params = load_param_defaults()

# %%
# beta = params["transmission"]
# beta = 2

params["infectious_period"] = 5
params["transmission"] = 3/params["infectious_period"]
params["immune_period"] = 0

params["B_fear"] = 8/params["infectious_period"]
params["N_social"] = 0.4 / params["infectious_period"]
params["B_const"] = 0.0 / params["infectious_period"]
params["N_const"] = 0.6 / params["infectious_period"]

params["B_social"] = (
    1.1 * (params["N_social"] + params["N_const"]))

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
print(M.Nstar)


# %%


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
    qrate = gillespy2.Parameter(name="qrate", expression=4)

    model.add_parameter([gamma, beta_nn, beta_nb, beta_bn, beta_bb,
                         w1, w2, w3, a1, a2, qrate])

    # Species
    Sn = gillespy2.Species(name="Sn", initial_value=P -
                           I0 - B0, mode="discrete")
    In = gillespy2.Species(name="In", initial_value=I0, mode="discrete")
    Rn = gillespy2.Species(name="Rn", initial_value=0, mode="discrete")
    Sb = gillespy2.Species(name="Sb", initial_value=B0, mode="discrete")
    Ib = gillespy2.Species(name="Ib", initial_value=0, mode="discrete")
    Rb = gillespy2.Species(name="Rb", initial_value=0, mode="discrete")

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

    # Events
    intervention_trigger = gillespy2.EventTrigger(expression="t>=10")
    intervention_reaction = gillespy2.EventAssignment(
        variable=model.listOfParameters["w1"], expression="qrate * w1")
    intervention = gillespy2.Event(name="intervention", trigger=intervention_trigger,
                                   assignments=[intervention_reaction])
    model.add_event(intervention)

    tspan = gillespy2.TimeSpan.linspace(t=t_end, num_points=t_end + 1)
    model.timespan(tspan)
    return model

# %%


num_trajectory = 100

# t_end = 300

start_time = datetime.now()

model = bad_ctmc(param_vals=params, P=P, I0=I0, B0=B0, t_end=t_end)
results = model.run(number_of_trajectories=num_trajectory)

print(f"Time taken: {datetime.now()-start_time}")

# %%
plt.figure()
B_avg = np.zeros(101)
B_med = np.zeros(101)

n_count = 0
for idx in range(num_trajectory):
    trajectory = results[idx]
    B = (trajectory["Sb"] + trajectory["Ib"] + trajectory["Rb"]) / P
    if B[-1] > 0.001:
        B_avg += B
        n_count += 1
        B_med = np.column_stack((B_med, B))
    I = (trajectory["Ib"] + trajectory["In"]) / P
    plt.plot(trajectory["time"], B, color="blue", alpha=0.2)
    plt.plot(trajectory["time"], I, color="red", alpha=0.2)
tmp = early_behaviour_dynamics(M)
tt = [i for i in range(len(tmp)) if tmp[i] < 1]
plt.plot(range(tt[-1] + 1), tmp[tt], color="orange")

tmp2 = early_behaviour_dynamics(M, method="poly", M=3)
tt = [i for i in range(len(tmp2)) if tmp2[i] < 1]
plt.plot(range(tt[-1] + 1), tmp2[tt], color="black")
tmp2 = early_behaviour_dynamics(M, method="poly", M=2)
tt = [i for i in range(len(tmp2)) if tmp2[i] < 1]
plt.plot(range(tt[-1] + 1), tmp2[tt], color="black")
tmp2 = early_behaviour_dynamics(M, method="poly", M=1)
tt = [i for i in range(len(tmp2)) if tmp2[i] < 1]
plt.plot(range(tt[-1] + 1), tmp2[tt], color="black")
tmp2 = early_behaviour_dynamics(M, method="poly", M=4)
tt = [i for i in range(len(tmp2)) if tmp2[i] < 1]
plt.plot(range(tt[-1] + 1), tmp2[tt], color="black")

B_avg /= n_count
plt.plot(trajectory["time"], B_avg, color="grey")

B_med = B_med[:, 1:]
B_med = np.median(B_med, axis=1)
plt.plot(trajectory["time"], B_med, color="purple")

# R0 = M.Rzero()
# f = (I0/P) * np.exp(((R0 - 1) /
#                      params["infectious_period"]) * trajectory["time"])
# tt2 = [i for i in range(len(tmp)) if f[i] < 1]
# plt.plot(tt2, f[tt2],
#          color="grey")
plt.xlabel("time")
plt.ylabel("Count")
plt.legend(["Behaviour", "Infections"])
plt.show()

# %%
count = 0
for idx in range(num_trajectory):
    trajectory = results[idx]
    B = (trajectory["Sb"] + trajectory["Ib"] + trajectory["Rb"])
    if (P-B[-1]) > int(0.9*P):
        count += 1
print(count/num_trajectory)

# %%
res = []

tmp = early_behaviour_dynamics(M)
tt = [i for i in range(len(tmp)) if tmp[i] < 1]

for idx in range(num_trajectory):
    trajectory = results[idx]
    B = (trajectory["Sb"] + trajectory["Ib"] + trajectory["Rb"])/P
    res.append(simpson((tmp[tt] - B[tt])**2, x=tt))

print(np.mean(res))
res2 = []

tmp2 = early_behaviour_dynamics(M, method="poly", M=3)
tt2 = [i for i in range(len(tmp2)) if tmp2[i] < 1]


for idx in range(num_trajectory):
    trajectory = results[idx]
    B = (trajectory["Sb"] + trajectory["Ib"] + trajectory["Rb"])/P
    res2.append(simpson((tmp2[tt] - B[tt])**2, x=tt))
print(np.mean(res2))

# %%

with open("test.json", "w") as f:
    json.dump(results.to_json(), f)

# %%

with open("test.json", "r") as f:
    test_load = json.load(f)

test_load_dict = gillespy2.core.jsonify.Jsonify.from_json(test_load)
