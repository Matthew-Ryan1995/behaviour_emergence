#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 08:13:19 2024

@author: rya200
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 11:40:52 2024

@author: rya200
"""


import matplotlib.pyplot as plt
import gillespy2
import numpy as np
from scipy.integrate import solve_ivp
from datetime import datetime
start_time = datetime.now()

# %%


def SIR(parameter_values=None, t_end=20, N=100, I_0=1):

    model = gillespy2.Model(name="SIRS")

    if parameter_values is not None:
        params = parameter_values
    else:
        params = {"transmission": 2, "recovery_rate": 1, "immune_rate": 1/40}

    # Params
    beta_0 = gillespy2.Parameter(
        name="beta_0", expression=params["transmission"])
    beta = gillespy2.Parameter(
        name="beta", expression="beta_0/(gamma)")
    gamma = gillespy2.Parameter(
        name="gamma", expression=params["recovery_rate"])
    nu = gillespy2.Parameter(
        name="nu", expression=params["immune_rate"])

    model.add_parameter([beta_0,  gamma, nu])

    # States
    S = gillespy2.Species(name="Susceptible", initial_value=N-I_0)
    I = gillespy2.Species(name="Infectious", initial_value=I_0)
    R = gillespy2.Species(name="Recovered", initial_value=0)

    model.add_species([S, I, R])

    model.add_parameter([beta])

    # Reactions
    s_to_i = gillespy2.Reaction(name="infect",
                                reactants={S: 1, I: 1},
                                products={I: 2},
                                rate="beta"
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

R = 0

S = N-I-R
IC = [S, I, R]


params = {"transmission": beta, "recovery_rate": gamma, "immune_rate": nu}

model = SIR(parameter_values=params, t_end=t_end, N=N, I_0=I)
results = model.run(number_of_trajectories=num_trajectory)

# %%


def sir_ode(t, PP):
    Y = np.zeros(3)

    Y[0] = -beta * PP[0] * PP[1] / N + nu * PP[2]
    Y[1] = beta * PP[0] * PP[1] / N - gamma * PP[1]
    Y[2] = gamma * PP[1] - nu * PP[2]

    return Y


t_start = 0

t_span = np.arange(t_start, t_end + 1, step=1, )

res = solve_ivp(fun=sir_ode, t_span=[t_start, t_end], y0=IC, t_eval=t_span)

dat = res.y.T

# %%
for index in range(0, num_trajectory):
    trajectory = results[index]
    plt.plot(trajectory['time'], trajectory['Susceptible'], 'g', alpha=0.1)
    plt.plot(trajectory['time'], trajectory['Infectious'],   'r', alpha=0.1)
    plt.plot(trajectory['time'], trajectory['Recovered'],   'b', alpha=0.1)
plt.plot(t_span, dat[:, 0], color="g", linewidth=2,  label="S")
plt.plot(t_span, dat[:, 1], color="r", linewidth=2, label="I")
plt.plot(t_span, dat[:, 2], color="b", linewidth=2, label="R")
plt.legend()
plt.show()

plt.figure()
for index in range(0, num_trajectory):
    trajectory = results[index]
    plt.plot(trajectory['Susceptible'],
             trajectory['Infectious'], 'b', alpha=0.1)
plt.plot(dat[:, 0], dat[:, 1], color="g", linewidth=2)
plt.show()


results.plot()

# %%

count = 0
for index in range(0, num_trajectory):
    trajectory = results[index]
    if trajectory["Susceptible"][-1] > (N-10):
        count += 1

print(count/num_trajectory)

print(f"Time taken: {datetime.now()-start_time}")

# plt.figure()
# plt.plot(t_span, dat[:, 0], color="g", linestyle=":", label="S")
# plt.plot(t_span, dat[:, 1], color="r", linestyle=":", label="I")
# plt.plot(t_span, dat[:, 2], color="b", linestyle=":", label="R")
# plt.legend()
# plt.show()
