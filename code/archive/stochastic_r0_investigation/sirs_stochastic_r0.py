#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 09:30:20 2023

@author: rya200
"""

# %% libraries

import numpy as np
import matplotlib.pyplot as plt

# %% Inital parameters

ND = 360.0

# %% Define events


class sir(object):

    def __init__(self, **kwargs):
        args = {"beta": 2, "gamma": 1, "nu": 1 /
                240, "P": np.array([1e4, 1, 0]), "t": 0}

        args.update(kwargs)

        for key, value in args.items():  # this is because I like the . notation. e.g. self.transmission
            self.__setattr__(key, value)

    def event_infect(self):
        self.P[0] -= 1
        self.P[1] += 1

    def event_recover(self):
        self.P[1] -= 1
        self.P[2] += 1

    def event_waning_immunity(self):
        self.P[2] -= 1
        self.P[0] += 1

    def rates(self):
        N = self.P.sum()

        rate_infect = self.beta * self.P[0] * self.P[1]/N
        rate_recover = self.gamma * self.P[1]
        rate_immune = self.nu * self.P[2]

        return np.array([rate_infect, rate_recover, rate_immune])

    def perform_event(self):

        rates = self.rates()

        rate_total = rates.sum()

        R1 = np.random.rand()
        R2 = np.random.rand()

        dt = -np.log(R1)/(rate_total)

        p = R2*rate_total

        cum_rates = np.cumsum(rates)

        p_event = [y for y in range(cum_rates.size) if p <= cum_rates[y]][0]

        if p_event == 0:
            self.event_infect()
            event = "infect"
        elif p_event == 1:
            self.event_recover()
            event = "recover"
        else:
            self.event_waning_immunity()
            event = "immune"

        self.t += dt

        return event


def run_iterations(model):
    T = [0]
    count = 0

    res = model.P.reshape(1, 3)

    run_mod = model

    while (T[count] < ND) and (res[-1, 1] > 0):
        count += 1
        event = run_mod.perform_event()

        T.append(run_mod.t)

        res = run_mod.P.reshape(1, 3)

        if event == "recover":
            break

    return res[-1, 1], T


# %%


if __name__ == "__main__":

    R0_vals = []

    iteration_runs = 10000

    for x in range(iteration_runs):

        PP = np.array([1e4, 1, 0])

        R0 = 5
        gamma = 0.2

        args = {"beta": R0*gamma, "gamma": gamma, "P": PP, "t": 0}

        mod = sir(**args)

        res, t = run_iterations(mod)

        R0_vals.append(res)

    R0_vals = np.array(R0_vals)

    print(f"estimated R0 is {R0_vals.mean()}")