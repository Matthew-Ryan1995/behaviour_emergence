#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 11:55:29 2024


Run stochastic simulations for behaviour transitions, no disease

@author: rya200
"""

# %% libraries

from enum import IntEnum
import numpy as np
import json
import matplotlib.pyplot as plt

np.random.seed(20230927)  # for posterity and reproducibility

# %% Classes


class Compartments(IntEnum):
    """
    for speed ups whilst maintaining readability of code
    """
    N = 0
    B = 1


class nbn(object):
    def __init__(self, **kwargs):
        """
        Written by: Rosyln Hickson
        Required parameters when initialising this class, plus deaths and births optional.
        :param transmission: double, the transmission rate from those infectious to those susceptible.
        :param infectious_period: scalar, the average infectious period.
        :param immune_period: scalar, average Ibmunity period (for SIRS)
        :param susc_B_efficacy: probability (0, 1), effectiveness in preventing disease contraction if S wears mask (c)
        :param inf_B_efficacy: probability (0, 1), effectiveness in preventing disease transmission if I wears mask (p)
        :param N_social: double, social influence of non-mask wearers on mask wearers (a1)
        :param N_fear: double, Fear of disease for mask wearers to remove mask (a2)
        :param B_social: double, social influence of mask wearers on non-mask wearers (w1)
        :param B_fear: double, Fear of disease for non-mask wearers to put on mask (w2)
        :param av_lifespan: scalar, average life span in years
        """
        args = self.set_defaults()  # load default values from json file
        # python will overwrite existing values in the `default` dict with user specified values from kwargs
        args.update(kwargs)

        args["S"] = np.array([1e4 - 1, 1])
        args["t"] = 0

        for key, value in args.items():  # this is because I like the . notation. e.g. self.transmission
            self.__setattr__(key, value)
        self.N_fear = 0

        self.event_list = ["social_n_to_b",
                           "spontaneous_n_to_b",
                           "social_b_to_n",
                           "spontaneous_b_to_n",
                           ]
        self.events = [self.event_social_n_to_b,
                       self.event_spontaneous_n_to_b,
                       self.event_social_b_to_n,
                       self.event_spontaneous_b_to_n,
                       ]

    def set_defaults(self, filename="/Users/rya200/Library/CloudStorage/OneDrive-CSIRO/Documents/03_projects/reid-mask_sir_toymodel/code/model_parameters.json"):
        """
        Written by: Rosyln Hickson
        Sull out default values from a file in json format.
        :param filename: json file containing default parameter values, which can be overridden by user specified values
        :return: loaded expected parameter values
        """
        with open(filename) as json_file:
            json_data = json.load(json_file)
        for key, value in json_data.items():
            json_data[key] = value["exp"]
        return json_data

    def update_params(self, **kwargs):
        args = kwargs
        for key, value in args.items():  # this is because I like the . notation. e.g. self.transmission
            self.__setattr__(key, value)
        self.N_fear = 0

    # Define events

    def event_social_n_to_b(self):
        if self.S[Compartments.N] > 0:
            self.S[Compartments.N] -= 1
            self.S[Compartments.B] += 1

    def event_spontaneous_n_to_b(self):
        if self.S[Compartments.N] > 0:
            self.S[Compartments.N] -= 1
            self.S[Compartments.B] += 1

    def event_social_b_to_n(self):
        if self.S[Compartments.B] > 0:
            self.S[Compartments.N] += 1
            self.S[Compartments.B] -= 1

    def event_spontaneous_b_to_n(self):
        if self.S[Compartments.B] > 0:
            self.S[Compartments.N] += 1
            self.S[Compartments.B] -= 1

    def rates(self):
        P = self.S.sum()

        B = self.S[Compartments.B].sum()
        N = self.S[Compartments.N].sum()

        rate_social_n_to_b = self.B_social * B * N / \
            (P-1)  # Don't interact with self
        rate_social_b_to_n = self.N_social * N * B / \
            (P-1)  # Don't interact with self

        rate_spontaneous_n_to_b = self.B_const * N
        rate_spontaneous_b_to_n = self.N_const * B

        return np.array([rate_social_n_to_b,
                         rate_spontaneous_n_to_b,
                         rate_social_b_to_n,
                         rate_spontaneous_b_to_n,
                         ])

    def perform_event(self):

        rates = self.rates()

        rate_total = rates.sum()

        dt = np.random.exponential(scale=1/rate_total, size=1)[0]

        idx = np.random.choice(range(rates.size), p=rates/rate_total)

        # R1 = np.random.rand()
        # R2 = np.random.rand()

        # dt = -np.log(R1)/(rate_total)

        # p = R2*rate_total

        # cum_rates = np.cumsum(rates)

        # p_event = [y for y in range(cum_rates.size) if p <= cum_rates[y]][0]

        self.events[idx]()

        self.t += dt

        return self.event_list[idx]


def runModel(mod, endTime):

    storage = np.zeros([2, endTime+1])

    storage[:, 0] = mod.S

    store_idx = 1

    while mod.t < endTime:
        mod.perform_event()

        if np.floor(mod.t) > store_idx:
            storage[:, store_idx] = mod.S
            store_idx += 1

    return storage


if __name__ == "__main__":

    params = {"B_social": 4}
    M1 = nbn(**params)

    endTime = 200

    tmp = runModel(M1, endTime)

    plt.figure()
    plt.plot(tmp[1, :])
    plt.show()
