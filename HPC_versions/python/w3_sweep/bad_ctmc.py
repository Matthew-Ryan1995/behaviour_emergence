#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 09:54:49 2024

Central place to save the BaD CTMC model

@author: rya200
"""
# %%

import gillespy2
from scipy.optimize import fsolve
import json
import gzip

# %%


def bad_ctmc(param_vals: dict, P: int = 100, I0: int = 1, B0: int = 1, t_end: int = 100, event: dict = {}):
    """


    Parameters
    ----------
    param_vals : dict
        Parameter value to define the model.  Inputs are:
            :param transmission: float, the transmission rate from those infectious to those susceptible.
            :param infectious_period: int, the average infectious period.
            :param immune_period: int (optional), average Immunity period (for SIRS)
            :param susc_B_efficacy: probability (0, 1), effectiveness in preventing disease contraction if S wears mask (c)
            :param inf_B_efficacy: probability (0, 1), effectiveness in preventing disease transmission if I wears mask (p)
            :param N_social: float, social influence of non-behaviour on behaviour (a1)
            :param N_const: float, Spontaneous abandonement of behaviour (a2)
            :param B_social: float, social influence of behaviour on non-behaviour (w1)
            :param B_fear: float, Fear of disease for non-behaviour to behaviour(w2)
            :param B_const: float, Spontaneous uptake of behaviour (w2)
    P : int, optional
        Population size.  Assume closed population. The default is 100.
    I0 : int, optional
        Initial number of infectious people. The default is 1.
    B0 : int, optional
        Initial number of people performing behaviour. The default is 1.
    t_end : int, optional
        End time for simulations. The default is 100.
    event : dict, optional
        Dictionary containing intervention information. The default is {}.  If non-empty, must contain:
            :param strength: float, Numeric multiplier of parameter due to intervention.  Greater than 0.
            :param target: str, Target parameter for intervention.  Must be w1, w2, w3, a1, or a2.
            :param day: int, Day to force intervention on, suggestions are 5, 10, or 15.

    Returns
    -------
    model : gillespy2
        gillespy2 model for stochastically simulating a BaD SIR/S model.

    """

    model = gillespy2.Model()

    # Parameters
    gamma = gillespy2.Parameter(
        name="gamma", expression=1/param_vals["infectious_period"])
    beta_nn = gillespy2.Parameter(
        name="beta_nn", expression=param_vals["transmission"] / (P-1))
    beta_nb = gillespy2.Parameter(name="beta_nb", expression=(
        1 - param_vals["susc_B_efficacy"]) * param_vals["transmission"] / (P-1))
    beta_bn = gillespy2.Parameter(name="beta_bn", expression=(
        1 - param_vals["inf_B_efficacy"]) * param_vals["transmission"] / (P-1))
    beta_bb = gillespy2.Parameter(name="beta_bb", expression=(
        1 - param_vals["inf_B_efficacy"]) * (1 - param_vals["susc_B_efficacy"]) * param_vals["transmission"] / (P-1))
    w1 = gillespy2.Parameter(
        name="w1", expression=param_vals["B_social"] / (P-1))
    w2 = gillespy2.Parameter(
        name="w2", expression=param_vals["B_fear"] / (P-1))
    w3 = gillespy2.Parameter(name="w3", expression=param_vals["B_const"])
    a1 = gillespy2.Parameter(
        name="a1", expression=param_vals["N_social"] / (P-1))
    a2 = gillespy2.Parameter(name="a2", expression=param_vals["N_const"])

    model.add_parameter([gamma, beta_nn, beta_nb, beta_bn, beta_bb,
                         w1, w2, w3, a1, a2])

    # Species
    Sn = gillespy2.Species(name="Sn", initial_value=P -
                           I0 - B0, mode="discrete")
    In = gillespy2.Species(name="In", initial_value=I0, mode="discrete")
    Rn = gillespy2.Species(name="Rn", initial_value=0, mode="discrete")
    Sb = gillespy2.Species(name="Sb", initial_value=B0, mode="discrete")
    Ib = gillespy2.Species(name="Ib", initial_value=0, mode="discrete")
    Rb = gillespy2.Species(name="Rb", initial_value=0, mode="discrete")

    # Track total infections and final size
    I_total = gillespy2.Species(name="I_total",
                                initial_value=I0, mode="discrete")

    model.add_species([Sn, Sb, In, Ib, Rn, Rb, I_total])

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
                                      reactants={Sn: 1, In: 1}, products={In: 2, I_total: 1},
                                      rate=beta_nn)
    in_infect_sb = gillespy2.Reaction(name="in_infect_sb",
                                      reactants={Sb: 1, In: 1}, products={In: 1, Ib: 1, I_total: 1},
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
                                      reactants={Sn: 1, Ib: 1}, products={In: 1, Ib: 1, I_total: 1},
                                      rate=beta_bn)
    ib_infect_sb = gillespy2.Reaction(name="ib_infect_sb",
                                      reactants={Sb: 1, Ib: 1}, products={Ib: 2, I_total: 1},
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

    # Add immunity if not 0
    if param_vals["immune_period"] > 0:
        nu = gillespy2.Parameter(
            name="nu", expression=1/param_vals["immune_period"])
        model.add_parameter([nu])

        rn_to_sn = gillespy2.Reaction(name="rn_to_sn",
                                      reactants={Rn: 1},
                                      products={Sn: 1},
                                      rate=nu)
        rb_to_sb = gillespy2.Reaction(name="rb_to_sb",
                                      reactants={Rb: 1},
                                      products={Sb: 1},
                                      rate=nu)

        model.add_reaction([rn_to_sn, rb_to_sb])

    # Add event if not 0
    if len(event) > 0:
        intervention_strength = event["strength"]
        intervention_target = event["target"]
        intervention_day = event["day"]

        intervention_trigger = gillespy2.EventTrigger(
            expression=f"t>={intervention_day}")
        intervention_reaction = gillespy2.EventAssignment(variable=model.listOfParameters[intervention_target],
                                                          expression=f"{intervention_strength} * {intervention_target}")
        intervention = gillespy2.Event(name="intervention", trigger=intervention_trigger,
                                       assignments=[intervention_reaction])
        model.add_event(intervention)

    tspan = gillespy2.TimeSpan.linspace(t=t_end, num_points=t_end + 1)
    model.timespan(tspan)
    return model


# def get_w1(Bstar, params: dict):
#     """
#     This calculation assumes that a1 =/= w1.
#     Parameters
#     ----------
#     Bstar : Float
#         Target behaviour level in infection-free equilibrium.
#     params : dict
#         Parameter value to define the model.  Inputs are:
#             :param transmission: float, the transmission rate from those infectious to those susceptible.
#             :param infectious_period: int, the average infectious period.
#             :param immune_period: int (optional), average Immunity period (for SIRS)
#             :param susc_B_efficacy: probability (0, 1), effectiveness in preventing disease contraction if S wears mask (c)
#             :param inf_B_efficacy: probability (0, 1), effectiveness in preventing disease transmission if I wears mask (p)
#             :param N_social: float, social influence of non-behaviour on behaviour (a1)
#             :param N_const: float, Spontaneous abandonement of behaviour (a2)
#             :param B_social: float, social influence of behaviour on non-behaviour (w1)
#             :param B_fear: float, Fear of disease for non-behaviour to behaviour(w2)
#             :param B_const: float, Spontaneous uptake of behaviour (w2)

#     Returns
#     -------
#     Float
#         The social influence (w1) needed to ensure beahviour level Bstar given a1, a2, w2, and w3.

#     """
#     a1 = params["N_social"]
#     w3 = params["B_const"]

#     k = a1 + w3 + params["N_const"]

#     def solve_w1(x):
#         ans = ((1-2*Bstar)**2 - 1) * x**2 + (2*(2*Bstar*a1 - k)*(1-2*Bstar) -
#                                              (4*w3 - 2*k)) * x + ((2*Bstar*a1 - k)**2 - (k**2 - 4*a1*w3))
#         return ans

#     w1 = fsolve(solve_w1, x0=[0, 1, 2])
#     if w1.min() > 0:
#         ans = w1.min()
#     else:
#         ans = w1.max()
#     return ans

def get_w1(Bstar, params: dict):
    """
    This calculation assumes that a1 =/= w1.
    Parameters
    ----------
    Bstar : Float
        Target behaviour level in infection-free equilibrium.
    params : dict
        Parameter value to define the model.  Inputs are:
            :param transmission: float, the transmission rate from those infectious to those susceptible.
            :param infectious_period: int, the average infectious period.
            :param immune_period: int (optional), average Immunity period (for SIRS)
            :param susc_B_efficacy: probability (0, 1), effectiveness in preventing disease contraction if S wears mask (c)
            :param inf_B_efficacy: probability (0, 1), effectiveness in preventing disease transmission if I wears mask (p)
            :param N_social: float, social influence of non-behaviour on behaviour (a1)
            :param N_const: float, Spontaneous abandonement of behaviour (a2)
            :param B_social: float, social influence of behaviour on non-behaviour (w1)
            :param B_fear: float, Fear of disease for non-behaviour to behaviour(w2)
            :param B_const: float, Spontaneous uptake of behaviour (w2)

    Returns
    -------
    Float
        The social influence (w1) needed to ensure beahviour level Bstar given a1, a2, w2, and w3.

    """
    a1 = params["N_social"]
    w3 = params["B_const"]

    k = a1 + w3 + params["N_const"]

    A = (1-2*Bstar)**2 - 1
    B = -(1-4*Bstar**2) * a1 - 4*(Bstar * k - w3) + a1

    ans = B/A
    if ans < 0:
        ans = 0

    return ans


def get_w3(Bstar_min, params: dict):
    """
    Returns a value of w3 that will give a minimum behaviour prevalence of Bstar_min assuming
    that w1 = 0.  This gives a lower bound when varying w1.
    Parameters
    ----------
    Bstar_min : Float
        Minimum behaviour prevalence considered in the model.
    params : dict
        Parameter value to define the model.  Inputs are:
            :param transmission: float, the transmission rate from those infectious to those susceptible.
            :param infectious_period: int, the average infectious period.
            :param immune_period: int (optional), average Immunity period (for SIRS)
            :param susc_B_efficacy: probability (0, 1), effectiveness in preventing disease contraction if S wears mask (c)
            :param inf_B_efficacy: probability (0, 1), effectiveness in preventing disease transmission if I wears mask (p)
            :param N_social: float, social influence of non-behaviour on behaviour (a1)
            :param N_const: float, Spontaneous abandonement of behaviour (a2)
            :param B_social: float, social influence of behaviour on non-behaviour (w1)
            :param B_fear: float, Fear of disease for non-behaviour to behaviour(w2)
            :param B_const: float, Spontaneous uptake of behaviour (w2)

    Returns
    -------
    Float
        The social influence (w1) needed to ensure beahviour level Bstar given a1, a2, w2, and w3.

    """
    a1 = params["N_social"]
    a2 = params["N_const"]

    A = Bstar_min*(a1+a2) - a1*Bstar_min**2
    B = 1-Bstar_min

    ans = A/B
    if ans < 0:
        ans = 0

    return ans


def get_outbreak(dlr: dict, P: int = 100, outbreak_definition=0.001):
    """
    Parameters
    ----------
    dlr : dict
        Output from the run of a gillespy2 model for the Bad SIR/S CTMC.
    P : int, optional
        Population size. The default is 100.
    outbreak_definition : Float, optional
        Percentage of population needed to be infected to classify major outbreak.  Heuristically chosen. The default is 0.001.

    Returns
    -------
    ans : Binary 0/1
        1 if outbreak occured, 0 else.

    """
    I = dlr["I_total"]

    ans = 0
    if I[-1] > int(20):
        ans = 1

    return ans


def compress_data(data):
    """
    A convenience function taken from:
    https://gist.github.com/LouisAmon/4bd79b8ab80d3851601f3f9016300ac4

    Parameters
    ----------
    data : sr/json

    Returns
    -------
    compressed : Compressed version of json

    """
    # Convert to JSON
    # json_data = json.dumps(data, indent=2)
    # Convert to bytes
    encoded = data.encode('utf-8')
    # Compress
    compressed = gzip.compress(encoded)
    return compressed


# %%


if __name__ == "__main__":
    from datetime import datetime
    from BaD import *
    import matplotlib.pyplot as plt
    import numpy as np

    R0 = 3

    # Load and choose parameters
    params = load_param_defaults()

    params["infectious_period"] = 5
    params["transmission"] = R0/params["infectious_period"]
    params["immune_period"] = 0  # No waning immunity

    params["B_fear"] = 8/params["infectious_period"]
    params["N_social"] = 0.4 / params["infectious_period"]
    params["B_const"] = 0.01 / params["infectious_period"]
    params["N_const"] = 0.6 / params["infectious_period"]

    params["B_social"] = (
        0.9 * (params["N_social"] + params["N_const"]))

    # Define an intervention event
    # event = {
    #     "strength": 3.5,
    #     "target": "w1",
    #     "day": 10
    # }
    event = {}

    # Set up initial conditions
    P = 10000
    I0 = 1
    B0 = 1
    num_trajectory = 100

    Sn = P - I0 - B0

    IC = np.array([Sn, B0, I0, 0, 0, 0]) / P
    t_start, t_end = [0, 100]

    # Define and calcualte the ODE approximation
    M = bad(**params)
    M.run(IC=IC, t_start=t_start, t_end=t_end)

    # Run the CTMC
    start_time = datetime.now()

    model = bad_ctmc(param_vals=params, P=P, I0=I0,
                     B0=B0, t_end=t_end, event=event)
    results = model.run(number_of_trajectories=num_trajectory)

    print(f"Time taken: {datetime.now()-start_time}")

    # Plot results
    plt.figure()
    for idx in range(num_trajectory):
        trajectory = results[idx]
        B = (trajectory["Sb"] + trajectory["Ib"] + trajectory["Rb"]) / P
        I = (trajectory["Ib"] + trajectory["In"]) / P
        plt.plot(trajectory["time"], B, color="blue", alpha=0.2)
        plt.plot(trajectory["time"], I, color="red", alpha=0.2)

    # Plot/demonstrate early time approximations
    exp_approx = early_behaviour_dynamics(M)
    tt = [i for i in range(len(exp_approx)) if exp_approx[i] < 1]
    plt.plot(range(tt[-1] + 1), exp_approx[tt],
             linestyle="dashed", color="black")

    cubic_approx = early_behaviour_dynamics(M, method="poly", M=3)
    tt = [i for i in range(len(cubic_approx)) if cubic_approx[i] < 1]
    plt.plot(range(tt[-1] + 1), cubic_approx[tt],
             linestyle="dotted", color="black")

    plt.xlabel("time")
    plt.ylabel("Proportion")
    plt.legend(["Behaviour", "Infections"])
    plt.title("Dashed line - exponential\nDotted line - Cubic")
    plt.show()

    # Demonstrate calculating outbreak probabilities
    outbreak_prob = list(map(lambda x: get_outbreak(dlr=x, P=P), results))
    outbreak_prob = np.array(outbreak_prob)

    outbreak_prob = outbreak_prob.sum()/num_trajectory

    print(f"Probability of a major outbreak is {outbreak_prob}.")

    # Demonstrate calculating specific parameter values for given target Bstar
    Bstar = 0.02
    params["B_social"] = get_w1(Bstar, params)

    M2 = bad(**params)
    M2.run(IC=IC, t_start=t_start, t_end=t_end)

    M2.endemic_behaviour(I_eval=0)
    print(f"Endemic proportion doing behaviour {1-M2.Nstar}")

    # Plot final size distribution
    plt.figure()
    fs = []
    for idx in range(num_trajectory):
        trajectory = results[idx]
        fs.append(trajectory["I_total"][-1])
    fs = np.array(fs)
    plt.hist(fs, bins=num_trajectory)
    plt.xlabel("Final size")
    plt.ylabel("Frequency")
    plt.show()
