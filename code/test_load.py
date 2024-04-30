#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 15:00:03 2024

@author: rya200
"""
import json
import gillespy2
import pickle

with open("../data/simulations/interventions/target_w1_day_5_strength_0.0_seed_20240430_tend_200_trajectories_10.json", "r") as f:
    test_load = json.load(f)

test_load_dict = gillespy2.core.jsonify.Jsonify.from_json(test_load)

with open("../data/simulations/model/baseline_simulations_seed_20240430_tend_200_trajectories_10.json", "rb") as f:
    tmp = pickle.load(f)
f.close()
