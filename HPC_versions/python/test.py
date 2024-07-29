#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 13:42:50 2024

@author: rya200
"""
import os

# %%

mc_cores = os.environ["SLURM_NTASKS"]

print("The number of cores is ", int(mc_cores))

array_val = os.environ["SLURM_ARRAY_TASK_ID"]

print("The array numebr is ", int(array_val))
