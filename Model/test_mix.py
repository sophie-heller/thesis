import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from scipy.special import expit
from functions import step, plot_results_time, plot_results_height, import_variables
import os
import importlib.util

num_layers=6
mdot_in= np.full(6,3)
mdot_in[3]=4
mdot_out= np.full(6,2)
mdot_out[5]=9

# ensure mass conservation of the system
mdot_balance = np.sum(mdot_in) - np.sum(mdot_out)
print(mdot_balance)

# Calculate mdot mix_in for layer i. Positive means from below to above (i-1 -> i). i=0 is 0 (outside of tank) and i=N+1= 0 (outside of tank)
mdot_mix_in = np.zeros(num_layers+1)


"""
for i in range(num_layers-1):          # calculate mdot_mix_next between layers, flowing from layer i to i+1. There are num_layers-1 mixing streams
    mdot_mix_sum = 0 
    for j in range(0, i+1):                 #inlcude current layer i
        mdot_mix = 0
        mdot_mix = mdot_in[j] - mdot_out[j]
        mdot_mix_sum += mdot_mix
        
    mdot_mix_in[i+1] = mdot_mix_sum

print(mdot_mix_in)
"""

# Calculate cumulative sum of the differences
mdot_diff = mdot_in - mdot_out
cumulative_mdot_diff = np.cumsum(mdot_diff)

# Assign cumulative sums to mdot_mix_in (shifted by one position)
mdot_mix_in[1:num_layers] = cumulative_mdot_diff[:-1]

print(f"mdot going from layer below to layer above: mdot_mix_in {mdot_mix_in}")

#T_zero = np.full(6, 20)
mdot_prime_mix = np.zeros(num_layers)

for i in range(num_layers):
    mdot_prime_mix_in_above = 0
    mdot_prime_mix_below = 0
    
    if (-mdot_mix_in[i + 1]) > 0:
        mdot_prime_mix_in_above = (-mdot_mix_in[i + 1]) * (T_zero[i + 1] - T_zero[i])
        
    if (mdot_mix_in[i]) > 0:
        mdot_prime_mix_below = (mdot_mix_in[i]) * (T_zero[i - 1] - T_zero[i])
        
    mdot_prime_mix[i] += mdot_prime_mix_in_above + mdot_prime_mix_below

print("Layer temperature change due to mdot mix (mdot_mix):", mdot_mix)
