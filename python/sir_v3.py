# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 14:29:03 2020

@author: u6026797
"""

#%% libraries
import numpy as np
from numpy import exp
import matplotlib.pyplot as plt
import math 
import random

from statsmodels.tsa.stattools import pacf
from scipy.integrate import odeint

#%% params
pop_size = 10000
recovery_rate = 1 / 18
infection_rate = 1 / 5.2
mortality_rate = 0.05

i_0 = 1e-7
e_0 = 4 * i_0
s_0 = 1 - i_0 - e_0
d_0 = 0
r_0 = 0

x_0 = s_0, e_0, i_0, d_0, r_0

t_length = 550
grid_size = 1000
t_vec = np.linspace(0, t_length, grid_size)

#%% fun
def F(x, t, init_params, R0=1.6):
    """
    Time derivative of the state vector.

        * x is the state vector (array_like)
        * t is time (scalar)
        * R0 is the effective transmission rate, defaulting to a constant
    """
    s, e, i, d, r = x
    
    infection_rate = init_params[0]
    recovery_rate = init_params[1]
    mortality_rate = init_params[2]

    # New exposure of susceptibles
    transmission_rate = R0(t) * recovery_rate if callable(R0) else R0 * recovery_rate
    new_exposed = transmission_rate * s * i

    # Time derivatives
    ds = - new_exposed
    de = new_exposed - infection_rate * e
    di = infection_rate * e - recovery_rate * i
    dd = mortality_rate * i - d
    dr = (1 - mortality_rate) * i - r
    
    return ds, de, di, dd, dr

def solve_path(R0, t_vec, init_params = [1/5.2,1/18,1/20], x_init=x_0):
    """
    Solve for i(t) and c(t) via numerical integration,
    given the time path for R0.
    """
    G = lambda x, t: F(x, t, init_params, R0)
    s_path, e_path, i_path, d_path, r_path = odeint(G, x_init, t_vec).transpose()

    c_path = 1 - s_path - e_path - r_path      # cumulative cases
    return i_path, c_path, d_path

def plot_paths(paths, labels, times=t_vec):

    fig, ax = plt.subplots()

    for path, label in zip(paths, labels):
        ax.plot(times, path, label=label)

    ax.legend(loc='upper left')

    plt.show()
#%% simple run
R0_vals = np.linspace(1.6, 3.0, 6)
labels = [f'$R0 = {r:.2f}$' for r in R0_vals]
i_paths, c_paths, d_paths = [], [], []

for r in R0_vals:
    i_path, c_path, d_path = solve_path(r, t_vec)
    i_paths.append(i_path)
    c_paths.append(c_path)
    d_paths.append(d_path)
plot_paths(d_paths, labels)

#%% varying, periodic r0

def R0_mitigating(t, r0=3, η=1, r_bar=1.6, fpc=7, spc=30, pcc=.05):
    R0 = (r0*exp(-η*t)+(1-exp(-η*t))*r_bar) * \
        (1+(np.sin((2*math.pi/fpc)*t))*pcc) * \
        (1+(np.sin((2*math.pi/spc)*t))*pcc)
    return R0

η_vals = 0.008, 1/5, 1/10, 1/20, 1/50, 1/100
labels = [fr'$\eta = {η:.2f}$' for η in η_vals]

fig, ax = plt.subplots()

for η, label in zip(η_vals, labels):
    ax.plot(t_vec, R0_mitigating(t_vec, η=η), label=label)

ax.legend()
plt.show()


η=0.008
temp_x=R0_mitigating(t_vec, η=η)
plt.plot(temp_x)
#%% run and plot with varying n

i_paths, c_paths, d_paths = [], [], []

for η in η_vals:
    R0 = lambda t: R0_mitigating(t, η=η, pcc = .1)
    i_path, c_path, d_path = solve_path(R0, t_vec)
    i_paths.append(i_path)
    c_paths.append(c_path)
    d_paths.append(d_path)

plot_paths(i_paths, labels)

#plot projected deaths
paths = [path * pop_size for path in d_paths]
plot_paths(paths, labels)


#%%

#%% test with pacf
plt.plot(utah_new)
plt.plot(pacf(utah_new))

#fast periodic component
fast_pc= (np.where((pacf(utah_new))==(max(pacf(utah_new)[6:]))))[0][0]+1
#slow component (above 14, not a multiple of the above)
slow_pc = 41
for i in range(15,40,1):
    slow_pc_check = (np.where((pacf(utah_new))==(max(pacf(utah_new)[(i):]))))[0][0]+1
    if ((slow_pc_check % fast_pc) != 0) & (slow_pc_check < slow_pc):
        slow_pc=slow_pc_check

#%%
#pop_size = 3206000
#recovery_rate = 1 / 18
#infection_rate = 1 / 5.2
#mortality_rate = 0.05

init_params_guess = [.05,.15,.05]

from scipy.optimize import minimize
def L1(init_params,comp_vec=utah_new):
    R0 = lambda t: R0_mitigating(t, η=0.008, pcc = .1)
    i_path, c_path, d_path = solve_path(R0, t_vec, init_params=init_params)
    #force length
    i_path = i_path[:len(comp_vec)]
    out = [path * pop_size for path in i_path]
    resid_vec = comp_vec - out 
    return sum(resid_vec**2)

plt.plot(out)
plt.plot(utah_new)
plt.show()

L1(init_params=init_params_guess)
init_params = [0.09024598, 0.15103293, 0.05052019]

bnds = ((0,None),(0,None),(0,None))
fun_a = lambda x: L1(init_params=x)
init_params_guess = [10,5,1]
minimize(fun_a,x0=init_params_guess, options={'verbose': 1,
                                              'maxiter': 10000},bounds=bnds)
