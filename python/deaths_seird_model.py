# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 21:28:11 2020

@author: u6026797
"""

#%% Libraries
import pandas as pd
import numpy as np
from numpy import exp
import matplotlib.pyplot as plt
import math 

from statsmodels.tsa.stattools import pacf
from scipy.integrate import odeint
from scipy.optimize import minimize

#%% Initial settings
pop_size = 3206000
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

#%% Read in comparison Vector
loc_name = 'UT'

comp_vec = pd.read_csv('/home/ec2-user/covid_data/COVID19_USTracking.csv')
comp_vec = comp_vec[comp_vec['state']==loc_name]
comp_vec = comp_vec['New_death'].to_numpy()
comp_vec = comp_vec[::-1]
#plt.plot(comp_vec)

comp_vec_i = pd.read_csv('/home/ec2-user/covid_data/COVID19_USTracking.csv')
comp_vec_i = comp_vec_i[comp_vec_i['state']==loc_name]
comp_vec_i = comp_vec_i['New_confirm'].to_numpy()
comp_vec_i = comp_vec_i[::-1]

date_vec = pd.read_csv('/home/ec2-user/covid_data/COVID19_USTracking.csv')
date_vec = date_vec[date_vec['state']==loc_name]
date_vec = date_vec['Date']

date_vec = date_vec.astype('datetime64')
date_vec = date_vec[::-1]

#%% Functions
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

def R0_mitigating(t, r0=3, intervention=1, r_bar=1.6, fpc=8, spc=28, pcc=.05):
    R0 = (r0*exp(-intervention*t)+(1-exp(-intervention*t))*r_bar) * \
        (1+(np.sin((2*math.pi/fpc)*t))*pcc) * \
        (1+(np.sin((2*math.pi/spc)*t))*pcc)
    return R0

def L1(init_params,comp_vec=comp_vec):
    R0 = lambda t: R0_mitigating(t, intervention=0.008, pcc = .1)
    i_path, c_path, d_path = solve_path(R0, t_vec, init_params=init_params)
    #force length
    d_path = d_path[:len(comp_vec)]
    out = [path * pop_size for path in d_path]
    resid_vec = comp_vec - out 
    return sum(resid_vec**2)

#%% Run
#run to find slow and periodic components for each sequence as desired
fast_pc= (np.where((pacf(comp_vec))==(max(pacf(comp_vec)[6:]))))[0][0]+1
#slow component (above 14, not a multiple of the above)
slow_pc = 41
for i in range(15,40,1):
    slow_pc_check = (np.where((pacf(comp_vec))==(max(pacf(comp_vec)[(i):]))))[0][0]+1
    if ((slow_pc_check % fast_pc) != 0) & (slow_pc_check < slow_pc):
        slow_pc=slow_pc_check

#best estimates gathered from case run
init_params_guess = [0.09024598, 0.15103293, 0.05052019]
bnds = ((0,None),(0,None),(0,None))
fun_a = lambda x: L1(init_params=x)

best_est_params = minimize(fun_a,x0=init_params_guess,
                           options={'verbose': 1,'maxiter': 10000},bounds=bnds)

best_est_params['x']

#%% plot to visualize fit
R0 = lambda t: R0_mitigating(t, intervention=0.008, pcc = .1)
i_path, c_path, d_path = solve_path(R0, t_vec, init_params=best_est_params['x'])
#plot deaths data
d_path = d_path[:len(comp_vec)]
out = [path * pop_size for path in d_path]

#plot infected data
i_path = i_path[:len(comp_vec)]
out_i = [path * pop_size for path in i_path]


sns.set(font_scale=1.5)
f, (ax1, ax2) = plt.subplots(2,1,figsize=(16,8))
ax1.plot(date_vec,comp_vec,label='Observed Deaths')
ax1.plot(date_vec,out,label='SEIDR Estimated Deaths')
ax1.legend(loc='upper left')
ax1.set_title('SEIDR Model Fit for Utah')

ax2.plot(date_vec,comp_vec_i,label='Observed Infected')
ax2.plot(date_vec,out_i,label='SEIDR Estimated Infected')
ax2.legend(loc='upper left')
plt.show()


#%% loop for locations in set
comp_vec = pd.read_csv('/home/ec2-user/covid_data/COVID19_USTracking.csv')

#trim infinite values, report mean
temp_result_vec = abs(comp_vec_i - out_i)/comp_vec_i
np.mean(temp_result_vec[temp_result_vec!=np.inf])
