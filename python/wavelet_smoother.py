# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 15:18:44 2020

@author: u6026797
"""

#%% libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy
from statsmodels.tsa.stattools import adfuller
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy import signal
import seaborn as sns
import pywt
#%% functions
def mean_squared_error(y_hat,y):
    return np.average((y_hat-y)*(y_hat-y))
def cal_h(x_1,y_1,x_0,x_2,y_2):
    m = (y_1 - y_2)/(x_1 - x_2)
    c = ((x_1/x_2)*y_2 - y_1)/((x_1/x_2)-1)
    return m*x_0 + c
def weight_cal(input_data,peak_find_data):
    # Find peaks(max).
    peak_indexes = signal.argrelextrema(peak_find_data, np.greater)
    peak_indexes = peak_indexes[0]
 
    # Find valleys(min).
    valley_indexes = signal.argrelextrema(peak_find_data, np.less)
    valley_indexes = valley_indexes[0]

    turn_points = np.sort(np.concatenate((peak_indexes,valley_indexes),axis=0))
    
    weight_arr = np.zeros(len(input_data))
    if(turn_points.size == 0):
        weight_arr[weight_arr == 0] = 1
        return  weight_arr
    
    first_element = turn_points[0]-1
    if(first_element < 0):
        first_element = 0
    last_element = turn_points[-1]+1
    if(last_element >= len(input_data)):
        last_element = len(input_data) - 1

    turn_points = np.insert(turn_points,0,first_element)
    turn_points = np.append(turn_points,last_element)
    
    peak_arr = []
    for i in range(1,len(turn_points)-1):
        h = cal_h(turn_points[i-1],input_data[turn_points[i-1]],turn_points[i],turn_points[i+1],input_data[turn_points[i+1]])
        peak_arr.append(input_data[turn_points[i]]-h)
    peak_arr = np.array(peak_arr)

    peak_arr = np.abs((peak_arr-np.mean(peak_arr))/np.std(peak_arr))
    peak_w = 1./(1+peak_arr)
    
    
    for i in range(0,len(weight_arr)):
        is_in = np.where(turn_points[1:-1] == i)
        if(np.size(is_in) > 0):
            weight_arr[i] = peak_w[is_in[0]]
    weight_arr[weight_arr == 0] = 1
    
    return weight_arr*weight_arr
def moving_average(a,n=3):
    m = int(n/2)
    MA = []
    for i in range(0,len(a)):
        pl = i-m
        ph=i+m
        if(pl < 0):
            pl = 0
        if(ph > len(a)-1):
            ph = len(a)-1
        MA.append(np.average(a[pl:ph+1]))
    return np.array(MA)
def weighted_ma(a,weight,n=3):
    m = int(n/2)
    MA = []
    for i in range(0,len(a)):
        pl = i-m
        ph=i+m
        if(pl < 0):
            pl = 0
        if(ph > len(a)-1):
            ph = len(a)-1
        MA.append(np.sum(a[pl:ph+1]*weight[pl:ph+1])/np.sum(weight[pl:ph+1]))
    return np.array(MA)
def cal_moving_std(input_data,window):
    count = 0
    tot_std = 0
    for i in range(0,len(input_data)-window):
        count = count + 1
        tot_std  = tot_std + np.std(input_data[i:i+window])
    return tot_std/count
def cal_moving_var(input_data,window):
    count = 0
    tot_std = 0
    for i in range(0,len(input_data)-window):
        count = count + 1
        tot_std  = tot_std + np.var(input_data[i:i+window])
    return tot_std/count
def segment_count_bias(input_noisy, input_smoothed, threshold, window):
    over_t_arr = []
    over_thersh = 0
    
    res = (input_smoothed - input_noisy)/input_smoothed
    
    for i in range(0,len(res)-window):
        if(np.abs(np.mean(res[i:i+window])) > threshold):
            over_thersh = over_thersh + 1
    over_thersh = over_thersh/(len(res)-window)
    return over_thersh
def perform_smooth(input_data,num_of_iterations, window_size):
    smoothed_arr = []
    var_arr = []
    
    weight_arr_itr = weight_cal(input_data,input_data)
    smooth_itr = weighted_ma(input_data,weight_arr_itr,window_size)
    weight_arr_c = weight_arr_itr
    
    smoothed_arr.append(smooth_itr)
    var_arr.append(cal_moving_std(smooth_itr,7))
    
    for i in range(0,num_of_iterations):
        weight_arr_itr = weight_cal(input_data,smooth_itr)
        weight_arr_c = weight_arr_c*weight_arr_itr
        smooth_itr = weighted_ma(input_data,weight_arr_c,window_size)
        var_arr.append(cal_moving_std(smooth_itr,7))
        smoothed_arr.append(smooth_itr)
        
    return smoothed_arr, np.array(var_arr), weight_arr_c
def weighted_ma_theory(a,weight,theory,n=3):
    m = int(n/2)
    MA = []
    for i in range(0,len(a)):
        pl = i-m
        ph=i+m
        if(pl < 0):
            pl = 0
        if(ph > len(a)-1):
            ph = len(a)-1
        MA.append(np.sum(a[pl:ph+1]*weight[pl:ph+1]*theory[pl:ph+1])/np.sum(weight[pl:ph+1]*theory[pl:ph+1]))
    return np.array(MA)
def perform_wvlt_smoothing(Input_data, waveletname = 'sym7',\
                           modename = 'reflect', Window_list = np.arange(3,21,2),\
                           Delta_Smooth = 2, Max_Iterations = 50):

    #Perform basic smoothing with lower weights for fluctuations
    #Then pick the required number of iterations
    #By finding the window between 3:21 that meets smoothness requirements
    window_arr = []
    itr_arr = []
    for window in Window_list:
        smoothed, var, w_final = perform_smooth(Input_data,Max_Iterations,window)
        var_norm = np.abs(1- var[1:]/var[:-1])
        #go until pretty darn smooth ie window mean var < 0.001
        for i in range(0,len(var_norm)-10):
            if(np.mean(var_norm[i:i+10]) < 0.001):
                window_arr.append(window)
                itr_arr.append(i)
                break
            
    #Now perform the smooothing with the number of iterations selected in the above step
    #Note that one could optimize the speed by combining the following loop and the above loop
    smoothed_data = []
    final_weight_arr = []
    for i in range(0,len(window_arr)):
        smooth, t,weight = perform_smooth(Input_data,itr_arr[i], window_arr[i])
        smoothed_data.append(smooth[-1])
        final_weight_arr.append(weight)
    
    #Now remove the high frequency components using wavelet transformation
    wvlt_smoothed = []
    for smoothed_item in smoothed_data:
        (cA, cD) = pywt.dwt(smoothed_item, waveletname,mode=modename)
        (cA_1, cD_1) = pywt.dwt(cA, waveletname,mode=modename)
        wv_smooth_1 = pywt.idwt(cA_1, None,waveletname)
        wv_smooth = pywt.idwt(wv_smooth_1, None,waveletname)    
        wvlt_smoothed.append(wv_smooth)

    #Use the wavelet transformed smoothed data set as the trend.
    #Then use that trend to apply weights in a smoothing window
    #After this step you get a smooth function with trend corrected.
    #In this step you could smooth with a smaller window
    #New Window size = Old window size - Delta_Smooth is the substration
    trend_corrected_smooth = []
    for i in range(0,len(wvlt_smoothed)):
        theory_weight = 1.0/wvlt_smoothed[i]
        testing_sm = weighted_ma_theory(data,final_weight_arr[i],theory_weight,window_arr[i]-Delta_Smooth)
    
        (cA, cD) = pywt.dwt(testing_sm, waveletname,mode=modename)
        (cA_1, cD_1) = pywt.dwt(cA, waveletname,mode=modename)
        wv_smooth_1 = pywt.idwt(cA_1, None,waveletname)
        wv_smooth = pywt.idwt(wv_smooth_1, None,waveletname)

        trend_corrected_smooth.append(wv_smooth)

    #During the process of wavelet transformation and inverse wavelet transformation it adds extra data points to the smoothed function.
    #Following loop is a quick solution to get the length of the array right. 
    #May be there is a better way to do this.

    final = []
    for item in trend_corrected_smooth:
        Delta = len(item) - len(data)
        right = int(Delta/2)
        left = Delta - right
        if Delta == 0:
            final.append(item)
        elif right <= 0:
            final.append(item[left:])
        else:
            final.append(item[left:-1*right])   
    final = np.array(final)
    return final
#%%
