#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 17:29:34 2019

@author: franchesoni
"""


import numpy as np
import pickle

def persistence(filename, n=int(50000), LT=24):
    '''Returns a list
    [filename, errors, RMSDs]
    of persistence forecasting errors and RMSDs for
    lead times from 1 to LT'''

    file_data = [filename]
    data = np.load('vectors/'+filename)
    m = data[:, 2::]
    
    forecasts2 = []
    for lt in range(1, LT+1):
        forecasts = np.concatenate((np.zeros(lt), m[0:-lt, 0]))
        forecasts2.append(forecasts)
        
    predictions = np.array(forecasts2)
    del forecasts, forecasts2
    
    errors = []  # list of error vectors (for each lt)
    for lt in range(LT):  # for each lead time
        predictions[lt] = (predictions[lt].T * data[:, 1]).T  # kt* to GHI
        errors.append((data[:, 0] - predictions[lt].T).T)   # take the diff
    
    # PLOTS
    second_year_start_index = 100  # start evaluation at (to avoid first obs)
    errors = np.array([error[second_year_start_index::] for error in errors])
    RMSDs = np.sqrt(np.mean(errors ** 2, axis=1)) \
            / np.mean(data[:, 0], axis=0)  # get %RMSD
    
    file_data.append(errors)
    file_data.append(RMSDs)

    with open('data/{}_persistence'.format(filename[5:7]), 'wb') as f:
        pickle.dump(file_data, f, pickle.HIGHEST_PROTOCOL)
    
    return None
   
