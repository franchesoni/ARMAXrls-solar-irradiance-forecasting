#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 13:58:15 2018

@author: franchesoni
"""

import numpy as np
import matplotlib.pyplot as plt
import algorithm
import pickle
import os

from functools import partial
from multiprocessing import Pool
from time import time


def evaluate(filename, cloudness_index=True, two_orders=True, n=int(50000), LT=24):
        
    def build_EX(ex):
        if ex==[0]:
            EX = np.zeros(n)
        else:
            EX = m[:, ex].T
        if len(EX.shape) < 2:
            EX = np.expand_dims(EX, axis=0)
        return EX
    
    start = time()
    print(filename)
    
    data = np.load('vectors/'+filename)  # GHIme, GHIcs, m
    
    # Select only n samples
    m = data[:, 2::][0:n, :] 
    data = data[0:n, :]
    
#    legend = []  # titles will be added here
    # We evaluate different [order, ex_var]
    evaluate = [[(2, 1), [0]], [(2, 1), [1]], [(2, 1), [2]], [(2, 1), [1, 2]],
                [(4, 2), [0]], [(4, 2), [1]], [(4, 2), [2]], [(4, 2), [1, 2]]]
    
    if not cloudness_index:  # NEW
        evaluate = [x for x in evaluate if 1 not in x[1]]
    if not two_orders:
        evaluate = [x for x in evaluate if x[0]==(2, 1)]
    
    print('...')
    file_data = [filename]
    for i, (order, ex) in enumerate(evaluate):  # for each model
        file_data.append((order, ex))
        
        p, q = order[0], order[1]  # get the order of the AR and MA terms
        EX = build_EX(ex)  # get exogenous variables
        # GET PREDICTIONS
        
        armarls = partial(algorithm.ARMArls_X, m[:, 0], EX, p, q, 0.999)
        forecasts2 = []  # list of different lead time forecasts
        with Pool(os.cpu_count()-1) as p:
            forecasts2 = p.map(armarls, range(1, LT+1))
        
        predictions = np.array(forecasts2)  # make array of list
        del forecasts2  # clean
        
        # GET ERRORS
        errors = []  # list of error vectors (for each lt)
        for lt in range(LT):  # for each lead time
            predictions[lt] = (predictions[lt].T * data[:, 1]).T  # kt* to GHI
            errors.append((data[:, 0] - predictions[lt].T).T)   # take the diff
        
        second_year_start_index = 100  # start evaluation at (avoid first obs)
        errors = np.array([error[second_year_start_index::] \
                           for error in errors])
        RMSDs = np.sqrt(np.mean(errors ** 2, axis=1)) \
                / np.mean(data[:, 0], axis=0)  # get %RMSD
           
        file_data.append(errors)
        file_data.append(RMSDs)
    
    with open('data/{}_data'.format(filename[5:7]), 'wb') as f:
        pickle.dump(file_data, f, pickle.HIGHEST_PROTOCOL)
                
    end = time()
    print(end-start, 's')
    print()
    return None