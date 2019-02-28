#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 13:11:07 2019

@author: franchesoni
"""


import numpy as np
import os
import pickle
import gc

from functools import partial
from multiprocessing import Pool
from time import time

def evaluate(filename, orders=[(5, 0)], n=int(50000), LT=24):
    start = time()
    
    EX = np.zeros(n)  # no exogenous input
    EX = np.expand_dims(EX, axis=0)  # reshape to avoid errors
    data = np.load('vectors/'+filename)  # load data
    print('mean = ', np.mean(data[:, 0], axis=0))
    print('anual solar irradiance =',
          np.sum(600*data[:, 0])
          / (2*3600*1000*1000))  # 600s = 10 min. 2 years. 3.6E9 to MWh/m2
    m = data[0:n, 2::]  # cut the data until n
    data = data[0:n, :]  # cut the data until n
    print('...')
    orders_list = []
    RMSDs_list = []
    predictions_list = []
    for i, order in enumerate(orders):  # for each order
        print('started ', order)
        orders_list.append(order)  # add title to output
        p, q = order[0], order[1]  # get number of lags

        # Change of variable to allow multiprocessing
        armarls = partial(ARMArls_X, m[:, 0], EX, p, q, 0.999)
        predictions = []  # initialize as empty list
        with Pool(os.cpu_count()-1) as p:  # multiprocess for each lead time
            predictions = p.map(armarls, range(1, LT+1))
        predictions = np.array(predictions)
        print('...')
        # Find errors
        errors = []
        for lt in range(LT):
            predictions[lt] = (predictions[lt].T * data[:, 1]).T  # kt* x GHIcs
            errors.append((data[:, 0] - predictions[lt].T).T)
        start_index = 400  # cut series, ignore transient
        errors = np.array([error[start_index::] for error in errors])
        RMSDs = np.sqrt(np.mean(errors ** 2, axis=1)) \
            / np.mean(data[:, 0], axis=0)  # get relative RMSD
        
        predictions_list.append(predictions)
#        errors_list.append(errors)
        RMSDs_list.append(RMSDs)
    
#    errors_per = []
#    for lt in range(LT):
#        errors_per.append(data[lt+1::, 0]
#                          - data[0:-lt-1, 2] * data[lt+1::, 1])
#    errors_per = np.array([error[0:m.shape[0]-LT] for error in errors_per])  # -1 because diff, +1 because python
#    RMSDs_per = np.sqrt(np.mean(errors_per**2, axis=1)) \
#                    / np.mean(data[:, 0], axis=0)
    
    # Save
#    with open('results/{}_errors_per'. format(filename[5:7]), 'wb') as f:
#        pickle.dump(errors_per, f, pickle.HIGHEST_PROTOCOL)
#    with open('results/{}_RMSDs_per'. format(filename[5:7]), 'wb') as f:
#        pickle.dump(RMSDs_per, f, pickle.HIGHEST_PROTOCOL)
    with open('results/{}_predictions'. format(filename[5:7]), 'wb') as f:
        pickle.dump(predictions_list, f, pickle.HIGHEST_PROTOCOL)
    with open('results/{}_orders'. format(filename[5:7]), 'wb') as f:
        pickle.dump(orders_list, f, pickle.HIGHEST_PROTOCOL)
#    with open('results/{}_errors'. format(filename[5:7]), 'wb') as f:
#        pickle.dump(errors_list, f, pickle.HIGHEST_PROTOCOL)
    with open('results/{}_RMSDs'. format(filename[5:7]), 'wb') as f:
        pickle.dump(RMSDs_list, f, pickle.HIGHEST_PROTOCOL)
    
    # Finish
    end = time()
    print('COMPLETED')
    print(end-start, 's')
    print()
    return None

def get_results_for_order(filename, order, n=int(50000), LT=24):
    '''Calculate for the different models MBE, RMSD, of both ARMA forecasts and
    persistence. Calculate FSkill of ARMA forecasts'''
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
    
    exs = [[0], [1], [2], [1, 2]]
    data = np.load('vectors/'+filename)  # load data
    print('mean = ', np.mean(data[:, 0], axis=0))
    print('anual solar irradiance =',
          np.sum(600*data[:, 0])
          / (2*3600*1000*1000))  # 600s = 10 min. 2 years. 3.6E9 to MWh/m2
    m = data[0:n, 2::]  # cut the data until n
    data = data[0:n, :]  # cut the data until n
    print('...')
    MBDs_list = []
    RMSDs_list = []  # one element per model
    predictions_list = []
    errors_list = []
    p, q = order[0], order[1]  # get number of lags

    for ex in exs:
        print('started ', order, ex)
        EX = build_EX(ex)
        # Change of variable to allow multiprocessing
        armarls = partial(ARMArls_X, m[:, 0], EX, p, q, 0.999)
        predictions = []  # initialize as empty list
        with Pool(os.cpu_count()-1) as pul:  # multiprocess for each lead time
            predictions = pul.map(armarls, range(1, LT+1))
        del pul
        predictions = np.array(predictions)
        print('...')
        # Find errors
        errors = []
        for lt in range(LT):
            predictions[lt] = (predictions[lt].T * data[:, 1]).T  # kt* x GHIcs
            errors.append((data[:, 0] - predictions[lt].T).T)
        start_index = 400  # cut series, ignore transient
        errors = np.array([error[start_index::] for error in errors])
        RMSDs = np.sqrt(np.mean(errors ** 2, axis=1)) \
            / np.mean(data[:, 0], axis=0)  # get relative RMSD
        MBDs = np.mean(errors, axis=1) / np.mean(data[:, 0], axis=0)
        gc.collect()
        
        predictions_list.append(predictions)
        errors_list.append(errors)
        RMSDs_list.append(RMSDs)
        MBDs_list.append(MBDs)
    
    errors_per = []
    for lt in range(LT):
        errors_per.append(data[lt+1::, 0]
                          - data[0:-lt-1, 2] * data[lt+1::, 1])
    errors_per = np.array([error[0:m.shape[0]-LT] for error in errors_per])  # -1 because diff, +1 because python
    RMSDs_per = np.sqrt(np.mean(errors_per**2, axis=1)) \
                    / np.mean(data[:, 0], axis=0)
    MBDs_per = np.mean(errors_per, axis=1) / np.mean(data[:, 0], axis=0)
    Fskills_list = [1 - RMSDs.squeeze() / RMSDs_per for RMSDs in RMSDs_list]
    
    # Finish
    end = time()
    print('COMPLETED')
    print(end-start, 's')
    print()
    return MBDs_per, RMSDs_per, MBDs_list, RMSDs_list, Fskills_list


def ARMArls_X(X, Ex, p, q, forget, h):
    """
    X : vector of measurments [n]
    h : horizon of forecast
    p : lag AR
    q : lag MA
    forget : forgetting factor
    Ex: exogenous variables [dim x n]
    -------------------
    Returns:
    OUT [n x 1] = zeros(m1+1+h), forecasts
    """
    dim = Ex.shape[0]
    param = np.zeros((1+p+q+dim, X.shape[0]))
    IN = np.zeros((1+p+q+dim, X.shape[0]))
    gain = np.empty((1+p+q+dim, 1+p+q+dim, X.shape[0]))
    OUT = np.zeros((X.shape[0], 1))

    # init at position max(p,q,h) - 1 (python)
    m1 = max([p, q, h])-1
    gain[:, :, m1] = np.eye(1+p+q+dim) * 1000
    # forecasting
    for m in range(m1+1, X.shape[0]-h):

        IN[:, m] = np.concatenate((np.ones(1), X[m-p+1:m+1], X[m+1-q:m+1] - OUT[m-q+1:m+1, 0], Ex[:, m]))
            
        OUT[m+h] = IN[:, m] @ param[:, m-1]

        inmlh = IN[:, m-h].reshape((IN.shape[0], 1))  # in[m-h] (m lagged h)
        gain[..., m] = 1 / forget * (gain[..., m-1] -
            gain[..., m-1] @ inmlh @ inmlh.T @ gain[..., m-1] /
            (forget + inmlh.T @ gain[..., m-1] @ inmlh))

        param[:, m] = param[:, m-1] + gain[..., m] @ inmlh @ (X[m] -
             inmlh.T @ param[:, m-1])
        
        # I suggest to put here output calculated with updated params
        OUT[m+h] = IN[:, m] @ param[:, m]
        
    return OUT

def stridency(prediction):
    diff = prediction[:, :, 1::] - prediction[:, :, 0:-1]
    return np.std(diff, axis=-1)

def amp(prediction):
    return np.std(prediction, axis=-1)