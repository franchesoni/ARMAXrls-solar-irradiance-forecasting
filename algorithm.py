#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 13:59:51 2018

@author: franchesoni
"""

import numpy as np

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
