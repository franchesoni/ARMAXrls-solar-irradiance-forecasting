#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 17:06:58 2018

@author: franchesoni
"""


import numpy as np
import os

'''In this script we take the data into numpy arrays for further use'''
# create titles
titles = {
    'Ano':0,
    'Diaano':1,
    'Mes':2,
    'Dia':3,
    'Hora':4,
    'Minuto':5,
    'GHIme':6,
    'GHIcs':7,
    'GHO':8,
    'kC':9,
    'kT':10,
    'kTperez':11,
    'varkC':12,
    'varkT':13,
    'varkTperez':14,
    'cosang':15,
    'hourangle':16,
    'nub':17,
    'IDT':18,
    'mask':19
    }

# load data
directory = 'datos_solar'

for filename in os.listdir(directory):
    if filename.endswith('.csv'):
        data = np.loadtxt(directory+'/'+filename, delimiter=',')
        
        # Choose our variables: kt* (kC), cloudness index (nub),
        # variability (var), GHImeasured, GHIclear-sky
        x = data[:, titles['kC']] 
        nub = data[:, titles['nub']] 
        var = data[:, titles['varkC']]
        GHIme = data[:, titles['GHIme']]
        GHIcs = data[:, titles['GHIcs']]
        
        # This could be more memory efficient, as there is redundancy. Here m
        # is the time series data and datos represents all of the data
        m = np.array([x, nub, var]).transpose()  # create matrix [n_obs x 3]
        datos = np.array([GHIme, GHIcs, x, nub, var]).transpose()
        
        np.save('vectors/m_{}'.format(filename[:-4]), m)
        np.save('vectors/data_{}'.format(filename[:-4]), datos)