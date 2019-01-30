#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 12:36:36 2019

@author: franchesoni
"""

import numpy as np
import os
import pickle

per_data = []
rls_data = []
for file in os.listdir('data'):
    if file[3::]=='persistence':
        with open('data/'+file, 'rb') as f:
            per_data.append(pickle.load(f))
    else:
        with open('data/'+file, 'rb') as f:
            rls_data.append(pickle.load(f))
            
for d in per_data:
    np.savetxt('csv/'+d[0][5:7]+'_persistence_errors.csv', d[1].T, delimiter=",")
    print(d[0])

print()
print('-------------------------')
print()

for d in rls_data:
    np.savetxt('csv/'+d[0][5:7]+'_rls_errors.csv', d[2][:, :, 0].T, delimiter=",")
    print(d[0])
