#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 16:18:55 2019

@author: franchesoni
"""
import os
import numpy as np
from functions import get_results_for_order
import gc
import matplotlib.pyplot as plt
#%%
order = [5, 0]
filenames = [file for file in os.listdir('vectors') if file[-3::]!='png']
MBDs_per_loc = []  # one element per location
RMSDs_per_loc = []
MBDs_list_loc = []
RMSDs_list_loc = []
Fskills_list_loc = []
for filename in filenames:
    MBDs_per, RMSDs_per, \
    MBDs_list, RMSDs_list, \
    Fskills_list = get_results_for_order(filename, order, csv=True)
    
    MBDs_per_loc.append(MBDs_per)
    RMSDs_per_loc.append(RMSDs_per)
    MBDs_list_loc.append(MBDs_list)
    RMSDs_list_loc.append(RMSDs_list)
    Fskills_list_loc.append(Fskills_list)
  
#%%
gc.collect()
MBDs_per_loc = np.array(MBDs_per_loc)  # 6 x 24
RMSDs_per_loc = np.array(RMSDs_per_loc)  # 6 x 24
Fskills_list_loc = np.array(Fskills_list_loc)  # 6 x 4 x 24
RMSDs_list_loc = np.array(RMSDs_list_loc).squeeze()  # 6 x 4 x 24
MBDs_list_loc = np.array(MBDs_list_loc).squeeze()  # # 6 x 4 x 24
#%%
# TABLE
print(np.round(np.mean(MBDs_per_loc * 100, axis=0), 1)[[0, 1, 2, 3, 4, 5, 11, 17, 23]])
print(np.round(np.mean(RMSDs_per_loc * 100, axis=0), 1)[[0, 1, 2, 3, 4, 5, 11, 17, 23]])
print(np.round(np.mean(MBDs_list_loc[:, 0, :] * 100, axis=0), 1)[[0, 1, 2, 3, 4, 5, 11, 17, 23]])
print(np.round(np.mean(RMSDs_list_loc[:, 0, :] * 100, axis=0), 1)[[0, 1, 2, 3, 4, 5, 11, 17, 23]])
print(np.round(np.mean(Fskills_list_loc[:, 0, :] * 100, axis=0), 1)[[0, 1, 2, 3, 4, 5, 11, 17, 23]])
print(np.round(np.mean(MBDs_list_loc[:, 1, :] * 100, axis=0), 1)[[0, 1, 2, 3, 4, 5, 11, 17, 23]])
print(np.round(np.mean(RMSDs_list_loc[:, 1, :] * 100, axis=0), 1)[[0, 1, 2, 3, 4, 5, 11, 17, 23]])
print(np.round(np.mean(Fskills_list_loc[:, 1, :] * 100, axis=0), 1)[[0, 1, 2, 3, 4, 5, 11, 17, 23]])
print(np.round(np.mean(MBDs_list_loc[:, 2, :] * 100, axis=0), 1)[[0, 1, 2, 3, 4, 5, 11, 17, 23]])
print(np.round(np.mean(RMSDs_list_loc[:, 2, :] * 100, axis=0), 1)[[0, 1, 2, 3, 4, 5, 11, 17, 23]])
print(np.round(np.mean(Fskills_list_loc[:, 2, :] * 100, axis=0), 1)[[0, 1, 2, 3, 4, 5, 11, 17, 23]])
print(np.round(np.mean(MBDs_list_loc[:, 3, :] * 100, axis=0), 1)[[0, 1, 2, 3, 4, 5, 11, 17, 23]])
print(np.round(np.mean(RMSDs_list_loc[:, 3, :] * 100, axis=0), 1)[[0, 1, 2, 3, 4, 5, 11, 17, 23]])
print(np.round(np.mean(Fskills_list_loc[:, 3, :] * 100, axis=0), 1)[[0, 1, 2, 3, 4, 5, 11, 17, 23]])
#%%
# PLOT AVG
plt.close('all')
plt.figure()
plt.plot(np.arange(1, 25)*10,
         100 * np.mean(RMSDs_per_loc, axis=0),
         linewidth=5, color=(0.1, 0.1, 0.1), label='Persistence')

plt.plot(np.arange(1, 25)*10,
         100 * np.mean(RMSDs_list_loc, axis=0)[0],
         '--', linewidth=4, color=(0.5, 0.5, 0), label='PGM')
plt.plot(np.arange(1, 25)*10,
         100 * np.mean(RMSDs_list_loc, axis=0)[1],
         '--', linewidth=4, color=(0, 0.5, 0), label='PGM - S')
plt.plot(np.arange(1, 25)*10,
         100 * np.mean(RMSDs_list_loc, axis=0)[2],
         '-o', linewidth=1, color=(0.5, 0, 0), label='PGM - V')
plt.plot(np.arange(1, 25)*10,
         100 * np.mean(RMSDs_list_loc, axis=0)[3],
         '-o', linewidth=1, color=(0, 0, 0), label='PGM - SV')
plt.ylim(15, 50)
plt.ylabel('Relative RMS deviation (%)')
plt.xlabel('Lead Time (min)')
plt.legend(loc=2)
#%%
plt.close(2)
plt.figure()
plt.plot(np.arange(1, 25)*10,
         100 * np.mean(Fskills_list_loc, axis=0)[0],
         '--', linewidth=4, color=(0.5, 0.5, 0), label='PGM')
plt.plot(np.arange(1, 25)*10,
         100 * np.mean(Fskills_list_loc, axis=0)[1],
         '--', linewidth=4, color=(0, 0.5, 0), label='PGM - S')
plt.plot(np.arange(1, 25)*10,
         100 * np.mean(Fskills_list_loc, axis=0)[2],
         '-o', linewidth=1, color=(0.5, 0, 0), label='PGM - V')
plt.plot(np.arange(1, 25)*10,
         100 * np.mean(Fskills_list_loc, axis=0)[3],
         '-o', linewidth=1, color=(0, 0, 0), label='PGM - SV')
#plt.ylim(15, 50)
plt.ylabel('Forecasting Skill (%)')
plt.xlabel('Lead Time (min)')
plt.legend(loc=1)
