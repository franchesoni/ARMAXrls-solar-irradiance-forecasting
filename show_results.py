#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 18:11:31 2019

@author: franchesoni
"""

import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib import cm
from matplotlib import rc, font_manager
from functions import amp, stridency
#%%
res_dir = 'results'
filenames = os.listdir(res_dir)
acronyms = set([filename[0:2] for filename in filenames if filename[-3::]!='png'])

orders = []  # places x 43
predictions = []  # places x 43 x 24 x 44120
RMSDs_loc = []  # places x 43 x 24

for acronym in acronyms:
    path = res_dir + '/' + acronym + '_'
    
    orders.append(np.load(path + 'orders'))
    if orders[-1] != orders[0]:
        raise ValueError('inconsistent orders')
    predictions.append(np.array(np.load(path + 'predictions')).squeeze())
    RMSDs_loc.append(np.array(np.load(path + 'RMSDs')).squeeze())
    
    print('-----------------------------')
    print(acronym)
    print('-----------------------------')
    print()

RMSDs_loc = np.array(RMSDs_loc)
orders = orders[-1]
RMSDs = np.mean(RMSDs_loc, axis=0).T
#%%
RMSDs_centered = RMSDs - np.mean(RMSDs, axis=1).reshape(RMSDs.shape[0], 1)
RMSDs_ranked = np.argsort(RMSDs, axis=1)

RMSDs_centered_avg = np.mean(RMSDs_centered, axis=0)
RMSDs_ranked_avg = np.mean(RMSDs_ranked, axis=0)

RMSDs_min = np.mean(np.min(RMSDs_loc, axis=1), axis=0)
#amplitudes = np.mean(np.array([amp(prediction) for prediction in predictions]),axis=0).T
#stridencies = np.mean(np.array([stridency(prediction) for prediction in predictions]), axis=0).T
#%%
lts = [0, 8, 23]

sizeOfFont = 10
fontProperties = {'weight' : 'normal', 'size' : sizeOfFont}
colors = [(0.5, 0.5, 0, 0.5), (0, 0.5, 0, 0.5),
          (0.5, 0, 0, 0.5), (0, 0, 0, 0.5)]


LT = 24
plt.close('all')
plt.rcParams.update({'font.size': 15})
fig = plt.figure()
ax = plt.gca()
plt.plot(np.arange(1, LT+1), RMSDs_min*100, '-o', linewidth=5)
plt.plot(np.arange(1, LT+1), RMSDs[:, 4]*100)
ax.set_yticks([20, 25, 30, 35, 40]) 
ax.set_ylim([18, 40])
ax.set_ylabel('Relative RMS deviation (%)')
ax.set_xlabel('Lead Time (10 minutes time step)')
ax.set_xticks([0, 5, 10, 15, 20])
ax.set_yticks([20, 25, 30, 35, 40]) 
ax.set_xticklabels(a.get_xticks(), fontProperties)
ax.set_yticklabels(a.get_yticks(), fontProperties)
plt.legend(['Optimum (p, q)', 'order = (5, 0)'])

#
## PLOTEAR PARCIAL A TIEMPOS lts
#lts = [0, 8, 23]
#
#sizeOfFont = 10
#fontProperties = {'weight' : 'normal', 'size' : sizeOfFont}
#colors = [(0.5, 0.5, 0, 0.5), (0, 0.5, 0, 0.5),
#          (0.5, 0, 0, 0.5), (0, 0, 0, 0.5)]
#
##plt.close('all')
#plt.rcParams.update({'font.size': 15})
#fig = plt.figure()
#ax = fig.add_subplot(1, 1, 1)
##plt.title('Average %RMSD')
#a = plt.gca()
#for i, order in zip(range(len(orders)-1, -1, -1), orders[::-1]):
#    for i, order in enumerate(orders):
#        ax.plot(np.arange(1, LT+1), RMSDs[:, i] * 100, color=colors[order[1]], label='MA order = {}'.format(order[1]))
#display = (1, 15, 29, 42)
#handles, labels = ax.get_legend_handles_labels()
#ax.legend([handle for i,handle in enumerate(handles) if i in display],
#      [label for i,label in enumerate(labels) if i in display], loc = 'best')
##ax.set_ylim(0.15, 0.45)
#ax.set_ylabel('Relative RMS deviation (%)')
#ax.set_xlabel('Lead Time (10 minutes time step)')
#ax.set_xticks([0, 5, 10, 15, 20])
#ax.set_yticks([20, 25, 30, 35, 40]) 
#a.set_xticklabels(a.get_xticks(), fontProperties)
#a.set_yticklabels(a.get_yticks(), fontProperties)
#
#plt.savefig('bf_orders.png')
#
#
#plt.rcParams.update({'font.size': 15})
#columns = np.arange(11)  # AR terms
#rows = np.arange(4)
#table1 = np.empty((4, 11))
#table2 = np.empty((4, 11))
#r, c = np.meshgrid(rows, columns)
#
#fig = plt.figure(figsize=plt.figaspect(0.3))
#for index, lt in enumerate(lts):
#    # Crear tabla
#    for i in rows:
#        for j in columns:
#            table1[i, j] = np.nan
#            ls = [a for a, x in enumerate(orders) if x[0] == j and x[1]==i]
#            if len(ls) == 1:
#                table1[i, j] = RMSDs[lt, ls[0]]
#    z1 = table1.T * 100
#    ax = fig.add_subplot(1, 3, index+1, projection='3d')
#    plt.tick_params(labelsize=8)
#    norm = mpl.colors.Normalize(vmin=np.amin(z1[1::, :]),
#                                vmax=np.amax(z1[1::, :]))
#    surf = ax.plot_surface(r[1::, :], c[1::, :], z1[1::, :],
#                           facecolors=plt.cm.jet(norm(z1[1::, :])),
#                           linewidth=1, antialiased=False)
#    surf.set_facecolor((0,0,0,0))   
#    ax.set_title('LT = {}m'.format((lt+1)*10))
#    ax.set_ylim(10, 0)
#    ax.set_ylabel('AR order')
#    ax.set_xlim(0, 3)
#    ax.set_xlabel('MA order')
#    ax.set_xticks([0, 1, 2, 3]) 
#    ax.set_yticks([1, 2, 4, 6, 8, 10]) 
#plt.show()
#fig.tight_layout()
#plt.savefig('rmsds.png')
#
#fig = plt.figure()
## Crear tabla
#for i in rows:
#    for j in columns:
#        table1[i, j] = np.nan
#        table2[i, j] = np.nan
#        ls = [a for a, x in enumerate(orders) if x[0] == j and x[1]==i]
#        if len(ls) == 1:
#            table1[i, j] = RMSDs_centered_avg[ls[0]]
#            if i + j <= 6:
#                table2[i, j] = RMSDs_centered_avg[ls[0]]
#z1 = table1.T * 100
#z2 = table2.T * 100
##ax = fig.add_subplot(1, 3, index+1, projection='3d')
#ax = fig.gca(projection='3d')
#plt.tick_params(labelsize=8)
#norm = mpl.colors.Normalize(vmin=np.amin(z1[1::, :]),
#                            vmax=np.amax(z1[1::, :]))
#surf = ax.plot_surface(r[1::, :], c[1::, :], z1[1::, :],
#                       facecolors=plt.cm.jet(norm(z1[1::, :])),
#                       linewidth=1, antialiased=False)
#surf2 = ax.plot_surface(r[1::, :], c[1::, :], z2[1::, :],
#                       facecolors=plt.cm.jet(norm(z2[1::, :])),
#                       linewidth=1, antialiased=False)
#surf.set_facecolor((0,0,0,0))   
#ax.set_ylim(10, 0)
#ax.set_ylabel('AR order')
#ax.set_xlim(0, 3)
#ax.set_xlabel('MA order')
#ax.set_xticks([0, 1, 2, 3])
#ax.set_yticks([1, 2, 4, 6, 8, 10])
#plt.show()
#fig.tight_layout()
#plt.savefig('average_anomalies.png')
















































#%%
#fig = plt.figure()
#fig.suptitle('RMSDs ranked')
## Crear tabla
#for i in rows:
#    for j in columns:
#        table1[i, j] = np.nan
#        table2[i, j] = np.nan
#        ls = [a for a, x in enumerate(orders) if x[0] == j and x[1]==i]
#        if len(ls) == 1:
#            table1[i, j] = RMSDs_ranked_avg[ls[0]]
#            if i + j <= 6:
#                table2[i, j] = RMSDs_ranked_avg[ls[0]]
#z1 = table1.T
#z2 = table2.T
##ax = fig.add_subplot(1, 3, index+1, projection='3d')
#ax = fig.gca(projection='3d')
#plt.tick_params(labelsize=8)
#norm = mpl.colors.Normalize(vmin=np.amin(z1[1::, :]),
#                            vmax=np.amax(z1[1::, :]))
#surf = ax.plot_surface(r[1::, :], c[1::, :], z1[1::, :],
#                       facecolors=plt.cm.jet(norm(z1[1::, :])),
#                       linewidth=1, antialiased=False)
#surf2 = ax.plot_surface(r[1::, :], c[1::, :], z2[1::, :],
#                       facecolors=plt.cm.jet(norm(z2[1::, :])),
#                       linewidth=1, antialiased=False)
#surf.set_facecolor((0,0,0,0))   
#ax.set_ylim(10, 0)
#ax.set_ylabel('AR order')
#ax.set_xlim(0, 3)
#ax.set_xlabel('MA order')
#ax.set_xticks([0, 1, 2, 3]) 
#plt.show()
#


#    point = (6, 0)
#    ax.scatter([point[1]], [point[0]], z[point[0], point[1]],
#               s=300, c='r', marker='.', zorder=10)
#    
    
    
    
    
    
