#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 18:00:44 2019

@author: franchesoni
"""

import os
import numpy as np
from functions import evaluate

'''Evaluate the performance of  orders  over the places in  vectors  and save
   the predictions, the orders, and the RMSDs'''
orders = [(1, 0), (2, 0), (3, 0), (4, 0), (5, 0),
          (6, 0), (7, 0), (8, 0), (9, 0), (10, 0),
          (0, 1), (1, 1), (2, 1), (3, 1), (4, 1),
          (5, 1), (6, 1), (7, 1), (8, 1), (9, 1), (10, 1),
          (0, 2), (1, 2), (2, 2), (3, 2), (4, 2),
          (5, 2), (6, 2), (7, 2), (8, 2), (9, 2), (10, 2),
          (0, 3), (1, 3), (2, 3), (3, 3), (4, 3),
          (5, 3), (6, 3), (7, 3), (8, 3), (9, 3), (10, 3)]

filenames = [file for file in os.listdir('vectors') if file[-3::]='npy']
for filename in filenames:
    evaluate(filename, orders, csv=True)

#filename = os.listdir('vectors')[-1]
#evaluate(filename, orders, csv=True)