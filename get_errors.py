#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 15:57:21 2019

@author: franchesoni
"""

import numpy as np
import os
import

from evaluate import evaluate
from persistence import persistence

data_filenames = ['TA', 'LB', 'LE', 'RC', 'ZU']
for filename in os.listdir('vectors'):
    if filename[0:5] == 'data_' and filename[-7:-4] == 'fil' \
    and filename[5:7] in data_filenames:
        evaluate(filename, two_orders=False)
        persistence(filename)

