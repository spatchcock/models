# -*- coding: utf-8 -*-
"""
Created on Mon May  5 23:39:06 2014

@author: spatchcock
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

depth = [0,1,3,10,20,30,40,50,60]
res   = np.array([2.51, 2.85, 2.7, 2.23, 2.04,2.73,5.33,11.4,83.0])

plt.plot(depth, 1/res)
plt.show()