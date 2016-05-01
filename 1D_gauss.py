# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 22:35:38 2016

@author: spatchcock
"""

import matplotlib.pyplot as plt
import numpy

domain_range = 3
dx = 0.001
meanx   = 0.0
stdevx  = 0.05

mass = 1000.0

A = mass/(stdevx*numpy.sqrt(2*numpy.pi))

x = numpy.arange(-domain_range/2,domain_range/2,dx)

def one_d_gauss(x, A, meanx, stdevx):
    
    g = numpy.zeros(numpy.size(x))
    
    for i in numpy.arange(0, numpy.size(x), 1):
        g[i] = A * numpy.exp(-((x[i] - meanx)**2.0)/(2.0*(stdevx**2.0)))
    
    return g

y = one_d_gauss(x, A, meanx, stdevx)

#%% plot

fig = plt.figure()

# 2D
gauss = fig.add_subplot(111,xlim=(min(x), max(x)), ylim=(0, max(y)))
gauss.plot(x,y, lw=3, color='b')
gauss.grid()

plt.show()

#%%

level = 0.1

within_contour = []

for i in numpy.arange(0, numpy.size(x), 1):
    if y[i] >= level:
        within_contour.append(y[i]*dx)

totalInt = sum(within_contour)
pedestal = level * numpy.size(within_contour)*dx
positiveInt = totalInt - pedestal
positiveFraction = positiveInt/mass

# find x for a given y
# plus or minus mean for both values
xDistance = numpy.sqrt(-numpy.log(level/A)*(2.0*(stdevx**2.0))) - meanx
normalizedXDistance = xDistance/mass

print(totalInt)
print(pedestal)
print(positiveInt)
print(positiveFraction)
print(xDistance)
print(normalizedXDistance)