# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 21:20:55 2014

@author: spatchcock
"""

# Plot a 2D dimensional gaussian surface
#
# f(x,y) = A exp { - (((x - meanx)^2) / 2stdevx^2) + ((y - meany)^2) / 2stdevy^2) }
#
# For a 1D normal distribution 68% of the area under the curve lies
# within 1 standard deviation of the mean. I wanted to figure out the 
# corresponding proprtion for a 2D normal distribution.
#
# Turns out to be ~39.3% (and verified analytically elsewhere)

import numpy
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

domain_range = 500

A = 100
meanx   = 0.0
stdevx  = 60.0
meany   = 0.0
stdevy  = 60.0

x = numpy.arange(-domain_range/2,domain_range/2,1)
y = numpy.arange(-domain_range/2,domain_range/2,1)

def two_d_gauss(x, y, A, meanx, meany, stdevx, stdevy):
    
    g = numpy.zeros((size(x), size(y)))
    
    for i in numpy.arange(0, size(x), 1):
        for j in numpy.arange(0, size(y), 1):
            g[i][j] = A * exp(-((((x[i] - meanx)**2.0)/(2.0*(stdevx**2.0))) + (((y[j] - meany)**2.0)/(2.0*(stdevy**2.0)))))
    
    return g

g = two_d_gauss(x, y, A, meanx, meany, stdevx, stdevy)

### Plot ###
fig = plt.figure()

# 2D
# gauss = fig.add_subplot(111, xlim=(min(x), max(x)), ylim=(0, A))
# line, = gauss.plot([], [], lw=3, color='b')
# line.set_data(x,g[100])

# 3D
# http://stackoverflow.com/questions/9170838/surface-plots-in-matplotlib

ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(x, y)
Z = g.reshape(X.shape)

ax.plot_surface(X, Y, g)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()


# Calculate the volume within 1 standard deviation as proportion of whole
#
# (Integration is simple since grid resolution is unity. No *dx *dy needed.)
#
within_one_stdev = []
for i in numpy.arange(0, size(x), 1):
        for j in numpy.arange(0, size(y), 1):
            
            # Use pythagoras to find those cells within 1 std dev
            if math.sqrt(x[i]**2 + y[j]**2) <= stdevx:
                within_one_stdev.append(g[i][j])
            
            
print("Proportion of volume underneath a 2D gaussian surface within 1 standard deviation from the mean is...")
print(sum(within_one_stdev)/sum(g))

# Calculate the area within 1 standard deviation as proportion of whole for just the central cross-section

print("Proportion of area underneath a 1D gaussian surface within 1 standard deviation from the mean is...")
print(sum(g[domain_range/2, domain_range/2-stdevy:domain_range/2+stdevy])/sum(g[domain_range/2]))