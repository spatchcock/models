# -*- coding: utf-8 -*-
"""
Created on Sun May  4 19:18:11 2014

@author: spatchcock
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#  (C{x,t}-C{x,t-1})/dt = D*(C{x+1,t-1} - 2C{x, t-1} + C{x-1,t-1})/dx^2 - w(C{x+1,t-1} - C{x-1,t-1})/2dx - u{x}*C{x,t-1} + Ra{x}
#
# Steady state
#
#  0 = D*(C{x+1,t-1} - 2C{x, t-1} + C{x-1,t-1})/dx^2 - w(C{x+1,t-1} - C{x-1,t-1})/2dx - u{x}*C{x,t-1} + Ra{x}
#
# Re-arrange for u
#
#  u{x} = [D*(C{x+1} - 2C{x} + C{x-1})/dx^2 - w(C{x+1} - C{x-1})/2dx + Ra{x}] / C{x}
#
# At x = 0
#
#  u{0} = [D*(C{2} - 2C{1} + C{0})/mean(dx)^2 - w(C{1} - C{0})/dx + Ra{0}] / C{0}
#
# At x = m (m = 60 cm)
#
#  u{m} = [D*(C{m} - 2C{m-1} + C{m-2})/mean(dx)^2 - w(C{m} - C{m-1})/dx + Ra{m}] / C{m}
#
#
#
#

high_file = '/home/spatchcock/Documents/assemblage/marmic/hippensteel_total_high_marsh.csv'
intm_file = '/home/spatchcock/Documents/assemblage/marmic/hippensteel_total_intermediate_marsh.csv'
low_file  = '/home/spatchcock/Documents/assemblage/marmic/hippensteel_total_low_marsh.csv'

# dataset = np.genfromtxt(high_file, dtype=float, delimiter=',', names=True) 
#dataset = np.genfromtxt(intm_file, dtype=float, delimiter=',', names=True) 
dataset  = np.genfromtxt(low_file,  dtype=float, delimiter=',', names=True)

species = 'JMAC'
depth_data = dataset['Depth']
dead_data  = dataset[species+'_Dead']
live_data  = dataset[species+'_Live']


## High marsh
# mixing_depth = 5.0
# mixing_rate  = 0.24
# sed_rate     = 0.62
##
## Intm marsh
#mixing_depth = 9.0
#mixing_rate  = 0.4175
#sed_rate     = 0.58
#
# Low marsh
mixing_depth = 13.0
mixing_rate  = 0.905
sed_rate     = 1.03

u = np.zeros(np.size(dead_data))

###############################################################################

u[0] = (mixing_rate*(dead_data[2] - 2*dead_data[1] + dead_data[0])/(depth_data[2]/2.0)**2 - sed_rate*(dead_data[1] - dead_data[0])/depth_data[1] + live_data[0]) / dead_data[0]

for i in range(1,(np.size(dead_data)-1)):
    if depth_data[i] <= mixing_depth:
        u[i] = (mixing_rate*(dead_data[i+1] - 2*dead_data[i] + dead_data[i-1])/((depth_data[i+1] - depth_data[i-1])/2.0)**2 - sed_rate*(dead_data[i+1] - dead_data[i-1])/((depth_data[i+1] - depth_data[i-1])) + live_data[i]) / dead_data[i]
    else:
        u[i] = (-sed_rate*(dead_data[i+1] - dead_data[i-1])/(depth_data[i+1] - depth_data[i-1]) + live_data[i]) / dead_data[i]
    
u[np.size(dead_data)-1] = (-sed_rate*(dead_data[np.size(dead_data)-1] - dead_data[np.size(dead_data)-2])/(depth_data[np.size(dead_data)-1] - depth_data[np.size(dead_data)-2]) + live_data[np.size(dead_data)-1]) / dead_data[np.size(dead_data)-1]

###############################################################################

def fitFunc(t, a, b):
   return a*np.exp(-b*t)# + c

x=np.linspace(0,60,500)
fitParams, fitCovariances = curve_fit(fitFunc, depth_data[1:], u[1:],maxfev=2000)
print(' fit coefficients:\n', fitParams)
print(' Covariance matrix:\n', fitCovariances)


#a,b=np.polynomial.polynomial.polyfit(high_data['Depth'],u,3,full=True)
x=np.linspace(0,60,500)
u_int=fitParams[0] * np.exp(-fitParams[1] * x) #+ fitParams[2]
    

###############################################################################

plt.figure()
plt.plot(u,depth_data)
plt.ylim(60,0)
plt.title(species)
plt.plot(u_int,x)
plt.show()