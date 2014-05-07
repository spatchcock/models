# -*- coding: utf-8 -*-
"""
Created on Fri May  2 23:21:00 2014

@author: spatchcock
"""

import numpy as np
import matplotlib.pyplot as plt

high_file = '/home/spatchcock/Documents/assemblage/marmic/hippensteel_total_high_marsh.csv'
intm_file = '/home/spatchcock/Documents/assemblage/marmic/hippensteel_total_intermediate_marsh.csv'
low_file  = '/home/spatchcock/Documents/assemblage/marmic/hippensteel_total_low_marsh.csv'

high_data = np.genfromtxt(high_file, dtype=float, delimiter=',', names=True) 
intm_data = np.genfromtxt(intm_file, dtype=float, delimiter=',', names=True) 
low_data  = np.genfromtxt(low_file,  dtype=float, delimiter=',', names=True) 


# High marsh
#  mixed layer: 5 cm (max)
#  mixing rate: 0.24 cm2/yr
#  burial rate: 6.2 mm/yr
#
#
# Intm marsh
#  mixed layer: 9 cm (max)
#  mixing rate: 0.4175 cm2/yr
#  burial rate: 5.8 mm/yr
#
#
# Low marsh
#  mixed layer: 13 cm (max)
#  mixing rate: 0.905 cm2/yr
#  burial rate: 10.3 mm/yr
#
#

# TDMA solver, a b c d can be NumPy array type or Python list type.
# Refer to http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
# 
# Source: https://gist.github.com/ofan666/1875903
#
def TDMAsolver(a, b, c, d):
    nf = len(a) # number of equations
    ac, bc, cc, dc = map(np.array, (a, b, c, d)) # copy the array
    for it in xrange(1, nf):
        mc = ac[it]/bc[it-1]
        bc[it] = bc[it] - mc*cc[it-1]
        dc[it] = dc[it] - mc*dc[it-1]
     
    xc = ac
    xc[-1] = dc[-1]/bc[-1]
     
    for il in xrange(nf-2, -1, -1):
        xc[il] = (dc[il]-cc[il]*xc[il+1])/bc[il]
     
    del bc, cc, dc # delete variables from memory
     
    return xc


fig = plt.figure()
plot = fig.add_subplot(1,1,1, ylim=(60.0, 0))
plot.plot(high_data['MFUS_Dead'],high_data['Depth'], marker='o', linestyle='None')
plot.plot(high_data['MFUS_Live'],high_data['Depth'], marker='o', linestyle='None')
plot.set_title('MFUS')
plot.grid()

a,b=np.polynomial.polynomial.polyfit(high_data['Depth'],high_data['MFUS_Live'],5,full=True)
    
# Setup model parameters

xStart = 0.0 # x at left end
xStop = 60.0 # x at right end
m = 1000     # Number of mesh spaces
dx = (xStop - xStart)/m # step size
x = np.arange(xStart,xStop + dx, dx)

u = 0.3 # decay
D = 0.24  # diffusion
w = 1.03 # advection
R = a[0] + a[1]*x + a[2]*x**2 + a[3]*x**3 + a[4]*x**4 + a[5]*x**5 # production

A = high_data['MFUS_Dead'][0] # boundary x(0)
Z = high_data['MFUS_Dead'][-1] # boundary x(m)


# Set up finite difference eqs.
    
# Interior points
a = np.ones((m))*(D/dx**2 + w/(2*dx))
b = np.ones((m + 1))*(-(2*D)/(dx**2) - u)
c = np.ones((m))*(D/dx**2 - w/(2*dx))
d = np.ones((m+2))*(-R)

# Boundary value x(0)
b[0] = 1.0
c[0] = 0.0
d[0]  = A

# Boundary value x(m)
b[-1] = 1.0
a[-1] = 0.0
d[-2] = Z # This should be unecessary. An extra node has crept in somewhere
d[-1] = Z


# Solve

C = TDMAsolver(a,b,c,d)
plot.plot(R,x)
plot.plot(C,x[0:-2])
#ylim(0)
#xlim([0,xStop+1])
#grid()