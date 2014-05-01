# -*- coding: utf-8 -*-
"""
Created on Sun Apr 27 19:42:14 2014

@author: spatchcock
"""

import matplotlib.pyplot as plt
import numpy as np

# Advection-diffusion-decay-source equation
#
#   D(d2C/dx2) - w(dc/dx) - uC + R(x) = 0
#
# Discretized equation
#
#   D[C{x+1} - 2*C{x} + C{x-1}]/dx**2 - w(C{x+1}= C{x-1})/2dx - uC{x} + R(x) = 0
#
# Tridiagonal form for interior points
#
#   (D/dx**2 + w/(2*dx))*C{x-1} + (-(2*D)/(dx**2) - u)*C{x} + (D/dx**2 - w/(2*dx))*C{x+1} = -R(x)
#
# So
#   a = D/dx**2 + w/(2*dx)
#   b = -(2*D)/(dx**2) - u
#   c = D/dx**2 - w/(2*dx)
#
#   d = -R(x)
#
# Boundaries
#
#   C{0} = A
#   C{m} = Z
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

# Setup model parameters

xStart = 0.0 # x at left end
xStop = 30.0 # x at right end
m = 1000     # Number of mesh spaces
dx = (xStop - xStart)/m # step size
x = np.arange(xStart,xStop + dx, dx)

u = 0.07 # decay
D = 0.008  # diffusion
w = 0.5 # advection
R = 50.0*np.exp(-0.1*x) # production

A = 200.0 # boundary x(0)
Z = 100.0 # boundary x(m)


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
plot(x[0:-2], C)
ylim(0)
xlim([0,xStop+1])
grid()



