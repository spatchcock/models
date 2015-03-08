# -*- coding: utf-8 -*-
"""
Created on Fri Mar 06 23:49:57 2015

@author: andrew.berkeley
"""

import numpy as np                 #loading our favorite library
import matplotlib.pyplot as plt    #and the useful plotting library

# This script repersents step 3 of Lorena Barba's "12 steps to Navier Stokes" series:
#
#  http://nbviewer.ipython.org/github/barbagroup/CFDPython/blob/master/lessons/04_Step_3.ipynb
#
# It describes a simple numerical implementation of 1-dimensional diffusion. The
# diffusion equation is:
#
#   du/dt = v (d^2 u / dx^2)
#
# where:
#  u is some property of interest
#  t is time
#  x is 1-dimensional space
#  v is the diffusion rate
#
# The second order derivative can be approximated by adding together the forward-
# and backward-difference versions of the Taylor series (which cancels every second 
# term) and rearranging for the second derivative. This produces:
#
#   d^2 u / dx^2 = (u[i+1] - 2u[i] + u[i-1]) / dx^2    +   O(dx^2)
#
# which is therefore accurate O(dx^2). 
#
# The soution described here iterates a top hat function through time according to
# the diffusion equation, and animates each timestep on a plot.
#
# Points to note
# --------------
#
# 1. This equation is a second order, partial differential equation. Diffusion in
#    time is proportional to the second derivative of the function in space.
#
# 2. The discritization is a "central difference" rather than a backward or forward
#    difference. This is the reason for the increase in the order of accuracy (it is
#    derived by adding the forward and backward Taylor series expansions).
#
# 3. The dt value is defined in relation to dx**2 as opposed to just dx as in the 
#    previous examples. Presumably this is because if the dx**2 which appears in the
#    discritized equation. Also the diffusion speed appears explicitly in the definition
#    of dt (previously it was always unity so ignored).
#
# %%

def step():
    un = u.copy() ##copy the existing values of u into un
    for i in range(1,nx-1):
        u[i] = un[i] + nu*dt/dx**2*(un[i+1]-2*un[i]+un[i-1])

def init():
    line.set_data([], [])
    return line,

def animate(i):
    step()
    line.set_data(np.linspace(0,xmax,nx),u)
    return line,


# %%

xmax  = 2.0
nx    = 41
dx    = xmax/(nx-1)
nt    = 20             # the number of timesteps we want to calculate
nu    = 0.3            # the value of viscosity
sigma = .2             # sigma is a parameter, we'll learn more about it later
dt    = sigma*dx**2/nu # dt is defined using sigma ... more later!


u = np.ones(nx)             # a numpy array with nx elements all equal to 1.
u[.75/dx : 1.25/dx+1] = 10  # setting u = 2 between 0.5 and 1 as per our I.C.s

un = np.ones(nx) # our placeholder array, un, to advance the solution in time

# %%

fig = plt.figure()
ax = fig.add_subplot(111,xlim=(0,xmax), ylim=(0,10))
ax.grid()
line, = ax.plot([], [], lw=3)


# %%

ani = animation.FuncAnimation(fig, animate, frames=1, interval=100, blit=True, init_func=init)

plt.show()