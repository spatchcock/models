# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 21:30:09 2014

@author: spatchcock
"""

import numpy as np                 
import matplotlib.pyplot as plt
import time, sys
import matplotlib.animation as animation

# This is patt 1 of Lorena Barba's "12 steps to Navier-Stokes" series,
#
#  http://nbviewer.ipython.org/github/barbagroup/CFDPython/blob/master/lessons/01_Step_1.ipynb
#
# It describes simple, linear advection in 1 dimension. The advection equation is:
#
#   du/dt + c . du/dx = 0
#
# or, alternatively:
#
#   du/dt = - c . du/dx
#
# where,
#  u is some property which varies with space (1D) and time
#  x is the space dimension
#  t is time
#  c is the advection speed 
#
# Points to note
# --------------
#
# 1. Advection is dependent on the scalar, constant advection speed c. This makes the
#    model linear, as opposed to the non-linear variant which advects at speed u (i.e.
#    the property u is velocity). 
# 
# 2. Numerical scheme. This numerical solution uses the forward-difference scheme for
#    the time derivative and the backward difference scheme for the space derivative.
#    This makes it accurate to the order of magnitude of dx - the step size of the 
#    space dimension, i.e. O(dx).
# 
# 3. Numerical diffusion. 
#    Depending on the time and space steps, the original spatial state becomes stretched
#    or squashed. This is because on discretization, the du/dx term gets multiplied
#    by c.dt/du which is dimensionless. This effectively becomes the scaling factor
#    and whether or not it is grater than or less than unity depends on all three
#    values.
#
# 3. Cycling.
#    The iterator starts from the first cell. This means that the backward differnce
#    calculation uses a negative index (-1). Conveniently (?) this translates to the
#    syntax for the last element in the u array and therefore has the effect of cycling
#    the process back to the start. Any stretching or squashing then becomes compounding,
#    meaning that the signal becomes dampened or amplified.
#

# %%

def step():
    un = u.copy()       
  
    for i in range(nx):
    #for i in range(1,nx):
        u[i] = un[i]-c*dt/dx*(un[i]-un[i-1])


# %%

def init():
    line.set_data([], [])
    return line,

def animate(i):
    step()
    line.set_data(np.linspace(0,xmax,nx),u)
    return line,

# %%

xmax = 2.   # domain length
nx = 75     # try changing this number from 41 to 81 and Run All ... what happens?
dx = xmax/(nx-1)
nt = 1      # nt is the number of timesteps we want to calculate
dt = .025   # dt is the amount of time each timestep covers (delta t)
c = 1.      # assume wavespeed of c = 1

u  = np.ones(nx)  # numpy function ones()
un = np.ones(nx)  # initialize a temporary array for using in each timestep

# Set initial conditions
u[.5/dx : 1/dx+1] = 2  # setting u = 2 between 0.5 and 1 as per our I.C.s


# %%

fig = plt.figure()
ax = fig.add_subplot(111,xlim=(0,xmax), ylim=(0,10))
ax.grid()
line, = ax.plot([], [], lw=3)

# %%

ani = animation.FuncAnimation(fig, animate, frames=1, interval=100, blit=True, init_func=init)

plt.show()