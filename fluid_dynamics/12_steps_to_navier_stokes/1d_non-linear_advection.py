# -*- coding: utf-8 -*-
"""
Created on Fri Mar 06 10:57:12 2015

@author: andrew.berkeley
"""

import numpy as np                 
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# This is part 2 of Lorena Barba's "12 steps to Navier-Stokes" series,
#
#   http://nbviewer.ipython.org/github/barbagroup/CFDPython/blob/master/lessons/02_Step_2.ipynb
#
# This model described here is identical to the one descibed in Step 1 except that 
# the advection speed c is replaced by u. This makes the model non-linear as u is
# multiplied by its own derivative.
#
#   du/dt + u . du/dx = 0
#
# or, alternatively:
#
#   du/dt = - u . du/dx
#
# where,
#  u is some property which varies with space (1D) and time
#  x is the space dimension
#  t is time
#
# Points to note
# --------------
#
# 1. Since the advection speed is the same variable as the property of interest,
#    this basically means that velocity is the property of interest! The model
#    describes the propagation of momentum.
#
# 2. This model becomes unstable much more easily that the linear model. It becomes
#    unstable using the same parameters which provide stability in the linear model.
#    Increasing the time resolution or decreasing the space resolution increases 
#    stability. This is because the CFL condition is altered by the varying values
#    of u, and that these values are greater than 1, which was the advection speed 
#    in the linear model. So the ratio of time step and space step needs to be 
#    different in this case to handle the larger advection speeds.
#

# %%

def step():
    un = u.copy()
  
    for i in range(nx):
    #for i in range(1,nx):
        u[i] = un[i]-un[i]*dt/dx*(un[i]-un[i-1])


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
nx = 21     # try changing this number from 41 to 81 and Run All ... what happens?
dx = xmax/(nx-1)
nt = 20      # nt is the number of timesteps we want to calculate
dt = .025   # dt is the amount of time each timestep covers (delta t)

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