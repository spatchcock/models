# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 21:30:09 2014

@author: spatchcock
"""

import numpy as np                 
import matplotlib.pyplot as plt
import time, sys
import matplotlib.animation as animation

# Advection through time 1-D spatial state. Dependent on advection speed c.
# Depending on the time and space steps, the original spatial state becomes stretched
# or squashed. This is because on discretization, the du/dx term gets multiplied
# by c.dt/du which is dimensionless. This effectively becomes the scaling factor
# and whether or not it is grater than or less than unity depends on all three
# values.
#
# The iterator starts from the first cell. This means that the backward differnce
# calculation uses a negative index (-1). Conveniently (?) this translates to the
# syntax for the last element in the u array and therefore has the effect of cycling
# the process back to the start. Any stretching or squashing then becomes compounding,
# meaning that the signal becomes dampened or amplified.

xmax = 2.   # domain length
nx = 71     # try changing this number from 41 to 81 and Run All ... what happens?
dx = xmax/(nx-1)
nt = 1      # nt is the number of timesteps we want to calculate
dt = .025   # dt is the amount of time each timestep covers (delta t)
c = 1.      # assume wavespeed of c = 1

u = np.ones(nx)        # numpy function ones()
u[.5/dx : 1/dx+1] = 2  # setting u = 2 between 0.5 and 1 as per our I.C.s

# initialize a temporary array
un = np.ones(nx) 

def step():
    un = u.copy()       
  
    for i in range(nx):
        u[i] = un[i]-c*dt/dx*(un[i]-un[i-1])

#------------------------------------------------------------

fig = plt.figure()
ax = fig.add_subplot(111,xlim=(0,xmax), ylim=(0,2))
ax.grid()
line, = ax.plot([], [], lw=3)

#------------------------------------------------------------

def init():
    line.set_data([], [])
    return line,

def animate(i):
    step()
    line.set_data(np.linspace(0,xmax,nx),u)
    return line,


ani = animation.FuncAnimation(fig, animate, frames=1, interval=10, blit=True, init_func=init)

plt.show()