# -*- coding: utf-8 -*-
"""
Created on Fri Mar 06 23:49:57 2015

@author: andrew.berkeley
"""

import numpy as np                 #loading our favorite library
import matplotlib.pyplot as plt    #and the useful plotting library


# second order

# central difference

# dt defined using viscosity (analogous to speed) and dx**2
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

xmax = 2.0
nx = 41
dx = xmax/(nx-1)
nt = 20    #the number of timesteps we want to calculate
nu = 0.3   #the value of viscosity
sigma = .2 #sigma is a parameter, we'll learn more about it later
dt = sigma*dx**2/nu #dt is defined using sigma ... more later!


u = np.ones(nx)      #a numpy array with nx elements all equal to 1.
u[.75/dx : 1.25/dx+1]=10  #setting u = 2 between 0.5 and 1 as per our I.C.s

un = np.ones(nx) #our placeholder array, un, to advance the solution in time

# %%

fig = plt.figure()
ax = fig.add_subplot(111,xlim=(0,xmax), ylim=(0,10))
ax.grid()
line, = ax.plot([], [], lw=3)


# %%

ani = animation.FuncAnimation(fig, animate, frames=1, interval=100, blit=True, init_func=init)

plt.show()