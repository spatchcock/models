# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# Implementation of the "bathtub" model describing the dynamics of a water
# surface following an initial perturbation. The model is a 1-dimensional, 
# depth-averaged model based on the depth-averaged volume continuity and
# momentum equations:
#
#   dn/dt = -H.(du/dx)
#
#   du/dt = -dn/dx - u/T*H
# 
# where,
#  n is the water level above some datum (m)
#  H is the total water depth (m)
#  u is the current velocity (m3 m-2 s-1 or m s-1)
#  T is the frictional damping time
#
# In this implementation, friction is represented by a linear 'dampening time'
# parameter (i.e. the reciprocal of friction coefficient). The H quotient in the 
# friction terms effectively represents the averaging of all frictional effects 
# over the entire water column, in effect, the net frictional force of the air-water
# interface and the sea bed.
#
# The model is based on the similar implementation described in The Dynamics of 
# Coastal Models by Clifford J. Hearn (Cambridge).
#
# The animation pattern is based on the animation tutorial found here: 
#  http://jakevdp.github.io/blog/2012/08/18/matplotlib-animation-tutorial/
#


# %% Set up initial state and global variables

N   = 30      # cell number
dx  = 1.0/N   # distance step, i.e. distance between cells
dt  = 0.1*dx  # time step
tau = 0.1     # dampening time (reciprocal of coefficient of friction)
h   = 10      # height to datum (not the same as 'height' in the model 
              # iteration which represents the total height of the 
              # water surface)


# %% Set up initial water elevations. Several choices, (un)comment as appropriate 

eta = np.zeros(N)

# Step function 

#for i in range(0,N): 
#    if i < N/2:
#        eta[i] = 0.5
#    else:
#        eta[i] = -0.5


# Smooth gradient

for i in range(0,N): 
    eta[i] = 2.0 * (N/2 - i) / (N/2)

# Top hat

#eta[np.ceil(N/4):np.ceil(3*N/4)] = 1.0 


# %% Initialize containers for local (cell) current velocities (u), water 
# heights and water slopes (px)

u      = np.zeros(N+1)
height = np.zeros(N+1)
px     = np.zeros(N)


# %% Set up iteration-of-timestep function. We do this, rather than a loop, because
# the animation pattern requires a callable function to drive each timestep

# Iterate one timestep
def step():
    
    # Update local water heights based on last elevations
    for i in range(1,N):
        height[i] = h + eta[i]
        
    # Handle boundary water heights
    height[N] = height[N-1]
    height[0] = h
    
    # Calculate local water slopes (backward differences)
    for i in range(1,N):
        px[i] = (eta[i] - eta[i-1])/dx
    
    # Calculate local current velocities
    for i in range(1,N):
        u[i] = u[i] + dt * (-px[i] - u[i]/(tau * height[i]))

    # Calculate new local elevations (forward differences)
    for i in range(0,N):
        eta[i] = eta[i] - dt * ((height[i+1] * u[i+1] - height[i]*u[i])/dx)
        
        if eta[i] < -1:
            eta[i] = -1
    

# %% Set up figure

fig = plt.figure()

# Limit x-axis to hide stationary points
ax = fig.add_subplot(111, xlim=(1, N), ylim=(0, h*1.5))
ax.grid()

line, = ax.plot([], [], lw=3)
step_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)


# %% Set up animation functions

# Clear frame on each interation
def init():
    line.set_data([], [])
    return line,

# Invoke model timestep and replot data on each iteration
def animate(i):
    step()
    
    line.set_data(range(N+1), height)
    step_text.set_text('iter: %.1f' % i)
    return line, step_text


# %% Run!

ani = animation.FuncAnimation(fig, 
                              animate, 
                              frames=10000,
                              interval=10, 
                              blit=True, 
                              init_func=init)

plt.show()