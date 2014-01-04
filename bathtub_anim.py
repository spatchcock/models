# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib.animation as animation

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
# parameter (i.e. the reciprocal of friction coefficient) and is scaled by the
# water height.
#
# The animation pattern is based on the animation tutorial found here: 
#  http://jakevdp.github.io/blog/2012/08/18/matplotlib-animation-tutorial/
#

class Bathtub:
    
    # This class is just a container for the model
    
    def __init__(self, N, dx, dt, tau, initial_surface, h):
        
        self.N  = N     # cell number
        self.dx = dx    # distance step, i.e. distance between cells
        self.dt = dt    # time step
        
        self.h   = h    # height to datum (not the same as 'height' in the model 
                        # iteration which represents the total height of the 
                        # water surface)
        
        self.tau = tau  # dampening time (reciprocal of coefficient of friction)
        
        # Initialize containers for local (cell) current velocities (u), water 
        # heights and water slopes (px)
        self.u      = [0 for i in range(self.N+1)]
        self.height = [0 for i in range(self.N+1)]
        self.px     = [0 for i in range(self.N)]

        # Initial water elevation relative to datum
        self.eta = initial_surface 
        
    # Iterate one timestep
    def step(self):
        
        # Update local water heights based on last elevations
        for i in range(1,self.N):
            self.height[i] = self.h + self.eta[i]
            
        # Handle boundary water heights
        self.height[N] = self.height[self.N-1]
        self.height[0] = self.h
        
        # Calculate local water slopes
        for i in range(1,self.N):
            self.px[i] = (self.eta[i] - self.eta[i-1])/self.dx
        
        # Calculate local current velocities
        for i in range(1,self.N):
            self.u[i] = self.u[i] + self.dt * (-self.px[i] - self.u[i]/(self.tau * self.height[i]))

        # Calculate new local elevations
        for i in range(0,self.N):
            self.eta[i] = self.eta[i] - self.dt * ((self.height[i+1] * self.u[i+1] - self.height[i]*self.u[i])/self.dx)
            
            if self.eta[i] < -1:
                self.eta[i] = -1
    

#------------------------------------------------------------

# set up initial state and global variables

N   = 50     
dx  = 1.0/N  
dt  = 0.1*dx 
tau = 0.1    
h   = 10 

#------------------------------------------------------------

# Set up initial water elevations as simple step function
initial_surface = [0 for i in range(N)]

### Step function ###
  
#for i in range(0,N):
#    if i < N/2:
#        initial_surface[i] = 0.05
#    else:
#        initial_surface[i] = -0.05

### Smooth gradient ##

for i in range(0,N):
    initial_surface[i] = 2.0 * (N/2 - i) / (N/2)

#------------------------------------------------------------

# Initialize model object

bathtub = Bathtub(N, dx, dt, tau, initial_surface, h)

#------------------------------------------------------------

# set up figure

fig = plt.figure()

# Limit x-axis to hide stationary points
ax = fig.add_subplot(111, xlim=(1, N), ylim=(0, h*1.5))
ax.grid()

line, = ax.plot([], [], lw=3)
step_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

#------------------------------------------------------------

# Set up and run animation

# Clear frame on each interation
def init():
    line.set_data([], [])
    return line,

# Invoke model timestep and replot data on each iteration
def animate(i):
    global bathtub       
    bathtub.step()
    
    line.set_data(range(N+1), bathtub.height)
    step_text.set_text('iter: %.1f' % i)
    return line, step_text


ani = animation.FuncAnimation(fig, animate, frames=10000,
                              interval=1, blit=True, init_func=init)

plt.show()