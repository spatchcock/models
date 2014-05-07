# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 22:52:11 2014

@author: spatchcock
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Advection - diffusion - Decay - production
#
# Differential equation
#
#  dC/dt = D(d^2C/dx^2) - w(dC/dx) - uC + Ra(x)
# 
# Difference equation
#
#  (C{x,t} = C{x,t-1} + dt * [D*(C{x+1,t-1} - 2C{x, t-1} + C{x-1,t-1})/dx^2 - w(C{x+1,t-1} - C{x-1,t-1})/2dx - u*C{x,t-1} + Ra{x}]
#
# Initial conditions
#
#  C(x,0) = R{x}
#

###############################################################################

# Numerical scheme

N_x = 101
max_depth = 30.0
dx = max_depth/N_x
sigma = 0.1
dt = sigma*dx


###############################################################################

# Dependent and independent variables (C, x)
x  = np.linspace(0.0,max_depth,N_x) # depth
C  = np.zeros(N_x)                  # concentration
    
###############################################################################

# Set up parameters

# Each parametercan,in principle vary withdepth, x. Initialise arrarys for each
# although we can set a constant valueif required.
Ra = np.zeros(N_x)                  # production
D  = np.zeros(N_x)                  # diffusion (mixing rate)
u  = np.zeros(N_x)                  # taphonomic decay rate
w  = np.zeros(N_x)                  # advection speed (sedimentation rate)
Cu = np.zeros(N_x)                  # memoize last timestep concentration

# Taponomic decay, constant with depth
u[:] = 0.005                       

# Sedimentation rate, constant with depth
w[:] = 0.6                         

# Mixing/diffusion, constant in upper mixed zone, zero below
max_depth_mixing  = 10.0
max_x_mixing      = int(max_depth_mixing/max_depth*N_x)
D[0:max_x_mixing] = 0.2399          

# Production, three options, comment as required
Ra_0 = 50.0
Ra_attenuation = 0.5
Ra_peak_depth  = 5
Ra_gamma       = 4
max_x_Ra       = int(Ra_peak_depth/max_depth*N_x)

Ra[0:max_x_Ra] = Ra_0                                                   # constant over interval
#Ra[0:] = Ra_0 * np.exp(-Ra_attenuation*x[0:])                           # exponential decrease
#Ra[0:] = Ra_0 * np.exp(-Ra_attenuation*(x[0:]-Ra_peak_depth)**Ra_gamma) # subsurface peak, normally distributed

# Initial conditions
#C = Ra * dt

###############################################################################

# Implement discretized equations as invokable timestep function

def step():
    # memoize last timestep
    Cu[:]   = C[:]
    
    # boundary, surficial layer(x=0)
    C[0] = dt * Ra[0]
    
    # Interior points  
    C[1:-1] = Cu[1:-1] + dt * (D[1:-1]*(Cu[2:] - 2.0*Cu[1:-1] + Cu[0:-2])/dx**2.0 - w[1:-1]*(Cu[2:] - Cu[0:-2])/2.0*dx - u[1:-1]*Cu[1:-1] + Ra[1:-1])
    
    # boundary, bottomost layer (x=max_depth)
    C[-1]   = C[-2]     


###############################################################################

# Set up plots

fig   = plt.figure()

Ra_plot  = fig.add_subplot(151, ylim=(max_depth, 0), xlim=(0, max(Ra)*1.5))
Ra_line, = Ra_plot.plot([], [], lw=3)
Ra_plot.grid()
Ra_plot.axes.get_xaxis().set_ticks([0.0, max(Ra)])
Ra_plot.set_xlabel('Ra')

D_plot  = fig.add_subplot(152, ylim=(max_depth, 0), xlim=(0, max(D)*1.5))
D_line, = D_plot.plot([], [], lw=3)
D_plot.grid()
D_plot.axes.get_yaxis().set_ticklabels([])
D_plot.axes.get_xaxis().set_ticks([0.0, max(D)])
D_plot.set_xlabel('D')

w_plot  = fig.add_subplot(153, ylim=(max_depth, 0), xlim=(0, max(w)*1.5))
w_line, = w_plot.plot([], [], lw=3)
w_plot.grid()
w_plot.axes.get_yaxis().set_ticklabels([])
w_plot.axes.get_xaxis().set_ticks([0.0, max(w)])
w_plot.set_xlabel('w')

u_plot  = fig.add_subplot(154, ylim=(max_depth, 0), xlim=(0, max(u)*1.5))
u_line, = u_plot.plot([], [], lw=3)
u_plot.grid()
u_plot.axes.get_yaxis().set_ticklabels([])
u_plot.axes.get_xaxis().set_ticks([0.0, max(u)])
u_plot.set_xlabel('u')

C_plot  = fig.add_subplot(155, ylim=(max_depth, 0), xlim=(0, 1000))
C_line, = C_plot.plot([], [], lw=3)
step_text = C_plot.text(0.2, 0.02, '', transform=C_plot.transAxes)
C_plot.grid()
C_plot.axes.get_yaxis().set_ticklabels([])
C_plot.set_xlabel('C')

plt.subplots_adjust(wspace=0.1)

###############################################################################

# Set up animation

# Clear frame on each interation
def init():
    # Reset each line
    Ra_line.set_data([], [])
    D_line.set_data([], [])
    w_line.set_data([], [])
    u_line.set_data([], [])
    C_line.set_data([], [])
    
    return Ra_line,D_line,w_line,u_line,C_line, 


# Invoke model timestep and replot data on each iteration
def animate(i):
    # Iterate model
    step()
    
    # Update each line
    Ra_line.set_data(Ra, x)
    D_line.set_data(D, x)
    w_line.set_data(w, x)
    u_line.set_data(u, x)
    C_line.set_data(C, x)

    step_text.set_text('iter: %.1f' % i)

    return Ra_line,D_line,w_line,u_line,C_line,step_text


###############################################################################

# Run animation
ani = animation.FuncAnimation(fig, animate, frames=10000000, interval=1, blit=True, init_func=init)



    