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

# %% DEFINE NUMERICAL SCHEME

max_depth = 30.0           # maximum depth of domain of interest
N_x       = 101            # number of nodes across 1D domain
dx        = max_depth/N_x  # cell size (regular/uniform)
sigma     = 0.1            # CFL sigma value. Scales the timestep according to the depth step. 
                           # Ensures timestep is sufficiently smaller that distance step to provide
                           # stability (although this also depends on the sedimentation rate)
dt        = sigma*dx       # time step


# %% SET UP PLACEHOLDER ARRAYS FOR PARAMETERS AND VARIABLES

# Each parameter and variable will be represented by a value at each node across
# the 1D domain.

# Dependent and independent variables (C, x)
x  = np.linspace(0.0,max_depth,N_x) # depth
C  = np.zeros(N_x)                  # concentration

# Parameters -  Each parameter can, in principle vary with depth, x. Initialise arrays 
# for each, although we can set a constant value for all x if required.
Ra = np.zeros(N_x)     # production (product of standing crop, a, and reproduction rate, R)
D  = np.zeros(N_x)     # diffusion (mixing rate)
u  = np.zeros(N_x)     # taphonomic decay rate
w  = np.zeros(N_x)     # advection speed (sedimentation rate)
Cu = np.zeros(N_x)     # placeholder for memoizing previous timestep concentrations


# %% DEFINE DEPTH-DEPENDENT FUNCTION FOR TAPHONOMIC DECAY

# It is likely that taphonomic decay decreases with depth so most circumstances probably
# require a function for the taphonomic decay rate that decrease through the domain. In
# some circumstances, considering decay rates to be constant across some or all of the domain
# might be appropriate. Three choices are presented below. Comment/uncomment as required
# or set u[] to another appropriate function of depth.

### Constant function ###

# This simply sets the same decay rate for all values of x.

#u[:] = 0.005


### Decreasing function ###

# This drescribes taphonic decay rate as decreasing exponential with depth frmom
# some maximum value at the surface. This is the simplest decreasing function that
# asymptotes with depth.

u_0           = 0.005                              # value at surface, i.e. x = 0
u_attenuation = 0.05                               # rate at which decay rate decreases with depth
u[0:]         = u_0 * np.exp(-u_attenuation*x[0:]) # exponentially decreasing taphonomic decay rate


### Step function ###

# This sets the decay rate as a constant across some limited upper interval of the
# sediment. This resembles the commonly invoked concept of the Taphonomically Active Zone
# (the "TAZ"). Of course, any other more complicated step function could be defined in a 
# similar way.

#max_depth_decay  = 10.0                               # Maximum depth of decay
#max_x_decay      = int(max_depth_decay/max_depth*N_x) # Index of maximum decay depth
#u[0:max_x_decay] = 0.005                              # Step function


# %% DEFINE DEPTH DEPENDENT FUNCTION FOR SEDIMENTATION RATE

# In principle, sedimentation rate may have varied during the time in which a given
# sediment interval has accumulated. For now, we'll just assume that it is constant.

### Constant function ##
w[:] = 0.6                         


# %% DEFINE DEPTH DEPENDENT FUNCTION FOR MIXING/BIOTURBATION

# constant in upper mixed zone, zero below
max_depth_mixing  = 15.0
max_x_mixing      = int(max_depth_mixing/max_depth*N_x)
D[0:max_x_mixing] = 0.2399          


# %% DEFINE DEPTH-DEPENDENT FUNCTION FOR TEST PRODUCTION

Ra_0 = 30.0
Ra_attenuation = 0.05
Ra_peak_depth  = 2
Ra_gamma       = 4
max_x_Ra       = int(Ra_peak_depth/max_depth*N_x)

#Ra[0:max_x_Ra] = Ra_0                                                   # constant over interval
#Ra[0:] = Ra_0 * np.exp(-Ra_attenuation*x[0:])                           # exponential decrease
Ra[0:] = Ra_0 * np.exp(-Ra_attenuation*(x[0:]-Ra_peak_depth)**Ra_gamma) # subsurface peak, normally distributed


# %% IMPLEMENT DISCRETIZED EQUATION AS INVOKABLE TIMESTEP FUNCTION

def step():
    # memoize last timestep
    Cu[:]   = C[:]
    
    # boundary, surficial layer (x=0)
    C[0] = dt * Ra[0]
    
    # Interior points  
    C[1:-1] = Cu[1:-1] + dt * (D[1:-1]*(Cu[2:] - 2.0*Cu[1:-1] + Cu[0:-2])/dx**2.0 - w[1:-1]*(Cu[2:] - Cu[0:-2])/2.0*dx - u[1:-1]*Cu[1:-1] + Ra[1:-1])
    
    # boundary, bottomost layer (x=max_depth)
    C[-1]   = C[-2]     


# %% SET UP PLOTS

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

# %% SET ANIMATION

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


# %% RUN ANIMATION
ani = animation.FuncAnimation(fig, animate, frames=10000000, interval=1, blit=True, init_func=init)



    