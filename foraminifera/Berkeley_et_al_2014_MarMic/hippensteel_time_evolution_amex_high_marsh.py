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

# %%

high_file = './data/hippensteel_total_high_marsh.csv'

dataset = np.genfromtxt(high_file, dtype=float, delimiter=',', names=True) 

species = 'AMEX'
depth_data = dataset['Depth']
dead_data  = dataset[species+'_Dead']/10.0
live_data  = dataset[species+'_Live']/10.0 

# High marsh
mixing_depth = 5.0
mixing_rate  = 0.24
sed_rate     = 0.62

#%%

# Numerical scheme

N_x = 201
max_depth = 60.0
dx = max_depth/N_x
sigma = 0.1
dt = sigma*dx


# %%

# Dependent and independent variables (C, x)
x  = np.linspace(0.0,max_depth,N_x) # depth
C  = np.zeros(N_x)                  # concentration
    
# %%
    
# Set up parameters

# Each parameter can,in principle, vary with depth, x. Initialise arrays for each
# although we can set a constant valueif required.

Ra = np.zeros(N_x)                  # production
D  = np.zeros(N_x)                  # diffusion (mixing rate)
u  = np.zeros(N_x)                  # taphonomic decay rate
w  = np.zeros(N_x)                  # advection speed (sedimentation rate)
Cu = np.zeros(N_x)                  # memoize last timestep concentration

# Sedimentation rate, constant with depth
w[:] = sed_rate  

# %%

# Mixing/diffusion, constant in upper mixed zone, zero below
max_depth_mixing  = mixing_depth
max_x_mixing      = int(max_depth_mixing/max_depth*N_x)
D[0:max_x_mixing] = mixing_rate          

# Taponomic decay, constant with depth
max_depth_taz = max_depth_mixing
max_x_taz = int(max_depth_taz/max_depth*N_x)
u_0 = 0.21
u[0:max_x_taz] = u_0
u[max_x_taz:] = u_0*np.exp(-0.128*x[:-max_x_taz])

# Production
a,b=np.polynomial.polynomial.polyfit(depth_data,live_data,5,full=True)
x=np.linspace(0,60,N_x)
a=a[0] + a[1]*x + a[2]*x**2 + a[3]*x**3 + a[4]*x**4 + a[5]*x**5
a[Ra<0] = 0

R = np.zeros(N_x)
R_0 = 1.0
R = R_0*np.exp(-0.135*x)
Ra = R*a

# %%

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

a_plot  = fig.add_subplot(151, ylim=(max_depth, 0), xlim=(0, 3))
a_line, = a_plot.plot([], [], linewidth=1.5,color='k')
a_plot.grid()
a_plot.axes.get_xaxis().set_ticks([0.0, 1.0, 2.0, 3.0])
a_plot.set_xlabel('a')
a_plot.plot(live_data, depth_data, marker='o', linestyle='None',color='k')

D_plot  = fig.add_subplot(153, ylim=(max_depth, 0), xlim=(0, 0.3))
D_line, = D_plot.plot([], [], linewidth=1.5,color='k')
D_plot.grid()
D_plot.axes.get_yaxis().set_ticklabels([])
D_plot.axes.get_xaxis().set_ticks([0.0, 0.1, 0.2, 0.3])
D_plot.set_xlabel('D')

R_plot  = fig.add_subplot(152, ylim=(max_depth, 0), xlim=(0, 1.5))
R_line, = R_plot.plot([], [], linewidth=1.5,color='k')
R_plot.grid()
R_plot.axes.get_yaxis().set_ticklabels([])
R_plot.axes.get_xaxis().set_ticks([0,.5,1.0,1.5])
R_plot.set_xlabel('R')

u_plot  = fig.add_subplot(154, ylim=(max_depth, 0), xlim=(0, 0.6))
u_line, = u_plot.plot([], [], linewidth=1.5,color='k')
u_plot.grid()
u_plot.axes.get_yaxis().set_ticklabels([])
u_plot.axes.get_xaxis().set_ticks([0.0, 0.2,0.4,0.6])
u_plot.set_xlabel('u')

C_plot  = fig.add_subplot(155, ylim=(max_depth, 0), xlim=(0, 8))
C_line, = C_plot.plot([], [], linewidth=1.5,color='k')
#step_text = C_plot.text(0.2, 0.02, '', transform=C_plot.transAxes)
C_plot.grid()
C_plot.axes.get_yaxis().set_ticklabels([])
C_plot.axes.get_xaxis().set_ticks([0,2,4,6,8])
C_plot.set_xlabel('C')
C_plot.plot(dead_data,depth_data, marker='o', linestyle='None',color='k')

plt.subplots_adjust(wspace=0.3)

###############################################################################

# Set up animation

# Clear frame on each interation
def init():
    # Reset each line
    a_line.set_data([], [])
    D_line.set_data([], [])
    R_line.set_data([], [])
    u_line.set_data([], [])
    C_line.set_data([], [])
    
    return a_line,D_line,R_line,u_line,C_line, 


# Invoke model timestep and replot data on each iteration
def animate(i):
    # Iterate model
    step()
    
    # Update each line
    a_line.set_data(a, x)
    D_line.set_data(D, x)
    R_line.set_data(R, x)
    u_line.set_data(u, x)
    C_line.set_data(C, x)

#    step_text.set_text('iter: %.1f' % i)

    return a_line,D_line,R_line,u_line,C_line#,step_text


###############################################################################

# Run animation
ani = animation.FuncAnimation(fig, animate, frames=50000, interval=1, blit=True, init_func=init, repeat=False)

plt.show()


    