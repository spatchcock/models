# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 21:57:49 2014

@author: spatchcock
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 22:52:11 2014

@author: spatchcock
"""

import numpy as np
import matplotlib.pyplot as plt
from sympy.functions import coth
import matplotlib.animation as animation

# Advection - diffusion - Decay - production
#
# Differential equation
#
#  dC/dt = D(d^2C/dx^2) - w(dC/dx) - uC + R
# 
# Difference equation derivation
#
#  (C{x,t} - C{x,t-1}) / dt = D*(C{x+1,t-1} - 2C{x,t-1} + C{x-1,t-1})/dx^2 - w(C{x+1,t-1} - C{x-1,t-1})/2dx - u*C{x,t-1} + R{x}
#  (C{x,t} = C{x,t-1} + dt * [D*(C{x+1,t-1} - 2C{x, t-1} + C{x-1,t-1})/dx^2 - w(C{x+1,t-1} - C{x-1,t-1})/2dx - u*C{x,t-1} + R{x}]
#
# Initial conditions
#
#  C(x,0) = R{x}
# 
#
# Bioturbation coefficients (D_b ; cm^2/yr) for total tracers; ranges are given in parentheses (from Kohl and Martin, 1999)
#                       February 1998       June 1998           July 1998           September 1998
# High marsh            0.06 (0.06–0.08)    0.08 (0.08–0.09)    0.38 (0.06–0.69)    0.44 (0.07–0.54)        
# Intermediate marsh    0.41 (0.35–0.41)    0.41 (0.38–0.57)    0.44 (0.13–0.50)    0.41 (0.17–0.66)        
# Low marsh             1.20 (1.10–1.55)    0.66 (0.60–0.69)    1.29 (1.10–2.18)    0.47 (0.17–3.15)
#
# Mean
#  High 0.2399
#  Int  0.417499
#  Low  0.905
#
#In fact, the proposed 0–5 cm analogue encompasses between *12 years based on average 
#sedimentation rates in the area (*0.4 cm year -1 ; Leorri et al., 2006) and *8 years 
#based on burial rates derived from artificial bead layers for the high and intermediate 
#marsh plots (*0.6 cm year -1 ; Leorri et al., 2008b).

#def step():
#    Cu[:]   = C[:]      # memoize last timestep
#    C[0]    = dt * R[0] # boundary, surficial layer
#    C[1:-1] = Cu[1:-1] + dt * (D[1:-1]*(Cu[2:] - 2.0*Cu[1:-1] + Cu[0:-2])/dx**2.0 - w[1:-1]*(Cu[2:] - Cu[0:-2])/2.0*dx - u[1:-1]*Cu[1:-1] + R[1:-1])
#    C[-1]   = C[-2]     # boundary, bottomost layer
#
#
#def step_and_plot():
#    step()
#    
#    plt.plot(x,C)
#    plt.show()
#
#
## Numerics
#max_depth = 30.0
#def step():
#    Cu[:]   = C[:]      # memoize last timestep
#    C[0]    = dt * R[0] # boundary, surficial layer
#    C[1:-1] = Cu[1:-1] + dt * (D[1:-1]*(Cu[2:] - 2.0*Cu[1:-1] + Cu[0:-2])/dx**2.0 - w[1:-1]*(Cu[2:] - Cu[0:-2])/2.0*dx - u[1:-1]*Cu[1:-1] + R[1:-1])
#    C[-1]   = C[-2]     # boundary, bottomost layer
#
#
#def step_and_plot():
#    step()
#    
#    plt.plot(x,C)
#    plt.show()
#
#
## Numerics
max_depth = 30.0
N_x = 501
dx = max_depth/N_x
sigma = 0.1
dt = sigma*dx 

# Scaling factor for advection term (see Boudreau)
#  cosh/sinh is the coth, the hyperbolic co-tangent
#peclet_number = ((w*dx)/(2*D))
#sigma = np.cosh(peclet_number)/np.sinh(peclet_number) - 1.0/peclet_number

# Initialize
x = np.linspace(0.0,max_depth,N_x) # depth
C = np.zeros(N_x)                  # concentration
R = np.zeros(N_x)                  # production
D = np.zeros(N_x)                  # diffusion (mixing rate)
u = np.zeros(N_x)                  # decay rate
w = np.zeros(N_x)                  # advection speed (sedimentation rate)

Cu = np.zeros(N_x)                  # memoize last timestep concentration

# Parameters
u[:] = 0.05                      # constant with depth
w[:] = 0.6                       # constant with depth

max_depth_mixing = 10.0
max_x_mixing = int(max_depth_mixing/max_depth*N_x)

D[0:max_x_mixing] = 0.2399          # constant in upper mixed zone, zero below

R_0 = 20.0
R_attenuation = 0.5
R_peak_depth  = 5
R_gamma       = 4
max_x_R       = int(R_peak_depth/max_depth*N_x)

#R[0:max_x_R] = R_0                                                 # constant over interval
R[0:] = R_0 * np.exp(-R_attenuation*x[0:])                         # exponential decrease
#R[0:] = R_0 * np.exp(-R_attenuation*(x[0:]-R_peak_depth)**R_gamma) # subsurface peak, normally distributed

# initial conditions
C[0] = 100
C[1] = 100

for i in range(2, size(x)-2):
    if D[i] == 0:
        C[i] = C[i-1] + dx*R[i]/w[i] - dx*u[i]*C[i-1]/w[i]
    else:
        C[i] =(-2.0*C[i-1] + C[i-2] + (w[i]*dx)/(2.0*D[i])*C[i-2] - (dx**2) * u[i]*C[i-1]/D[i] + (dx**2) * R[i]/D[i] ) / (((w[i]*dx)/(2.0*D[i])) - 1)

#        C[i] = (D[i]*C[i-2] - 2.0*D[i]*C[i-1] + w[i]*dx*C[i-1] - (dx**2.0)*u[i]*C[i-1] + (dx**2)*R[i])/(w[i]*dx - D[i])
    


plt.plot(x,C)

#
#N_x = 101
#dx = max_depth/N_x
#sigma = 0.1
#dt = sigma*dx 
#
## Scaling factor for advection term (see Boudreau)
##  cosh/sinh is the coth, the hyperbolic co-tangent
##peclet_number = ((w*dx)/(2*D))
##sigma = np.cosh(peclet_number)/np.sinh(peclet_number) - 1.0/peclet_number
#
## Initialize
#x = np.linspace(0.0,max_depth,N_x) # depth
#C = np.zeros(N_x)                  # concentration
#R = np.zeros(N_x)                  # production
#D = np.zeros(N_x)                  # diffusion (mixing rate)
#u = np.zeros(N_x)                  # decay rate
#w = np.zeros(N_x)                  # advection speed (sedimentation rate)
#
#Cu = np.zeros(N_x)                  # memoize last timestep concentration
#
## Parameters
#u[:] = 0.05                      # constant with depth
#w[:] = 0.6                       # constant with depth
#
#max_depth_mixing = 10.0
#max_x_mixing = int(max_depth_mixing/max_depth*N_x)
#
#D[0:max_x_mixing] = 0.2399          # constant in upper mixed zone, zero below
#
#R_0 = 20.0
#R_attenuation = 0.5
#R_peak_depth  = 5
#R_gamma       = 4
#max_x_R       = int(R_peak_depth/max_depth*N_x)
#
#R[0:max_x_R] = R_0                                                 # constant over interval
##R[0:] = R_0 * np.exp(-R_attenuation*x[0:])                         # exponential decrease
##R[0:] = R_0 * np.exp(-R_attenuation*(x[0:]-R_peak_depth)**R_gamma) # subsurface peak, normally distributed
#
## initial conditions
#C = R * dt
#
#ss = [1]
#
##while ss[-1] > 0.00000002:
##    step()
#
##plt.plot(ss)
##plt.plot(x,C)
#
#
## Set up and run animation
#
#fig   = plt.figure()
#
#R_plot = fig.add_subplot(151, ylim=(max_depth, 0), xlim=(0, max(R)*1.5))
#R_line, = R_plot.plot([], [], lw=3)
#R_plot.grid()
#R_plot.axes.get_xaxis().set_ticks([0.0, max(R)])
#R_plot.set_xlabel('R')
#
#D_plot = fig.add_subplot(152, ylim=(max_depth, 0), xlim=(0, max(D)*1.5))
#D_line, = D_plot.plot([], [], lw=3)
#D_plot.grid()
#D_plot.axes.get_yaxis().set_ticklabels([])
#D_plot.axes.get_xaxis().set_ticks([0.0, max(D)])
#D_plot.set_xlabel('D')
#
#w_plot = fig.add_subplot(153, ylim=(max_depth, 0), xlim=(0, max(w)*1.5))
#w_line, = w_plot.plot([], [], lw=3)
#w_plot.grid()
#w_plot.axes.get_yaxis().set_ticklabels([])
#w_plot.axes.get_xaxis().set_ticks([0.0, max(w)])
#w_plot.set_xlabel('w')
#
#u_plot = fig.add_subplot(154, ylim=(max_depth, 0), xlim=(0, max(u)*1.5))
#u_line, = u_plot.plot([], [], lw=3)
#u_plot.grid()
#u_plot.axes.get_yaxis().set_ticklabels([])
#u_plot.axes.get_xaxis().set_ticks([0.0, max(u)])
#u_plot.set_xlabel('u')
#
#C_plot = fig.add_subplot(155, ylim=(max_depth, 0), xlim=(0, 1000))
#C_line, = C_plot.plot([], [], lw=3)
#step_text = C_plot.text(0.2, 0.02, '', transform=C_plot.transAxes)
#C_plot.grid()
#C_plot.axes.get_yaxis().set_ticklabels([])
#C_plot.set_xlabel('C')
#
#plt.subplots_adjust(wspace=0.1)
#
## Clear frame on each interation
#def init():
#    R_line.set_data([], [])
#    D_line.set_data([], [])
#    w_line.set_data([], [])
#    u_line.set_data([], [])
#    C_line.set_data([], [])
#    
#    return R_line,D_line,w_line,u_line,C_line, 
#
## Invoke model timestep and replot data on each iteration
#def animate(i):
#    step()
#    
#    R_line.set_data(R, x)
#    D_line.set_data(D, x)
#    w_line.set_data(w, x)
#    u_line.set_data(u, x)
#    C_line.set_data(C, x)
#
#    step_text.set_text('iter: %.1f' % i)
#
#    return R_line,D_line,w_line,u_line,C_line,step_text
#
#ani = animation.FuncAnimation(fig, animate, frames=10000000, interval=1, blit=True, init_func=init)
#
#plt.show()




    