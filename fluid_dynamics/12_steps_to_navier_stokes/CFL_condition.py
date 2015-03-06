# -*- coding: utf-8 -*-
"""
Created on Fri Mar 06 12:44:26 2015

@author: andrew.berkeley
"""

# This script is based on part 3 of Lorena Barba's "12 steps to Navier Stokes" series. It
# introduces the concept of the CFL condition.
#
# In the earlier lessons it was seen that varying the space step either caused numerical
# diffusion (smoothing of wave as derivatives poorly approximated) or instability. As the 
# space resolution is increased, the numerical diffusion is decreased, increasing the model 
# accuracy but only up to a point, beyond which instability is caused.
#
# The reason is that when the space resolution becomes sufficiently fine relative to the 
# time resolution, the distance that the wave travels in 1 time step may be larger than
# a single space step. Stability can therefore be enforced if the timestep is defined
# relative to the space step, ensuring that it is always sufficiently small whatever
# space step is chosen.
#
# This example simply generates several model runs of the 1D linear advection model but
# with the dt term derived as a function of the dx term. The proportionality constant, 
# here represented by sigma, is called the Courant Number, and defined in the general case 
# as:
#
#   sigma = c*dt/dx < sigma_max
#
# This basically states that there exists a value, sigma_max, that depends on the discritization 
# used that will ensure stability. As long as the realised sigma value is less than this value, 
# the model will be stable. 
#
# This example wraps up the model code from the 1D linear advection lesson into a function
# which can be invoked specifying the space resolution.
#
# Points to note
# --------------
#
# 1. Increasing the space resolution decreases the numerical diffusion - the original top
#    hat function is preserved much better with increasing numbers of nx. 
#
# 2. Unlike previous examples, the resolution can be increased much higher than nx = 81
#    without causing instability. This means we can reduce the numerical diffusion with
#    much greater spatial resolution and still have a stable model.
#
# 3. sigma is basically defined as dt/dx in the implementation below, which is not quite the
#    definition above. This is just because c is chosen to be 1. If c is increased to,
#    say, 4, this changes the implied Courant number and in fact returns the model to 
#    instability. This requires altering the timestep, in accordance with the definition of 
#    the Courant number, which in this implementation is easily done by reducing the sigma 
#    value, to say 0.1. So if we have faster advection, we needs a smaller timestep.
#
# 4. If we're only considering a fixed number of timesteps, changing the size of the time
#    step obviously results in a smaller amount of advection in the model run, since a 
#    smaller absolute time window is being simulated.
#
# 5. This example illustrates the need for a sufficiently fine spatial resolution so as to 
#    minimize numerical diffusion, but given the spatial resolution the time resolution 
#    needs to be sufficiently and comparatively fine to ensure stability. Given that numerical 
#    diffusion occurs when derivatives are under-estimated due to a coarse grid, it stands 
#    to reason that spatial resolution needs to be finest where gradients are expected to vary 
#    rapidly across space. This might be the case near a boundary, e.g. near a land boundary 
#    or at the sea bed in a coastal hydrodynamic model.
#
#

import numpy as np                 #numpy is a library for array operations akin to MATLAB
import matplotlib.pyplot as plt    #matplotlib is 2D plotting library

# %% Wrap 1D linear advection model in a function which takes space resolution as argument

def linearconv(nx):
    dx = 2./(nx-1)
    nt = 20    # nt is the number of timesteps we want to calculate
    c = 2
    
    # Instead of setting the dt term as an absolute, fixed value, let's define it relative to
    # dx. The proportionality constant is sigma, as represented by the Courant Number.
    sigma = 0.5
    dt = sigma*dx  # 

    u = np.ones(nx)      # defining a numpy array which is nx elements long with every value equal to 1.
    u[.5/dx : 1/dx+1]=2  # setting u = 2 between 0.5 and 1 as per our I.C.s

    un = np.ones(nx) # initializing our placeholder array, un, to hold the values we calculate for the n+1 timestep

    for n in range(nt):  # iterate through time
        un = u.copy()    # copy the existing values of u into un
        for i in range(1,nx):
            u[i] = un[i]-c*dt/dx*(un[i]-un[i-1])
    
    plt.figure()    
    plt.plot(np.linspace(0,2,nx),u)


 # %% Plot some results

linearconv(41)
linearconv(81)
linearconv(101)
linearconv(121)
linearconv(221)