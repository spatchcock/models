# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 21:51:35 2015

@author: spatchcock
"""
# %%
import numpy
from matplotlib import pyplot
import matplotlib.animation as animation

numpy.set_printoptions(precision=3)

# %% Numerical scheme

L = 1.
J = 100
dx = float(L)/float(J-1)
x_grid = numpy.array([j*dx for j in range(J)])

T = 200
N = 1000
dt = float(T)/float(N-1)
t_grid = numpy.array([n*dt for n in range(N)])

w = 0.1
sigma = float(w*dt)/float((4.*dx))

# %% Initial conditions

#C =  numpy.zeros(J)
#C[10:20] = 5

gaussian = lambda z, height, position, hwhm: height * numpy.exp(-numpy.log(2) * ((z - position)/hwhm)**2)
C = gaussian(x_grid, 5, 0.5, 0.1)

# %% Set up matrices

A_C = numpy.diagflat([-sigma for i in range(J-1)], -1) +\
      numpy.diagflat([1.-sigma]+[1 for i in range(J-2)]+[1.+sigma]) +\
      numpy.diagflat([sigma for i in range(J-1)], 1)
        
B_C = numpy.diagflat([sigma for i in range(J-1)], -1) +\
      numpy.diagflat([1.+sigma]+[1. for i in range(J-2)]+[1.-sigma]) +\
      numpy.diagflat([-sigma for i in range(J-1)], 1)

# %% Run

#C_record = []

#C_record.append(C)

def step(C):
    C = numpy.linalg.solve(A_C, B_C.dot(C))
    #C = C_new
    #C_record.append(C)


# %% Set up figure

fig = pyplot.figure()

# Limit x-axis to hide stationary points
ax = fig.add_subplot(111, xlim=(0, L), ylim=(0, 10))
ax.grid()

pyplot.plot(x_grid, C, lw=3, color='k')

line, = ax.plot([], [], lw=3)
step_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)


# %% Set up animation functions

# Clear frame on each interation
def init():
    line.set_data([], [])
    return line,

# Invoke model timestep and replot data on each iteration
def animate(i):
    step(C)
    
    line.set_data(x_grid, C)
    step_text.set_text('iter: %.1f' % i)
    return line, step_text




pyplot.show()
     
anim = animation.FuncAnimation(fig, 
                              animate, 
                              frames=200,
                              interval=20, 
                              blit=True, 
                              init_func=init)
