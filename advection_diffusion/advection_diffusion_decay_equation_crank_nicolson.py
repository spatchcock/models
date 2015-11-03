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

w = 0.01
D = 0.0000
lamda = 0.01

L = 1.
J = 500
dx = float(L)/float(J-1)
x_grid = numpy.array([j*dx for j in range(J)])

courant = 0.5

dt = dx*courant/w

T = 50
N = int(T/dt + 1)

t_grid = numpy.array([n*dt for n in range(N)])

sigma = float(w*dt)/float((4.*dx))
sigma

rho = float(D*dt)/(2*dx**2) 


mu = float(lamda*dt/2)

# %% Initial conditions

#C =  numpy.zeros(J)
#C[100:200] = 5

gaussian = lambda z, height, position, hwhm: height * numpy.exp(-numpy.log(2) * ((z - position)/hwhm)**2)
C = gaussian(x_grid, 5, 0.15, 0.03)

# %% Set up matrices

A_C = numpy.diagflat([-(sigma+rho) for i in range(J-1)], -1) +\
      numpy.diagflat([1.-sigma+rho+mu]+[1+2*rho+mu for i in range(J-2)]+[1.+sigma+rho+mu]) +\
      numpy.diagflat([sigma-rho for i in range(J-1)], 1)
        
B_C = numpy.diagflat([sigma+rho for i in range(J-1)], -1) +\
      numpy.diagflat([1.+sigma-rho-mu]+[1.-2*rho-mu for i in range(J-2)]+[1.-sigma-rho-mu]) +\
      numpy.diagflat([rho-sigma for i in range(J-1)], 1)

# %% Run

C_record = []

C_record.append(C)

for t in numpy.arange(1,N-1):
    C = C_record[t-1]
    C_new = numpy.linalg.solve(A_C, B_C.dot(C))
    
    C_record.append(C_new)


# %% Set up figure

fig = pyplot.figure()

# Limit x-axis to hide stationary points
ax = fig.add_subplot(111, xlim=(0, L), ylim=(0, 5))
ax.grid()

#pyplot.plot(x_grid, C_record[0], lw=3, color='k')

line, = ax.plot([], [], lw=3)
step_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)


# %% Set up animation functions

# Clear frame on each interation
def init():
    line.set_data([], [])
    return line,

# Invoke model timestep and replot data on each iteration
def animate(i):  
    line.set_data(x_grid, C_record[i])
    step_text.set_text('iter: %.1f' % i)
    return line, step_text




pyplot.show()
     
anim = animation.FuncAnimation(fig, 
                              animate, 
                              frames=N,
                              interval=100, 
                              blit=True, 
                              init_func=init)
