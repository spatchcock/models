# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 21:56:50 2015

@author: spatchcock
"""

import numpy
from matplotlib import pyplot
import matplotlib.animation as animation  


numpy.set_printoptions(precision=3)

L = 1.
J = 100
dx = float(L)/float(J-1)
x_grid = numpy.array([j*dx for j in range(J)])

T = 200
N = 1000
dt = float(T)/float(N-1)
t_grid = numpy.array([n*dt for n in range(N)])

D_v = float(10.)/float(100.)
D_u = 0.01 * D_v

k0 = 0.067
f = lambda u, v: dt*(v*(k0 + float(u*u)/float(1. + u*u)) - u)
g = lambda u, v: -f(u,v)
 
sigma_u = float(D_u*dt)/float((2.*dx*dx))
sigma_v = float(D_v*dt)/float((2.*dx*dx))

total_protein = 2.26

no_high = 10
U =  numpy.array([0.1 for i in range(no_high,J)] + [2. for i in range(0,no_high)])
V = numpy.array([float(total_protein-dx*sum(U))/float(J*dx) for i in range(0,J)])

pyplot.ylim((0., 2.1))
pyplot.xlabel('x'); pyplot.ylabel('concentration')
pyplot.plot(x_grid, U)
pyplot.plot(x_grid, V)
pyplot.show()

A_u = numpy.diagflat([-sigma_u for i in range(J-1)], -1) +\
      numpy.diagflat([1.+sigma_u]+[1.+2.*sigma_u for i in range(J-2)]+[1.+sigma_u]) +\
      numpy.diagflat([-sigma_u for i in range(J-1)], 1)
        
B_u = numpy.diagflat([sigma_u for i in range(J-1)], -1) +\
      numpy.diagflat([1.-sigma_u]+[1.-2.*sigma_u for i in range(J-2)]+[1.-sigma_u]) +\
      numpy.diagflat([sigma_u for i in range(J-1)], 1)
        
A_v = numpy.diagflat([-sigma_v for i in range(J-1)], -1) +\
      numpy.diagflat([1.+sigma_v]+[1.+2.*sigma_v for i in range(J-2)]+[1.+sigma_v]) +\
      numpy.diagflat([-sigma_v for i in range(J-1)], 1)
        
B_v = numpy.diagflat([sigma_v for i in range(J-1)], -1) +\
      numpy.diagflat([1.-sigma_v]+[1.-2.*sigma_v for i in range(J-2)]+[1.-sigma_v]) +\
      numpy.diagflat([sigma_v for i in range(J-1)], 1)

print A_u

f = lambda u, v: v*(k0 + float(u*u)/float(1. + u*u)) - u

f_vec = lambda U, V: numpy.multiply(dt, numpy.subtract(numpy.multiply(V, 
                     numpy.add(k0, numpy.divide(numpy.multiply(U,U), numpy.add(1., numpy.multiply(U,U))))), U))

print f(U[0], V[0])

print f(U[-1], V[-1])

print f_vec(U, V)

U_record = []
V_record = []

U_record.append(U)
V_record.append(V)

for ti in range(1,N):
    U_new = numpy.linalg.solve(A_u, B_u.dot(U) + f_vec(U,V))
    V_new = numpy.linalg.solve(A_v, B_v.dot(V) - f_vec(U,V))
    
    U = U_new
    V = V_new
    
    U_record.append(U)
    V_record.append(V)

pyplot.ylim((0., 2.1))
pyplot.xlabel('x'); pyplot.ylabel('concentration')
pyplot.plot(x_grid, U)
pyplot.plot(x_grid, V)
pyplot.show()


U_record = numpy.array(U_record)
V_record = numpy.array(V_record)

fig, ax = pyplot.subplots()
pyplot.xlabel('x'); pyplot.ylabel('t')
heatmap = ax.pcolor(x_grid, t_grid, U_record, vmin=0., vmax=1.2)
colorbar = pyplot.colorbar(heatmap)

###

fig = pyplot.figure()

# Limit x-axis to hide stationary points
ax = fig.add_subplot(111, xlim=(1, J), ylim=(0, 3.0))
ax.grid()

ax.plot(range(J), V_record[0,:], lw=3, color='k')

line, = ax.plot([], [], lw=3)
step_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)


# %% Set up animation functions

# Clear frame on each interation
def init():
    line.set_data([], [])
    return line,

# Invoke model timestep and replot data on each iteration
def animate(i):   
    line.set_data(range(J), V_record[i,:])
    step_text.set_text('iter: %.1f' % i)
    return line, step_text

plt.show()
  
anim = animation.FuncAnimation(fig, 
                              animate, 
                              frames=200,
                              interval=200, 
                              blit=True, 
                              init_func=init)







