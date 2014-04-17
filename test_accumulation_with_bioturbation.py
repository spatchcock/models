# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 22:52:11 2014

@author: spatchcock
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from sympy.functions import coth

plt.ion()
#   
#w  = 1#0.025
#D  = 1#0.02
#u  = 1#0.001
#
#def decay(C,x):
#    f2 = -u*C[0]
#    
#    return [f2]
#
#def advection_decay(C,x):
#    f2 = -u/w*C[0]
#    
#    return [f2]
#
##D C'' - u C = 0
##
##c1 = C
##c2 = C'
##
##c1' = c2
##c2' = u/D * c1
#
#def diffusion_decay(C,x):
#    f1 = C[1]
#    f2 = u/D*C[0]
#    
#    return [f1,f2]
#    
#def advection_diffusion(C,x):
#    f2 = -u/w*C[0]
#    
#    return [f2]
#
##D C'' - w C' - u C = 0
##
##a = C
##b = C'
##
##a' = b
##b' = w/D * b + u/D * a

#def advection_diffusion_decay(C,x):
#    f1 = C[1]
#    f2 = w/D * C[1] + u/D*C[0]
#    
#    return [f1,f2]
#
#Define the time steps for the solution.
#
#depth = np.linspace(0.0,100.0,101)
#
##Call the integrate function to obtain the solution.
#
#soln = odeint(diffusion_decay,[1,1],depth)
#
##Step 4
##Plot the solution:
#
#plot(depth[:],soln[:,0])
#xlabel('x')
#ylabel('C')
#show()
#
#def deriv(y,t):
## return derivatives of the array y
#  a = -.9
#  b = -.01
#  
#  if t < 3:
#      c = 10
#  else:
#      c=0
#  
#  D=0.1
#  return np.array([ y[1], a/D*y[0]+b/D*y[1] + c/D])
#
#time = np.linspace(0.0,10.0,500)
#yinit = np.array([0,0])
## initial values
#y =odeint(deriv,yinit,time)
#plt.figure()
#plt.plot(time,y[:,1])
## y[:,0] is the first column of y
#plt.xlabel('t')
#plt.ylabel('y')
#plt.show()
#
#
##==============================================================================
#
##Decay
##
## dC/dx = -uC
##
## C{i} - C{i-1}/dx = -uC{i-1}
## c{i} = C{i-1} - dx *uC{i-1} 
#
## Numerics
#max_x = 50.0
#no_cells = 10001
#dx = max_x/no_cells
#
## Parameters
#u  = 0.1 
#
## Initialize
#x = np.linspace(0.0,max_x,no_cells)
#C  = np.zeros(10001)
#
## initial condition
#C[0] = 100
#
#for i in range(1,size(C)):
#    C[i] = C[i-1]- u * C[i-1] * dx
#
#plt.figure()
#plt.plot(x,C)
#plt.xlabel('x')
#plt.ylabel('C')
#plt.xlim(0,max_x)
#plt.ylim(0,100)
#plt.show()

#==============================================================================

# Advection - Decay
#
# dC/dx = -u/wC
#
# C{i} - C{i-1}/dx = -uC{i-1}
# c{i} = C{i-1} - dx *uC{i-1} 
#
## Numerics
#max_x = 50.0
#no_cells = 10001
#dx = max_x/no_cells
#
## Parameters
#u  = 0.001 
#w  = 0.025
#
## Initialize
#x = np.linspace(0.0,max_x,no_cells)
#C = np.zeros(10001)
#
## initial condition
#C[0] = 100
#
#for i in range(1,size(C)):
#    C[i] = C[i-1]- u/w * C[i-1] * dx
#
#plt.figure()
#plt.plot(x,C)
#plt.xlabel('x')
#plt.ylabel('C')
#plt.xlim(0,max_x)
#plt.ylim(0,100)
#plt.show()


#==============================================================================

# Advection - diffusion - Decay
#
# D(d^2C/dx^2) - w(dC/dx) - uC = 0
# dC/dx = D/w(d^2C/dx^2) - uC/w
#
# C{i} - C{i-1}/dx = D/w(C{i} - 2C{i-1} + C{i-2})/dx^2 - uC{i-1}/w
# C{i} - C{i-1} = (D*dx/w)*(C{i} - 2C{i-1} + C{i-2})/dx^2 - (u*dx/w)*C{i-1}
# C{i} = C{i-1} + (D/w*dx)*C{i} - (D/w*dx)*2C{i-1} + (D/w*dx)*C{i-2} - (u*dx/w)*C{i-1}
# C{i} - (D/w*dx)*C{i} = C{i-1} - (D/w*dx)*2C{i-1} + (D/w*dx)*C{i-2} - (u*dx/w)*C{i-1}
# C{i} * (1 - (D/w*dx)) = C{i-1} - (D/w*dx)*2C{i-1} + (D/w*dx)*C{i-2} - (u*dx/w)*C{i-1}
# C{i} = [ C{i-1} - (D/w*dx)*2C{i-1} + (D/w*dx)*C{i-2} - (u*dx/w)*C{i-1} ] / (1 - (D/w*dx))

## Numerics
#max_x = 50.0
#no_cells = 10001
#dx = max_x/no_cells
#
## Parameters
#u  = 0.001 
#w  = 0.025
#D  = 0.0002
#
## Initialize
#x = np.linspace(0.0,max_x,no_cells)
#C = np.zeros(10001)
#
## initial condition
#C[0] = 100
#C[1] = C[0]- u/w * C[0] * dx # Just advection-decay
#
#for i in range(2, size(C)):
#    C[i] = C[i-1]- u/w * C[i-1] * dx
#    C[i] = (C[i-1] - (2*D*C[i-1])/(w*dx) + (D*C[i-2])/(w*dx) - (u*dx*C[i-1])/w) / (1 - (D/(w*dx)))

#plt.figure()
#plt.plot(x,C)
#plt.xlabel('x')
#plt.ylabel('C')
#plt.xlim(0,max_x)
#plt.ylim(0,100)
#plt.show()

#
##==============================================================================
#
## Advection - diffusion - Decay - production
##
## D(d^2C/dx^2) - w(dC/dx) - uC + R = 0
## dC/dx = D/w(d^2C/dx^2) - uC/w R/w
##
## C{i} - C{i-1}/dx = D/w(C{i} - 2C{i-1} + C{i-2})/dx^2 - uC{i-1}/w + R{x}/w
## C{i} - C{i-1} = (D*dx/w)*(C{i} - 2C{i-1} + C{i-2})/dx^2 - (u*dx/w)*C{i-1} + R{x}dx/w
## C{i} = C{i-1} + (D/w*dx)*C{i} - (D/w*dx)*2C{i-1} + (D/w*dx)*C{i-2} - (u*dx/w)*C{i-1} + R{x}dx/w
## C{i} - (D/w*dx)*C{i} = C{i-1} - (D/w*dx)*2C{i-1} + (D/w*dx)*C{i-2} - (u*dx/w)*C{i-1} + R{x}dx/w
## C{i} * (1 - (D/w*dx)) = C{i-1} - (D/w*dx)*2C{i-1} + (D/w*dx)*C{i-2} - (u*dx/w)*C{i-1} + R{x}dx/w
## C{i} = [ C{i-1} - (D/w*dx)*2C{i-1} + (D/w*dx)*C{i-2} - (u*dx/w)*C{i-1} + R{x}dx/w ] / (1 - (D/w*dx))
#
## Numerics
#max_x = 50.0
#no_cells = 100001
#dx = max_x/no_cells
#
## Parameters
#u  = 0.01 
#w  = 0.5
#D  = 0.05
#
## Initialize
#x = np.linspace(0.0,max_x,no_cells)
#C = np.zeros(no_cells)
#R = np.zeros(no_cells)
#a = 0.1
#
## initial condition
#R0 = 5.0
#
#for r in range(1,size(R)):
#    R[r] = R0 * math.exp(-a*x[r])
#
#C[0] = (R[0]*dx)/w
#C[1] = C[0] + (R[1]*dx)/w - (u*dx*C[0])/w # Just advection-decay
#
#for i in range(2, size(C)):
#    C[i] = (C[i-1] - (2*D*C[i-1])/(w*dx) + (D*C[i-2])/(w*dx) - (u*dx*C[i-1])/w + (R[i]*dx)/w) / (1 - (D/(w*dx)))
#
##plt.figure()
#plt.plot(x,R)
##plt.xlim(0,max_x)
##plt.ylim(0,100)
#plt.show()
#
#plt.plot(x,C)
#plt.xlabel('x')
#plt.ylabel('C')
##plt.xlim(0,max_x)
##plt.ylim(0,100)
#plt.show()


#==============================================================================
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

# Numerics
max_x = 50.0
no_cells = 1001
dx = max_x/no_cells
dt = 0.1*dx #

depth_mix = 15.0
cell_mix_base = int(depth_mix/max_x*no_cells)

# Parameters
u  = 0.0001 
w  = 100.5
D  = 0.09
R0 = 10.0
a  = 0.5

#scaling factor for advection term (see Boudreau)
#cosh/sinh is the coth, the hyperbolic co-tangent
peclet_number = ((w*dx)/(2*D))
sigma = np.cosh(peclet_number)/np.sinh(peclet_number) - 1.0/peclet_number

# Initialize
x = np.linspace(0.0,max_x,no_cells)
C = np.zeros(no_cells)
R = np.zeros(no_cells)

# initial conditions

#set up R
#for r in range(0,size(R)):
#    R[r] = R0 * math.exp(-a*x[r])

depth_max = 5
gamma = 4

for r in range(0,size(R)):
    R[r] = R0 * math.exp(-a*(x[r]-depth_max)**gamma)

#R[0:99] = 5.0

#for i in range(no_cells):range(no_cells)
C = R * dt

def step():
    Cu = C.copy()
    
    C[0] = dt * R[0]
        
    for i in range(1, cell_mix_base):
       C[i] = Cu[i] + dt * (D*(Cu[i+1] - 2.0*Cu[i] + Cu[i-1])/dx**2.0 - w*(Cu[i+1] - Cu[i-1])/2.0*dx - u*Cu[i] + R[i])
        
#    for i in range(cell_mix_base, size(C)-1):
#        C[i] = Cu[i] + dt * (-w*(Cu[i+1] - Cu[i-1])/(2.0*dx) - u*Cu[i] + R[i])

def step_and_plot():
    step()
    
    plt.plot(x,C)
    plt.show()

# Set up and run animation

fig   = plt.figure()
ax    = fig.add_subplot(111, xlim=(0, max_x), ylim=(0, 100))
line, = ax.plot([], [], lw=3)
step_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

ax.grid()

# Clear frame on each interation
def init():
    line.set_data([], [])
    return line,

# Invoke model timestep and replot data on each iteration
def animate(i):
    step()
    
    line.set_data(x, C)
    step_text.set_text('iter: %.1f' % i)

    return line,step_text

ani = animation.FuncAnimation(fig, animate, frames=10000, interval=1, blit=True, init_func=init)

plt.show()