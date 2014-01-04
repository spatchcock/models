# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib.animation as animation

# v = u + 1/2 * a * t^2
#
# The animation pattern is based on the animation tutorial found here: 
#  http://jakevdp.github.io/blog/2012/08/18/matplotlib-animation-tutorial/
#

class Bounce:
    
    # This class is just a container for the model
    
    def __init__(self, initial_x, initial_y, u_y, dt):
        
        self.v_y          = u_y
        
        self.x = initial_x
        self.y = initial_y
        
        self.a = -9.8
        
        self.dt = dt
        

        
    # Iterate
    def step(self):
        if self.y < 0.0:
            self.v_y = -self.v_y
            
        self.v_y = self.v_y + (self.a * self.dt)
        self.y = self.y + self.v_y
        
        
    

#------------------------------------------------------------

# set up initial state and global variables

initial_x = 5
initial_y = 0
u_y       = 5
dt        = 0.01

#------------------------------------------------------------

# Initialize object
bounce = Bounce(initial_x, initial_y, u_y, dt)

#------------------------------------------------------------

# set up figure and animation
fig = plt.figure()
# Limit x-axis to hide stationary points
ax = fig.add_subplot(111, xlim=(1, 10), ylim=(0, 500))
ax.grid()

line, = ax.plot([], [], 'o-', lw=3) # 
step_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

def init():
    # Clear frame on each interation
    line.set_data([], [])
    return line,

def animate(i):
    global bounce       
    bounce.step()
    
    line.set_data(bounce.x, bounce.y)
    step_text.set_text('iter: %.1f' % i)
    return line, step_text

ani = animation.FuncAnimation(fig, animate, frames=500,
                              interval=100, blit=True, init_func=init)

plt.show()