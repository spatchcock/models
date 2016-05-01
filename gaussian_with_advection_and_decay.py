# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 10:45:25 2016

@author: spatchcock
"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import matplotlib.colors as colors


#%%

def two_d_gauss(x, y, M, meanx, meany, stdevx, stdevy):
    
    g = np.zeros((np.size(x), np.size(y)))
    
    for i in np.arange(0, np.size(x), 1):
        for j in np.arange(0, np.size(y), 1):
            g[i][j] = (M/(2*np.pi*stdevx*stdevy)) * np.exp(-((((x[i] - meanx)**2.0)/(2.0*(stdevx**2.0))) + (((y[j] - meany)**2.0)/(2.0*(stdevy**2.0)))))
    
    return g


### Invoke model timestep and replot data on each iteration
#def animate(i):
#    data = two_d_gauss(x, y, M, meanx[i], meany[i], stdevx[i], stdevy[i])
#
#    im.set_array(np.ravel(data)
#    step_text.set_text('iter: %.1f' % i)
#    plt.draw()

#%%

t = np.arange(1,1000,1)

domain_range = 100

x = np.arange(-domain_range/2,domain_range/2,1)
y = np.arange(-domain_range/2,domain_range/2,1)

u = 0.3*np.sin(2*np.pi*t/50)
v = 0.0
D_x = 0.5
D_y = 0.1

startx = 0.0
starty = 0.0

M = 1000
meanx   = startx + u*t
stdevx  = np.sqrt(2.0*D_x*t)
meany   = starty + v*t
stdevy  = np.sqrt(2.0*D_y*t)

#%%
X, Y = np.meshgrid(x, y)

Z = two_d_gauss(x, y, M, startx, starty, stdevx[0], stdevy[0])
Z_max = np.max(Z)
norm=colors.Normalize(vmin=0.,vmax=Z_max/10.0)

fig = plt.figure()

ims = []

ims.append((plt.pcolor(X,Y,Z, cmap='Reds', norm=norm),))

for ts in np.arange(2,100,1):
    Z = two_d_gauss(x, y, M, meanx[ts], meany[ts], stdevx[ts], stdevy[ts])
    ims.append((plt.pcolor(X,Y,Z, cmap='Reds', norm=norm),))


#%%
### Plot ###

im_ani = animation.ArtistAnimation(fig, ims, interval=50, repeat_delay=500, blit=True)
#im_ani.save('im.mp4', metadata={'artist':'Guido'})

plt.show()
