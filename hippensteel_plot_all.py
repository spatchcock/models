# -*- coding: utf-8 -*-
"""
Created on Sat May  3 21:38:54 2014

@author: spatchcock
"""
import numpy as np
import matplotlib.pyplot as plt


high_file = '/home/spatchcock/Documents/assemblage/marmic/hippensteel_total_high_marsh.csv'
intm_file = '/home/spatchcock/Documents/assemblage/marmic/hippensteel_total_intermediate_marsh.csv'
low_file  = '/home/spatchcock/Documents/assemblage/marmic/hippensteel_total_low_marsh.csv'

high_data = np.genfromtxt(high_file, dtype=float, delimiter=',', names=True) 
intm_data = np.genfromtxt(intm_file, dtype=float, delimiter=',', names=True) 
low_data  = np.genfromtxt(low_file,  dtype=float, delimiter=',', names=True) 

def add_subplot(master_fig, csv, species, plot_index):
    plot = master_fig.add_subplot(3,5,plot_index, xlim=(0.0,max(csv[species + '_Dead'])), ylim=(60.0, 0))
    plot.plot(csv[species + '_Dead'],csv['Depth'], marker='o', linestyle='None')
    plot.plot(csv[species + '_Live'],csv['Depth'], marker='o', linestyle='None')
    plot.set_title(species)
    plot.grid()
    
    a,b=np.polynomial.polynomial.polyfit(csv['Depth'],csv[species + '_Live'],4,full=True)
    x=np.linspace(0,60,500)
    y=a[0] + a[1]*x + a[2]*x**2 + a[3]*x**3 + a[4]*x**4# + a[5]*x**5# + a[6]*x**6
    
    plot.plot(y,x)
    

fig = plt.figure()
add_subplot(fig, high_data, 'AMEX',1)
add_subplot(fig, high_data, 'MFUS',2)
add_subplot(fig, high_data, 'PLIM',3)
add_subplot(fig, high_data, 'TINF',4)
add_subplot(fig, high_data, 'JMAC',5)
add_subplot(fig, intm_data, 'AMEX',6)
add_subplot(fig, intm_data, 'MFUS',7)
add_subplot(fig, intm_data, 'PLIM',8)
add_subplot(fig, intm_data, 'TINF',9)
add_subplot(fig, intm_data, 'JMAC',10)
add_subplot(fig, low_data,  'AMEX',11)
add_subplot(fig, low_data,  'MFUS',12)
add_subplot(fig, low_data,  'PLIM',13)
add_subplot(fig, low_data,  'TINF',14)
add_subplot(fig, low_data,  'JMAC',15)
fig.tight_layout()
plt.show()