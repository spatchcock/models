# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 22:11:12 2014

@author: spatchcock
"""

import math
import numpy
import matplotlib.pyplot as plt

# Plot the normal distribution function as well as its first and second derivatives
# 
# Use the numpy.vectorize function to handle array manupulation

# http://statistics.about.com/od/Mathstat/a/Inflection-Points-Of-The-Probability-Density-Function-Of-A-Normal-Distribution.htm

def norm(x, mean, sd):
    var = sd**2
    pi = 3.1415926
    denom = (2*pi*var)**.5
    num = math.exp(-(x-mean)**2/(2*var))
    return num/denom
    
def norm_first_deriv(x, mean, std):
    return -(x-mean)*norm(x, mean, std)/std**2

    
def norm_second_deriv(x, mean, std):
    return -norm(x, mean, std)/std**2 + (x-mean)**2*norm(x, mean, std)/std**4
    
    
v_norm              = numpy.vectorize(norm)
v_norm_first_deriv  = numpy.vectorize(norm_first_deriv)
v_norm_second_deriv = numpy.vectorize(norm_second_deriv)


mean = 0
std  = 1.9
a = numpy.arange(-5,5,0.1)
b = v_norm(a, mean, std)
c = v_norm_first_deriv(a, mean, std)
d = v_norm_second_deriv(a, mean, std)

fig = plt.figure()

norm = fig.add_subplot(111, xlim=(-6, 6), ylim=(-1, 1))
norm.grid()

line, = norm.plot([], [], lw=3, color='r')
line.set_data(a,b)

first = fig.add_subplot(111, xlim=(-6, 6), ylim=(-1, 1))
line, = first.plot([], [], lw=3, color='b')
line.set_data(a,c)

second = fig.add_subplot(111, xlim=(-6, 6), ylim=(-1, 1))
line, = second.plot([], [], lw=3, color='g')
line.set_data(a,d)


stddev = fig.add_subplot(111, xlim=(-6, 6), ylim=(-1, 1))
line, = stddev.plot([], [], lw=3, color='y')
line.set_data([-std, -std],[-1,1])


constant = fig.add_subplot(111, xlim=(-6, 6), ylim=(-1, 1))
line, = constant.plot([], [], lw=3, color='b')
line.set_data([-6, 6],[0.1,0.1])