# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 21:06:14 2023

@author: user
"""

import matplotlib.pyplot as plt
#import matplotlib as plt
import numpy as np

Nx = 1920 # number
Ny= 1080 # number
Lx = 15.62 # (mm)
Ly = 8.7 # (mm)
x = np.linspace(-Lx/2, Lx/2, Nx)
y = np.linspace(-Ly/2, Ly/2, Ny)
dx = Lx/np.size(x) # (mm)
dy = Ly/np.size(y) # (mm)
X, Y = np.meshgrid(x, y)

# Parameters
'''
N = 1
l = 895e-9 # lambda
k = 2*np.pi/l
f = 110e-3
'''

sigx = 1.783/dx # pixel
sigy = 1.855/dy # pixel
x0 = 0 # pixel
y0 = 0 # pixel
A = 1

# Generate super-Gaussian profile -- it is intensity
target = A*np.exp(-((X-x0*dx)**2/2/(sigx*dx)**2+(Y-y0*dy)**2/2/(sigx*dx)**2))
#target = N**2*np.exp(-1j*k/2/f*(X**2 + Y**2))
m = np.max(target)
fig, ax = plt.subplots()
t = plt.title("input")
ax.imshow(np.real(target), cmap = 'Blues')
ax.contour(target, cmap = 'Blues_r', levels = [0, m/5*1, m/5*2, m/5*3, m/5*4, m])
ax.contour(target, colors = 'r', levels = [m/2], linestyles = 'dashed')
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
t = plt.title("input_3d")
ax.plot_surface(X,Y, np.real(target), cmap = 'Blues')

#plt.colorbar()
# ----------------------------------------omega plane

oNx = 100 # total = 1296
oNy= 100 # total = 964
odx = 3.75e-3 # (mm)
ody = 3.75e-3 # (mm)
oLx = oNx * odx # (mm)
oLy = oNy * ody # (mm)
ox = np.linspace(-oLx/2, oLx/2, oNx)
oy = np.linspace(-oLy/2, oLy/2, oNy)

oX, oY = np.meshgrid(ox, oy)

# Parameters
'''
N = 1
l = 895e-9 # lambda
k = 2*np.pi/l
f = 110e-3
'''

sigx = 9.053e-3/odx # pixel
sigy = 8.7e-3/ody # pixel
x0 = 0 # pixel
y0 = 0 # pixel
A = 1

# Generate super-Gaussian profile -- it is intensity
fttarget = A*np.exp(-((oX-x0*odx)**2/2/(sigx*odx)**2+(oY-y0*ody)**2/2/(sigx*odx)**2))
#target = N**2*np.exp(-1j*k/2/f*(X**2 + Y**2))
m = np.max(fttarget)

fig, ax = plt.subplots()
t = plt.title("output")
ax.imshow(np.real(fttarget), cmap = 'Greens')
ax.contour(fttarget, cmap = 'Greens_r', levels = [0, m/5*1, m/5*2, m/5*3, m/5*4, m])
ax.contour(fttarget, cmap = 'Reds', levels = [m/2])
ax.contour(fttarget, colors = 'r', levels = [m/2], linestyles = 'dashed')

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
t = plt.title("output_3d")
ax.plot_surface(oX,oY, np.real(fttarget), cmap = 'Greens')
#plt.colorbar()
