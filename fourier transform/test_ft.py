# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 15:28:42 2023

@author: user
"""

import matplotlib.pyplot as plt
import numpy as np

# -----------------------------SLM plane (PLUTO-2.1)
#   15.62 * 8.7 (mm^2)
#   1920 * 1080 (pixel)
Nx = 1920 # number
Ny= 1080 # number
Lx = 15.62 # (mm)
Ly = 8.7 # (mm)
x = np.linspace(-Lx/2, Lx/2, Nx)
y = np.linspace(-Ly/2, Ly/2, Ny)
dx = Lx/np.size(x) # (mm)
dy = Ly/np.size(y) # (mm)
X, Y = np.meshgrid(x, y)

# -----------------------------CCD plane (CMLN13S2C)

oNx = 100 # total = 1296
oNy= 100 # total = 964
odx = 3.75e-3 # (mm)
ody = 3.75e-3 # (mm)
oLx = oNx * odx # (mm)
oLy = oNy * ody # (mm)
ox = np.linspace(-oLx/2, oLx/2, oNx)
oy = np.linspace(-oLy/2, oLy/2, oNy)
oX, oY = np.meshgrid(ox, oy)

#---------------------------------------input profile
sigx = 1.783/dx # pixel
sigy = 1.855/dy # pixel
x0 = 0 # pixel
y0 = 0 # pixel
A = 1
m = 1/2
# Generate Gaussian profile -- input

g = A*np.sin(X/m)/X*np.sin(Y/m)/Y
gx =  g[int(Nx/2), :]
gy =  g[:, int(Ny/2)]
#--------------------------------------Fourier transform

f = 110 # focal length
l = 895e-6 # lambda

def ft(init_x, init_y, omega_x, omega_y, function, f, l):
    factor = f*l
    ftx_operate = np.exp(1j*2*np.pi/factor*np.transpose(np.array([omega_x]))*np.array([init_x]))
    fty_operate = np.exp(1j*2*np.pi/factor*np.transpose(np.array([omega_y]))*np.array([init_y]))
    #---------------------------------------output profile
    o = np.dot(ftx_operate, np.transpose(function))
    o = np.dot(o, np.transpose(fty_operate))
    return o

o = ft(x, y, ox, oy, g, f, l)

#------------------------------------plot
def plot(x, y, function, colors):
    
    m = np.max(np.abs(function))
    fig, ax = plt.subplots()
    t = plt.title('before ft')
    ax.imshow(np.abs(function), cmap = colors)
    ax.contour(np.abs(function), cmap = colors + '_r', levels = [0, m/5*1, m/5*2, m/5*3, m/5*4, m])
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    t = plt.title('before ft')
    X, Y = np.meshgrid(x, y)
    ax.plot_surface(X,Y, np.abs(function), cmap = colors)
    return True
plot(x, y, g, 'Blues')
plot(ox, oy, o, 'Greens')