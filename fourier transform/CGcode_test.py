# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 19:55:56 2023

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt
#from numpy.fft import fft2, ifft2
from function_test import costfunction
from function_test import converter
#import tensorflow as tf
#import imageio
import time
#import os
#import shutil
import cv2
from fourier_transfrom import ft
from fourier_transfrom import plot

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


# -------------------------Target, Input and Diffraction plane

# Super-Gaussian profile parameters
mx = 3  # Super-Gaussian exponent
my = 1
A = 1  # Super-Gaussian amplitude
sigx = 80e-3/ody #53e-3/odx  # pixel
sigy = 10e-3/ody #10e-3/ody  # pixel
# Generate super-Gaussian profile -- it is intensity
target = np.square(A * np.exp(-(np.abs(oX)/(sigx*dx))**(2*mx)) * np.exp(-(np.abs(oY)/(sigy*dy))**(2*my)))
target /= np.max(target)  # Normalize matrix
targetamp = A * np.exp(-(np.abs(oX)/(sigx*dx))**(2*mx)) * np.exp(-(np.abs(oY)/(sigy*dy))**(2*my))
targetphase = np.ones([oNx, oNy])

ret, binary = cv2.threshold(target, 0.4*A, 1*A, cv2.THRESH_BINARY)
rg = binary/A/A # the range of costfunction selected

target = target*rg  # Normalize matrix
targetphase = targetphase*rg

# Calculate Gaussian beam profile -- it is electric field **amplitude**
sigx = 1.783*np.sqrt(2)/dx # pixel
sigy = 1.855*np.sqrt(2)/dy # pixel
x0 = 0 # pixel
y0 = 0 # pixel
A = 1

initial_profile = A*np.exp(-((X-x0*dx)**2/2/(sigx*dx)**2+(Y-y0*dy)**2/2/(sigy*dy)**2))
#plot(x, y, initial_profile, 'Greens')
#plot(ox, oy, targetamp, 'Blues')
# Defining DOE phase
#DOE = np.load('DOE_data.npy')
#DOE = np.random.rand(Ny,Nx)*2*np.pi
DOE = np.zeros((Ny, Nx))

#----------------------------------------Other parameters

s = 1 # external interation number
f = 110 # focal length
l = 895e-6 # lambda
factor = f*l

learn_rate = 0.01
# cost function = np.sum( target - f)**2

DOEphase = np.exp(1j * DOE)
iterf = ft(x, y, ox, oy, initial_profile * DOEphase, f, l)
ampf = np.abs(iterf)
angf = np.angle(iterf)

cost = np.sum((targetamp - ampf)**2)
sx, sy = np.gradient(initial_profile)
#somethingbad = (1j*2*np.pi/l/f*ox * initial_profile + sx)


cost_px = np.sum(2*(targetamp - ampf)*(-ft(x, y, ox, oy, sx, f, l)))










