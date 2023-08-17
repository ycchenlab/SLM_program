# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 19:50:58 2023

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2
from function_test import costfunction
from function_test import converter
import tensorflow as tf
import imageio
import time
import os
import shutil
import cv2

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
sigx = 8e-3/odx  # pixel
sigy = 2e-3/ody  # pixel
# Generate super-Gaussian profile -- it is intensity
target = np.square(A * np.exp(-(np.abs(oX)/sigx)**(2*mx)) * np.exp(-(np.abs(oY)/(sigy))**(2*my)))
target /= np.max(target)  # Normalize matrix
targetphase = np.ones([oNx, oNy])

ret, binary = cv2.threshold(target, 0.4*A, 1*A, cv2.THRESH_BINARY)
rg = binary/A/A # the range of costfunction selected

target = target*rg  # Normalize matrix
targetphase = targetphase*rg

# Calculate Gaussian beam profile -- it is electric field amplitude
sigx = 1.783/dx # pixel
sigy = 1.855/dy # pixel
x0 = 0 # pixel
y0 = 0 # pixel
A = 1

initial_profile = A*np.exp(-((X-x0*dx)**2/2/(sigx*dx)**2+(Y-y0*dy)**2/2/(sigx*dx)**2))

# Defining DOE phase
#DOE = np.load('DOE_data.npy')
DOE = np.random.rand(Nx,Ny)*2*np.pi-np.pi

#----------------------------------------Other parameters
# external interation number
s = 30

# Create an empty list to store frames
frames = []

# costType: 1 = simple cost function(Ct2), 2 = smoothing neighbor pixels(Cs), 3 = alternating Ct4 / Cs, 4 = alternating Ct2 / Cs, 5 = Ct4 / Ct2
costType = 1
learning_rate=0.005 # Optimizer's learn rate

for t in range(s):
    start_time = time.time() # Record iteration speed
    
    
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    












