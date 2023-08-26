# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 18:02:40 2023

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

oNx = 300 # total = 1296
oNy= 300 # total = 964
odx = 3.75e-3 # (mm)
ody = 3.75e-3 # (mm)
oLx = oNx * odx # (mm)
oLy = oNy * ody # (mm)
ox = np.linspace(-oLx/2, oLx/2, oNx)
oy = np.linspace(-oLy/2, oLy/2, oNy)
oX, oY = np.meshgrid(ox, oy)


# -------------------------Target, Input and Diffraction plane

# Super-Gaussian profile parameters
mx = 5  # Super-Gaussian exponent
my = 5
A = 1  # Super-Gaussian amplitude
tran = 1/(2*np.sqrt(2*np.log(2)))
sigx = 300.4e-3*tran/odx  # pixel
sigy = 321.2e-3*tran/ody  # pixel
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
plot(ox, oy, target, 'Blues')
# Defining DOE phase
#DOE = np.load('DOE_data.npy')
#DOE = np.random.rand(Ny,Nx)*2*np.pi
DOE = np.zeros((Ny, Nx))

#----------------------------------------Other parameters

s = 200 # external interation number
f = 110 # focal length
l = 895e-6 # lambda
factor = f*l

after_phase = np.exp(1j * np.zeros([oNy, oNx])) # output plane
before_phase = np.exp(1j * np.zeros([Ny, Nx])) # SLM plane

DOEphase = np.exp(1j * DOE)
before_phase += DOEphase

after_f = ft(x, y, ox, oy, initial_profile * DOEphase, f, l) # ft

#s = 300

for i in range(s):
    start_time = time.time() # Record iteration speed
    
    after_ampf = np.abs(after_f)
    after_angf = np.angle(after_f)
    after_ampe = (targetamp - after_ampf)
    after_phase += after_angf    
    
    before_e = ft(oy, ox, y, x, after_ampe * after_angf, f, l) # ift
    before_ampe = np.abs(before_e)
    before_ange = np.angle(before_e)
    before_ampe2 = initial_profile - before_ampe
    before_phase = before_ange

    #-------------------------------------------------
    
    after_f = ft(x, y, ox, oy, before_ampe2 * before_ange, f, l)
    '''
    after_ampf = np.abs(after_f)
    after_angf = np.angle(after_f)
    after_ampf = targetamp - after_ampf
    after_phase += after_angf
    '''
    
    #------------------------Error calculation
    error = target - np.square(after_ampf) # Calculate error
    E = np.sum(np.abs(error)) / np.sum(rg) # mean error for break iteration
    squaredDifferences = error**2 # error square
    meanSquaredDifferences = np.sum(squaredDifferences)/np.sum(rg) 
    rmse = np.sqrt(meanSquaredDifferences)
    
    # End the timer
    end_time = time.time()
    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    
    a = ft(x, y, ox, oy, initial_profile * before_phase, f, l)
    
    plt.imshow(np.square(np.abs(a)))
    text = f'Iteration: {i+1}\nRMSE: {round(rmse, 4)}\nElapsed time: {round(elapsed_time, 2)} seconds'
    plt.annotate(text, xy=(0.05, 0.8), xycoords='axes fraction', color='black', fontsize=7, weight='bold')
    
    
    save_path = r'C:\Users\user\git repo\SLM_program\fourier transform\tempPNG\\'
    filename = f'plot_{i}.png'
    plt.savefig(save_path + filename, dpi = 300)
    plt.close()
converter()

DOE = np.angle(before_phase)
# save DOE data for next time use
np.save('DOE_data.npy', DOE)
fig, ax = plt.subplots()
plt.title("SLM plane")
plt.imshow(DOE)
DOE = (DOE - np.min(DOE))%(2*np.pi)*255/2/np.pi

plt.imsave('DOE.png', DOE, cmap='gray')

fig, ax = plt.subplots()
plt.title("SLM plane")
plt.imshow(DOE)




























