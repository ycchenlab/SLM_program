# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 19:50:58 2023

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt
#from numpy.fft import fft2, ifft2
#from function_test import costfunction
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

oNx = 200 # total = 1296
oNy= 200 # total = 964
odx = 3.75e-3 # (mm)
ody = 3.75e-3 # (mm)
oLx = oNx * odx # (mm)
oLy = oNy * ody # (mm)
ox = np.linspace(-oLx/2, oLx/2, oNx)
oy = np.linspace(-oLy/2, oLy/2, oNy)
oX, oY = np.meshgrid(ox, oy)


# -------------------------Target, Input and Diffraction plane
tran = 1/(2*np.sqrt(2*np.log(2)))

# Super-Gaussian profile parameters
m = 5
A = 1  # Super-Gaussian amplitude
sigx = 500e-3*tran/odx  # pixel
sigy = 100e-3*tran/ody  # pixel
x0 = 0 # pixel
y0 = 0 # pixel
# Generate super-Gaussian profile -- it is **amplitude**
Rx = (oX-x0)**2/2/(sigx*odx)**2
Ry = (oY-y0)**2/2/(sigy*ody)**2
R = Rx + Ry
#target = np.exp(-R**m)
target = np.exp(-Rx**m) * np.exp(-Ry**m)
targetphase = np.zeros([oNx, oNy])

ret, binary = cv2.threshold(target, 0.01*A, 1*A, cv2.THRESH_BINARY)
rg = binary/A/A # the range of costfunction selected
rga = rg

#rg = np.ones([int(np.size(oy)*3/4), int(np.size(ox)*3/4)])
#rg = resize(rg, ox, oy)
bg = np.ones([np.size(oy), np.size(ox)]) - rg

#target = target*rg  # Normalize matrix
#targetphase = targetphase*rg

# Calculate Gaussian beam profile -- it is electric field **amplitude**

sigx = 300.4*odx*tran*np.sqrt(2)/dx  # pixel # 127.5e-6 m
sigy = 321.2*ody*tran*np.sqrt(2)/dy  # pixel # 136.4e-6 m
x0 = 0 # pixel
y0 = 0 # pixel
A = 1

initial_profile = A*np.exp(-((X-x0*dx)**2/2/(sigx*dx)**2+(Y-y0*dy)**2/2/(sigy*dy)**2))
plot(x, y, initial_profile, 'Greens')
plot(ox, oy, target, 'Greens')
# Defining DOE phase
DOE = np.load('DOE_data.npy')
#DOE = np.random.rand(Ny,Nx)*2*np.pi
#DOE = np.zeros((Ny, Nx))

#----------------------------------------Other parameters

s = 50 # external interation number
f = 250 # focal length
l = 895e-6 # lambda
factor = f*l


# Create an empty list to store frames
frames = []

# costType: 1 = simple cost function(Ct2), 2 = smoothing neighbor pixels(Cs), 3 = alternating Ct4 / Cs, 4 = alternating Ct2 / Cs, 5 = Ct4 / Ct2
costType = 1
learning_rate=0.005 # Optimizer's learn rate

for t in range(s):
    start_time = time.time() # Record iteration speed
    
    #------------------------Fourer transform
    DOEphase = np.exp(1j * DOE) # SLM phase
    iterf = ft(x, y, ox, oy, initial_profile * DOEphase, f, l) # Fourier transform
    intf = np.square(np.abs(iterf)) # normalized training intenstiy
    angf = np.angle(iterf) # normalized training angle
    
    aphase = np.exp(1j * angf)
    
    plot(ox, oy, intf, 'Blues')
    
    iterb = ft(oy, ox, y, x, target * aphase, f, l)
    angb = np.angle(iterb)
    DOE = angb
    
    #------------------------Error calculation
    error = target - intf # Calculate error
    E = np.sum(np.abs(error)) / np.sum(rg) # mean error for break iteration
    squaredDifferences = error**2 # error square
    meanSquaredDifferences = np.sum(squaredDifferences)/np.sum(rg) 
    rmse = np.sqrt(meanSquaredDifferences)
    
    
    ############################ Optimization funciton
    #cost, DOE_tf, learning_rate, optimizer_string = costfunction(DOE, target, initial_profile, N,t,learning_rate, costType,squaredDifferences, targetphase, rg)
    #DOE = DOE_tf.numpy() 
    
    
    if E < 0.000005:
        iteration = t
        break
    
    # End the timer
    end_time = time.time()
    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    
    #text = f'Iteration: {t+1}\nRMSE: {round(rmse, 4)}\nElapsed time: {round(elapsed_time, 2)} seconds\n'
    #print(text)
    plt.imshow(intf)
    text = f'Iteration: {t+1}\nRMSE: {round(rmse, 4)}\nElapsed time: {round(elapsed_time, 2)} seconds'
    plt.annotate(text, xy=(0.05, 0.8), xycoords='axes fraction', color='black', fontsize=7, weight='bold')
    
    
    save_path = r'C:\Users\user\git repo\SLM_program\fourier transform\tempPNG\\'
    filename = f'plot_{t}.png'
    plt.savefig(save_path + filename, dpi = 300)
    plt.close()
converter()
plot(ox, oy, intf, 'Blues')
    

# save DOE data for next time use
np.save('DOE_data.npy', DOE)
fig, ax = plt.subplots()
plt.title("SLM plane")
plt.imshow(DOE)
DOE = (DOE - np.min(DOE))%(2*np.pi)*255/2/np.pi
plt.imsave('DOE.png', DOE, cmap='gray')

xsum = np.sum(intf)










