# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 22:03:55 2023

@author: ycche
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import os
import imageio
from datetime import datetime


def costfunction(DOE, target, initial_profile, Nx, Ny, t, LR, costType, squaredDifferences,targetphase, costRG):
 # Set device placement for TensorFlow operations
 

    
# turn numpy objects into tensor (preparation for deep-learning-based optimization methods)
    complex_one = tf.constant(1j, dtype=tf.complex64)
    DOE_tf = tf.convert_to_tensor(DOE, dtype=tf.complex64)
    target_tf = tf.convert_to_tensor(target, dtype=tf.float32)
    initial_profile_tf = tf.convert_to_tensor(initial_profile, dtype=tf.complex64)
    targetphase_tf = tf.convert_to_tensor(targetphase, dtype=tf.float32)
    costRG_tf = tf.convert_to_tensor(costRG, dtype = tf.float32)
    
# initialize the optimization parameters
    DOE_tf = tf.math.real(DOE_tf)
    variables = tf.Variable(DOE_tf)
    #variables = tf.math.multiply(tf.Variable(DOE_tf),costRG_tf)
    
        
    learning_rate=LR
    optimizer = tf.optimizers.Adam(learning_rate)

    '''
    intf = tf.abs(iterf) / tf.reduce_max(tf.abs(iterf)) # normalized training intensity 
    '''
    cost = tf.reduce_sum(squaredDifferences)
    cost_values = []
    plt.figure()
    

    
    def costFnSmoothing (variables):
        # Pad the tensor to handle boundaries
        
        padded_tensor = tf.pad(tf.square(tf.abs(tf.signal.fft2d(initial_profile_tf * tf.exp(complex_one * tf.cast(variables, tf.complex64)))))/ tf.reduce_max(tf.square(tf.abs(tf.signal.fft2d(initial_profile_tf * tf.exp(complex_one * tf.cast(variables, tf.complex64)))))), [[2, 2], [2, 2]])  # Pad with 2 extra rows and columns

        input_tensor = tf.constant(padded_tensor)  # Your 500x500 tensor

        # Select the center element and adjacent elements
        center_elements = input_tensor[1:-1, 1:-1]
        up_elements = input_tensor[:-2, 1:-1]
        down_elements = input_tensor[2:, 1:-1]
        left_elements = input_tensor[1:-1, :-2]
        right_elements = input_tensor[1:-1, 2:]
        
        
        
        # Calculate the squared difference between center element and adjacent elements
        squaredDiff = tf.square(tf.square(center_elements) - tf.square(up_elements)) + tf.square(tf.square(center_elements) - tf.square(down_elements)) + tf.square(tf.square(center_elements) - tf.square(left_elements)) + tf.square(tf.square(center_elements) - tf.square(right_elements))
        
        # Sum up the squared differences for all elements
           
        smoothnessInfo = tf.reduce_sum(squaredDiff)
        # smoothnessInfo += feature_difference
        return smoothnessInfo, squaredDiff
    
    def conjugategrad (variables):
        d = 3
        DOEphase = tf.exp(complex_one * tf.cast(variables, tf.complex64))
        #iterf_tf = tf.math.multiply(tf.signal.fft2d(initial_profile_tf *DOEphase),costRG_tf) # training output electric field
        iterf_tf = tf.signal.fft2d(initial_profile_tf *DOEphase)
        intf_tf = tf.math.multiply(tf.square(tf.abs(iterf_tf)) / tf.math.reduce_max(tf.square(tf.abs(iterf_tf))), costRG_tf) # normalized training intenstiy
        angf_tf = tf.math.multiply(tf.math.angle(iterf_tf), costRG_tf)
        #print(tf.reduce_min(angf_tf))
        #print(tf.reduce_max(angf_tf))
        #angf_tf = angf_tf -tf.reduce_min(angf_tf)
        constant = tf.reduce_sum(tf.sqrt(tf.multiply(target_tf, target_tf)))
        #print(variables)
        costfun = (10**d)*((1-tf.reduce_sum(tf.sqrt(tf.multiply(intf_tf, target_tf)))/constant)**2)
        
        return costfun, angf_tf
    
    def costFnEfficient (variables):
        
        
        return something
    

    def costFnSimple (variables):
        DOEphase = tf.exp(complex_one * tf.cast(variables, tf.complex64))
        #iterf_tf = tf.math.multiply(tf.signal.fft2d(initial_profile_tf *DOEphase),costRG_tf) # training output electric field
        iterf_tf = tf.signal.fft2d(initial_profile_tf *DOEphase)
        intf_tf = tf.math.multiply(tf.square(tf.abs(iterf_tf)) / tf.math.reduce_max(tf.square(tf.abs(iterf_tf))), costRG_tf) # normalized training intenstiy
        costfuns = tf.reduce_sum(tf.pow(tf.square(target_tf) - intf_tf, pp))
        #print(variables)
        return costfuns

    num_iterations = 30
    for i in range(num_iterations):
    
        # Compute the gradient using tf.GradientTape
        with tf.GradientTape() as tape:
            tape.watch(variables)
        
            # 1 = simple cost function(Ct2),           
            if costType==1:
                pp = 2
                cost = costFnSimple(variables)
                    
            # 2 = smoothing neighbor pixels(Cs),            
            elif costType==2:
                cost, squaredDiff = costFnSmoothing(variables)
                             
             # 3 = alternating Ct4 / Cs,      
            elif costType==3:
                if i%2 == 1:
                    pp = 4    
                    cost = costFnSimple(variables)               
                else:
                    cost, squaredDiff = costFnSmoothing(variables)

            # 4 = alternating Ct2 / Cs, 
            elif costType==4:
                if i%2 == 1:
                    pp = 4    
                    cost = costFnSimple(variables)               
                else:
                    cost, squaredDiff  = costFnSmoothing(variables)

            # 5 = Ct4 / Ct2
            elif costType==5:
                if i%2 == 1:
                    pp = 4    
                    cost = costFnSimple(variables)               
                else:
                    pp = 2    
                    cost = costFnSimple(variables)   
                    
            elif costType==6:
                cost, con_angle = conjugategrad(variables)
                print(con_angle)
                        
            gradients = tape.gradient(cost, variables)
            gradients = tf.reshape(gradients,(Nx,Ny))
            print(gradients)
            
    # Perform optimization
        optimizer.apply_gradients([(gradients, variables)])
        cost_values.append(cost.numpy())
        print (cost.numpy())
        
        
    optimizer_string = str(optimizer)
    
    plt.plot(range(1, num_iterations + 1), cost_values)
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Cost Function')
    plt.annotate(f'Learning Rate: {learning_rate}\nIteration: {t+1}', xy=(0.05, 0.9), xycoords='axes fraction')
    plt.show()

    return cost, variables, learning_rate, optimizer_string


def get_file_creation_time(file_path):
    # Get the creation time of a file
    timestamp = os.path.getmtime(file_path)
    return datetime.fromtimestamp(timestamp)


def converter():
        # Folder path containing the images
    folder_path = 'C:/Users/user/git repo/SLM_program/fourier transform/tempPNG/'
    
    # List to store image file names
    image_files = []
    
    # Iterate through the files in the folder
    for file in os.listdir(folder_path):
        if file.startswith('plot_') and file.endswith('.png'):
            image_files.append(os.path.join(folder_path, file))
    
    # Sort the image files based on creation time
    image_files.sort(key=get_file_creation_time)
    
    # Create an empty list to store frames
    frames = []
    
    # Read the images and add them to the frames list
    for image_file in image_files:
        frame = imageio.imread(image_file)
        frames.append(frame)
    
    # Set the file path and name for the output video
    output_path = 'output.mp4'
    
    writer = imageio.get_writer(output_path, format='mp4', mode='I', fps=10)
    
    # Write the frames to the video
    for frame in frames:
        writer.append_data(frame)
    
    # Close the writer
    writer.close()
