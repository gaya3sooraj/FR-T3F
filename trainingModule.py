#import libraries
import tensorflow as tf
import numpy as np
import time
import sys

#import tensor train library t3f
import t3f

#import dataset and data processing function
import dataPreprocessing as dp
from datasetInfo import db_info

################### MULTI DIMENSIONAL ARRAY TO TENSOR #####################
def trainingDataToTensor(files, d1, d2, d3):
    images = []  
    
    for file_path in files:
        image = dp.readImage(file_path) 
        image = np.ravel(image, order='F')
        images.append(image) 

    images = np.array(images)  
   
    images = images.reshape(d3, d2, d1)
    image_tensor = tf.convert_to_tensor(images)
    train_tensor = tf.transpose(image_tensor, perm=[2,1,0])
    
    return train_tensor

################### IMAGE TENSOR TO TT-CORES #####################

def tensorToTT(train_tensor, R):

    image_tensor_tt = t3f.to_tt_tensor(train_tensor, max_tt_rank=R)
    g1=image_tensor_tt.tt_cores[0]
    g2=image_tensor_tt.tt_cores[1]
    g3=image_tensor_tt.tt_cores[2]
    if len(g1.shape)>2 and g1.shape[0]==1:
        g1 = tf.reshape(g1, [g1.shape[1], g1.shape[2]])
    g1 = -g1
    g2 = -g2
    if len(g3.shape)>2 and g3.shape[2]==1:
        g3 = tf.reshape(g3, [g3.shape[0], g3.shape[1]])
    
    return g1, g2, g3

################### TRAINING MODEL #####################

def trainingModel(dataset_name, rank=None, info=True):    
    
    files = dp.loadTrainFiles(dataset_name)
    db = db_info.get(dataset_name)

    d1 = db['d1']
    d2 = db['d2']
    d3 = db['d3']

    if info==True:
        print("DATASET INFO: ")
        print(" -> No of pixels in each image: ", d1)
        print(" -> No of subjects: ", d3)
        print(" -> No of images of each subject: ", d2)

    train_tensor = trainingDataToTensor(files, d1, d2, d3)
    
    if info==True:
        print("\n\n_________________________________________\n\n")
        print("Array of images has been successfully converted to a tensor using TensorFlow!")
        print("Size of the image tensor: ", train_tensor.shape)
    
    if rank == None:
        if info==True:
            print("\n\n_________________________________________\n\n")
            print("No rank entered. Model is trained with max rank: ", d2*d3)
        R = d2 * d3
    elif rank > (d2*d3): 
        print("\n\n_________________________________________\n\n")
        print("Wrong rank entered! Enter rank below or equal to max rank: ", d2*d3)
        sys.exit("Error: Invalid Rank value.")
    elif rank < 30 and rank > 0:
        if info==True:
            print("\n\n_________________________________________\n\n")
            print("Rank value too low. Accuracy of the model could be low or the model could run into an error.")
        R = rank
    elif rank == 0:
        print("\n\n_________________________________________\n\n")
        print("Rank value can't be 0.")
        sys.exit("Error: Invalid Rank value.")
    else:
        R = rank
                 
        
    start_time = time.process_time() 
    g1, g2, g3 = tensorToTT(train_tensor, R)
    end_time = time.process_time()  
    
    cpu_time = end_time - start_time
    
    if info==True:
        print("\n\n_________________________________________\n\n")
        print("Training completed!")
        print("Time taken to train the model: ", cpu_time)
    
    return [g1, g2, g3, dataset_name, cpu_time]

