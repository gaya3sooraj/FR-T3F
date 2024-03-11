import cv2

import tensorflow as tf
import t3f

import os
import glob
import time

import random
import numpy as np

from datasetInfo import db_info
import dataPreprocessing as dp

from sklearn.metrics import accuracy_score


def classificationModel(tt_outputs, info=True):
   
    dataset_name = tt_outputs[3]  
    db = db_info.get(dataset_name)   

    g1 = tt_outputs[0]
    g2 = tt_outputs[1]
    g3 = tt_outputs[2]
    
    file_ext = db['file_ext'] 
    path = db['test_path']
    files = glob.glob(os.path.join(path, file_ext))  
    
    random.shuffle(files)
    
    labels = []
    label_hats = []

    Ne = db['d2']
    Np = db['d3'] 

    cpu_time = []

    if info == True:
        print("TESTING T3F FR MODEL.....")
        print("Size of test dataset: ", len(files))

    for file in files:
        
        start_time = time.process_time() 
        distance = np.zeros((Ne, Np))
        
        label = db['name_pattern'](file)  
        labels.append(int(label))

        image = dp.readImage(file)        
        z = np.ravel(image, order='F')
        z_hat = np.dot(g1.numpy().T, z)     

        for e in range(Ne):
    
            G = g2[:, e, :].numpy()
            Qe, Re = np.linalg.qr(G, mode='reduced')
            A = Re
            B = np.dot(Qe.T, z_hat)
            al = np.linalg.lstsq(A, B, rcond=None)[0]
    
            for p in range(Np):
                distance[e, p] = np.linalg.norm(al - g3[:, p])

        d = np.min(distance, axis=0)      
        
        #Finding label_hat
        label_hat = np.argmin(d) + 1
        label_hats.append(label_hat)     

        end_time = time.process_time() 

        cpu_time.append(end_time - start_time)   

    if info == True:
        print("\n\n_________________________________________\n\n")
        print("Training Completed!")
        print("\n\n_________________________________________\n\n")
        print("Accuracy of the T3F model: ", accuracy_score(labels, label_hats))
        print("Average CPU time to classify an unknown image: ", sum(cpu_time)/len(cpu_time))
    
    return accuracy_score(labels, label_hats), sum(cpu_time)/len(cpu_time)

    
    
  