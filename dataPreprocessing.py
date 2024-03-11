import cv2
import tensorflow as tf
import t3f
import os
import glob
import numpy as np
from datasetInfo import db_info


def loadTrainFiles(dataset_name):
    db = db_info.get(dataset_name)
    if db:
        file_ext = db['file_ext']
        path = db['train_path']
    else:
        print(f"Dataset '{dataset_name}' not found in datasets_info")

    files = glob.glob(os.path.join(path, file_ext))    
    sorted_files = sorted(files, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    return sorted_files


def readImage(file):
    image = cv2.imread(file)
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_matrix = image.astype(float) / 255.0
    return image_matrix

def loadFiles(path):
    
    file_ext = "*.jpg"    
    files = glob.glob(os.path.join(path, file_ext))    
    return files