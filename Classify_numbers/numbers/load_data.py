# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 13:28:46 2024

@author: pschw
"""

#
# Verify Reading Dataset via MnistDataloader class
# #
import random
from os.path  import join
from Mnist_dataloader import MnistDataloader
import numpy as np

#
# Set file paths based on added MNIST Datasets
#

def get_data(input_path, samplesize, mode):
    # get file dirs 
    training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
    training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
    test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
    test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')
    
    # load the data 
    mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()
    
    # pull random samples
    samples = []
    
    # get control vectors y
    y = []
    
    if mode == 'train':
        for r in np.random.randint(0, 60000, samplesize):
            samples.append(x_train[r])
            number = np.zeros(10)
            number[y_train[r]] = 1
            y.append(number)
            
    elif mode == 'test':
        for r in np.random.randint(0, 10000, samplesize):
            samples.append(x_test[r])
            number = np.zeros(10)
            number[y_test[r]] = 1
            y.append(number)
        
    return samples, y

    
        