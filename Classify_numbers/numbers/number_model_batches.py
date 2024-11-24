# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 20:34:03 2024

@author: pschw
"""




import sys
sys.path.append('C:/Daten/Peter/Studium/A_Programme_Hiwi/Projekte/AI-Models/Classify_polygons/numbers')
# sys.path.append('C:/Users/wq271/AAA_programming/Projects/AI_models/AI Models/Classify_polygons/numbers')
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt 
import statistics as stat
from load_data import get_data

import os 
# lab pc dir
# os.chdir('C:/Users/wq271/AAA_programming/Projects/AI_models/AI Models/Classify_polygons/')

# laptop dir
os.chdir('C:/Daten/Peter/Studium/A_Programme_Hiwi/Projekte/AI-Models/Classify_polygons/')

path = 'C:/Daten/Peter/Studium/A_Programme_Hiwi/Projekte/AI-Models/Classify_polygons/numbers/'

'''TODO:
    - check out problem with minimizing to value around 1 and also increasing oscillations at 10k samples e.g.'''
      
     
### training params ###       
     
sample_size =1000

batch_size = 5


### get data info from target csv ###

# pc dir numbers
samples, y = get_data('numbers/data', sample_size, 'training')
                  

### introduce params from files ###

# introduce weight matricies
W1 = pd.read_csv('params/W1.txt')
W2 = pd.read_csv('params/W2.txt')
W3 = pd.read_csv('params/W3.txt')

W1 = np.array(W1)
W2 = np.array(W2)
W3 = np.array(W3)

# introduce bias vectors: LAPTOP DIR
bias1 = pd.read_csv('params/bias1.txt')
bias2 = pd.read_csv('params/bias2.txt')
bias3 = pd.read_csv('params/bias3.txt')

bias1 = np.array(bias1['0'])
bias2 = np.array(bias2['0'])
bias3 = np.array(bias3['0'])

# params = np.hstack((W1.flatten(), W2.flatten(), W3.flatten(), bias1.flatten(), bias2.flatten(), bias3.flatten()))
# print(params.size) # this model has 16740 params

batch_cost = []

weight_matricies = [W3, W2, W1]

bias_vectors = [bias3, bias2, bias1]

batch_dC_dw = [np.zeros((len(weight_matricies[i]), len(weight_matricies[i][0]))) for i in range(len(weight_matricies))]
batch_dC_db = [np.zeros(len(bias_vectors[i])) for i in range(len(bias_vectors))] 

zero_matricies = batch_dC_dw
zero_vectors   = batch_dC_db

# print([x.shape for x in batch_dC_dw])
# print([x.shape for x in batch_dC_db])


### init settings ### 

learning_rate = 0.01

weights = weight_matricies

biases = bias_vectors 

datapoints = [[], []]

### process image ###
# take index of png file and return image_array for png
def process_image(sample):
    global image_array
    # convert pixelinfo to DataFrame 
    image_frame = pd.DataFrame(sample)

    # image_frame.to_csv('C:/Users/wq271/AAA_programming/Projects/AI_models/AI Models/Classify_polygons/image_frame.csv', index = False)
    
    # flattened 1024 x 1 array with 0 and 1 
    image_array = np.array(image_frame).flatten() # with 1/4 of original res: this array has 1024 entries 
    
    # scale brightness to between 0 and 1
    image_array = 1/255 * image_array
    
    return image_array



### functions ###
def Cost(y_pred, y_true):
    return np.sum((y_pred-y_true)**2)

def Cost_prime(y_pred, y_true):
    return 2*(y_pred-y_true)

def sig(x):
    return 1 / (1 + np.exp(-x))

def sig_prime(x):
    return x * (1 - x)



### advanced methods ###
def cross_entropy(y_true, y_pred):
    epsilon = 1e-12  # Avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
    
def softmax(u):
    exp_x = np.exp(u - np.max(u))  
    return exp_x / np.sum(exp_x, axis=0)  

def relu(x):
    return np.maximum(0, x)

def relu_prime(x):
    return np.where(x > 0, 1, 0)



def forwards(image_array, weights, biases):
    ### calculation forwards step ###
    global cost
    # print('weights forward', weights[0][0][0])
    # first hidden layer created by image_array input layer and weight matrix W1 + bias1
    layer1 = sig(np.dot(weights[2], image_array) + biases[2])
    
    # second hidden layer created by output from first layer, weight matrix W2 and bias2
    layer2 = sig(np.dot(weights[1], layer1) + biases[1])
    
    # output layer from second hidden layer and weight matrix W3 
    output = softmax(np.dot(weights[0], layer2) + biases[0])

    # calculate cost of forward step
    cost = Cost(y[k], output)
    
    # print(cost)
    
    # store costs of batch in list for average cost of batch
    batch_cost.append(cost)

    if len(batch_cost) == batch_size:
        print(f'average cost batchsize {batch_size}', stat.mean(batch_cost))
    
    layers = [output, layer2, layer1, image_array]
    # nodes:    10       16      16        784
    
    return layers      
    

def backwards(layers, weights):
    ### calculations backwards step ###
    global batch_dC_dw, batch_dC_db
    ## calculation of dC/dlayer ##
    
    # dC_doutput
    dC_doutput = Cost_prime(layers[0], y[k])
    
    # dC_dlayer2 / dC_dlayer1
    dC_dlayer2 = []
    dC_dlayer1 = []
    
    # array that holds derviatives with respect to layer nodes 
    dC_dlayer = [dC_doutput, dC_dlayer2, dC_dlayer1]
    
    for l in range(len(dC_dlayer)-1):
        for i in range(len(layers[l+1])):
            dC_dlayer_i = sum([weights[l][j][i] * sig_prime(layers[l])[j] * dC_dlayer[l][j] for j in range(len(layers[l]))])
            dC_dlayer[l+1].append(dC_dlayer_i)
        
    dC_dlayer = [dC_doutput, np.array(dC_dlayer2), np.array(dC_dlayer1)]
    
    # print([x.shape for x in dC_dlayer])
    
    
    ## calculation of dC_dw ##
    
    dC_dw = [np.zeros((len(layers[i]), len(layers[i+1]))) for i in range(len(weights))] #TODO: include W1 in backprop?!

    
    for l in range(len(dC_dw)):
        for i in range(len(layers[l])):
            for j in range(len(layers[l+1])):
                dC_dw[l][i][j] = layers[l+1][j] * sig_prime(layers[l])[i] * dC_dlayer[l][i]
                
    for l in range(len(dC_dw)):
        batch_dC_dw[l] += dC_dw[l] 
    
    
            
    ## calculation of dC_db ##
    
    dC_db = [np.zeros(len(layers[i])) for i in range(len(layers)-1)]
    # print([x.shape for x in dC_db])
    
    
    for l in range(len(dC_db)):
        for i in range(len(dC_db[l])):
            dC_db_i = sig_prime(layers[l])[i] * dC_dlayer[l][i]
            dC_db[l][i] = dC_db_i
    
    for l in range(len(dC_db)):
        batch_dC_db[l] += dC_db[l] 
    
        
        
def gradient_descent(batch_dC_dw, batch_dC_db, learning_rate):
    global weights, biases 
    ### make stochastic gradient descent for weight adjustment ###
    # print('shapes of gradient objects',[x.shape for x in dC_dw], [x.shape for x in dC_db], 
    #       'shapes of original matricies', [x.shape for x in weight_matricies], [x.shape for x in bias_vectors]) 
        
    # gradient descent for weights 
    for l in range(len(weights)):
        weights[l] -= learning_rate * (1/batch_size*batch_dC_dw[l])
        
    for l in range(len(biases)):
        biases[l] -= learning_rate * (1/batch_size*batch_dC_db[l])
        
    return weights, biases 



for k,sample in enumerate(samples):
    process_image(sample)
    layers = forwards(image_array, weights, biases)
    backwards(layers, weights)
    if (k+1)%batch_size == 0:
        weights, biases = gradient_descent(batch_dC_dw, batch_dC_db, learning_rate)
        datapoints[0].append(k)
        datapoints[1].append(cost)
        batch_dC_dw = zero_matricies
        batch_dC_db = zero_vectors
        batch_cost = []

        
plt.scatter(datapoints[0], datapoints[1], color = 'red')
plt.xlabel('samples')
plt.ylabel('cost')
plt.show() 

#%%

for i,matrix in enumerate(weights):
    print(matrix.shape)
    matrix = pd.DataFrame(matrix)
    matrix.to_csv(path+f'W{len(weights)-i}_trained.csv',mode='w', index=False)
    
for i,vector in enumerate(biases):
    print(vector.shape)
    vector = pd.DataFrame(vector)
    vector.to_csv(path+f'b{len(biases)-i}_trained.csv',mode='w', index=False)

#%%
# from number_model_v1_1 import process_image

test_size = 5000

# test model
W1 = pd.read_csv(path+'W1_trained.csv')
W2 = pd.read_csv(path+'W2_trained.csv')
W3 = pd.read_csv(path+'W3_trained.csv')

# introduce weight matricies
# W1 = pd.read_csv('params/W1.txt')
# W2 = pd.read_csv('params/W2.txt')
# W3 = pd.read_csv('params/W3.txt')

W1 = np.array(W1)
W2 = np.array(W2)
W3 = np.array(W3)

# test model
bias1 = pd.read_csv(path+'b1_trained.csv')
bias2 = pd.read_csv(path+'b2_trained.csv')
bias3 = pd.read_csv(path+'b3_trained.csv')

#ref model
# bias1 = pd.read_csv('params/bias1.txt')
# bias2 = pd.read_csv('params/bias2.txt')
# bias3 = pd.read_csv('params/bias3.txt')

bias1 = np.array(bias1['0'])
bias2 = np.array(bias2['0'])
bias3 = np.array(bias3['0'])

test, y = get_data('numbers/data', test_size, 'test')

weights_test = [W3, W2, W1]
biases_test = [bias3, bias2, bias1]



counter_correct = 0 


### process image ###
# take index of png file and return image_array for png
def process_image(sample):
    global image_array
    # convert pixelinfo to DataFrame 
    image_frame = pd.DataFrame(sample)

    # LAPTOP DIR
    image_frame.to_csv('C:/Daten/Peter/Studium/A_Programme_Hiwi/Projekte/AI-Models/Classify_polygons/image_frame.csv', index = False)
    
    # flattened 1024 x 1 array with 0 and 1 
    image_array = np.array(image_frame).flatten() # with 1/4 of original res: this array has 1024 entries 
    
    # scale brightness to between 0 and 1
    image_array = 1/255 * image_array
    
    return image_array


def test_forward(image_array, weights, biases):
    global counter_correct
    ### calculation forwards step ###
    # first hidden layer created by image_array input layer and weight matrix W1 + bias1
    layer1 = sig(np.dot(weights_test[2], image_array) + biases_test[2])
    
    # second hidden layer created by output from first layer, weight matrix W2 and bias2
    layer2 = sig(np.dot(weights_test[1], layer1) + biases_test[1])
    
    # output layer from second hidden layer and weight matrix W3 
    output = softmax(np.dot(weights_test[0], layer2) + biases_test[0])
    
    guess = np.where(output == np.max(output))[0][0]

    solution = np.where(y[k] == 1)[0][0]
    
    print(f'guess: {guess} | solution: {solution}')
    
    if guess == solution:
        counter_correct += 1


for k,sample in enumerate(test):
    process_image(sample)
    layers = test_forward(image_array, weights_test, biases_test)

print('the model has the accuracy:', counter_correct/test_size)













