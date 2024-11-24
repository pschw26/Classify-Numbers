# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 18:12:41 2024

@author: wq271
"""

import pandas as pd 
import numpy as np


### init params ###
# weight matricies
W1 = np.random.randn(16, 784)   # Small random values for W1
W2 = np.random.randn(16, 16)     # Adjust W2 to match hidden layer sizes
W3 = np.random.randn(10, 16)    # Assuming 10 output classes


# bias vectors
bias1 = np.random.randn(16)   # Small random values for W1
bias2 = np.random.randn(16)     # Adjust W2 to match hidden layer sizes
bias3 = np.random.randn(10)     # Assuming 10 output classes


df1 = pd.DataFrame(W1)
df2 = pd.DataFrame(W2)
df3 = pd.DataFrame(W3)
df4 = pd.DataFrame(bias1)
df5 = pd.DataFrame(bias2)
df6 = pd.DataFrame(bias3)

# LAPTOP DIR
df2.to_csv('C:/Daten/Peter/Studium/A_Programme_Hiwi/Projekte/AI-Models/Classify_polygons/params/W2.txt', index = False)
df3.to_csv('C:/Daten/Peter/Studium/A_Programme_Hiwi/Projekte/AI-Models/Classify_polygons/params/W3.txt', index = False)
df4.to_csv('C:/Daten/Peter/Studium/A_Programme_Hiwi/Projekte/AI-Models/Classify_polygons/params/bias1.txt', index = False)
df1.to_csv('C:/Daten/Peter/Studium/A_Programme_Hiwi/Projekte/AI-Models/Classify_polygons/params/W1.txt', index = False)
df5.to_csv('C:/Daten/Peter/Studium/A_Programme_Hiwi/Projekte/AI-Models/Classify_polygons/params/bias2.txt', index = False)
df6.to_csv('C:/Daten/Peter/Studium/A_Programme_Hiwi/Projekte/AI-Models/Classify_polygons/params/bias3.txt', index = False)

# LAB PC DIR
# df2.to_csv('C:/Users/wq271/AAA_programming/Projects/AI_models/AI Models/Classify_polygons/params/W2.txt', index = False)
# df3.to_csv('C:/Users/wq271/AAA_programming/Projects/AI_models/AI Models/Classify_polygons/params/W3.txt', index = False)
# df4.to_csv('C:/Users/wq271/AAA_programming/Projects/AI_models/AI Models/Classify_polygons/params/bias1.txt', index = False)
# df1.to_csv('C:/Users/wq271/AAA_programming/Projects/AI_models/AI Models/Classify_polygons/params/W1.txt', index = False)
# df5.to_csv('C:/Users/wq271/AAA_programming/Projects/AI_models/AI Models/Classify_polygons/params/bias2.txt', index = False)
# df6.to_csv('C:/Users/wq271/AAA_programming/Projects/AI_models/AI Models/Classify_polygons/params/bias3.txt', index = False)











