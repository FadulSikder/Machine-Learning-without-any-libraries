#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import csv


# In[8]:


PATH = os.getcwd() + "\\"
DATA = "data_XOR.csv"


# In[9]:


def load_data():
    
    data_path = PATH + DATA
    df = pd.read_csv(data_path, header = None)
    data = df.to_numpy()
    
    X = data[:,:-2]
    bias = data[:,3:4]
    Y = data[:,-1:]
    
    # Replace -1 with 0 for sigmoid function
    X = np.where(X == -1, 0, X)
    Y = np.where(Y == -1, 0, Y)
    
    return X, Y, bias


# In[10]:


def plot_fn(incorrect_data):
    plt.plot(incorrect_data)
    plt.title(DATA)
    plt.xlabel('Iterations')
    plt.ylabel('Misclassified data')
    plt.show()


# In[11]:


# Including bias
# Layer 1 - 4 units
# Layer 2 - 2 units
class FCNN(object):
    def __init__(self, x, y, bias, layer1_unit, layer2_unit):
        self.x = x
        self.y = y
        self.iter = 100
        self.incorrect = np.zeros((self.iter, 1))
        
        # Add bias to input
        self.x = np.column_stack((bias, self.x))
        
        input_size = self.x.shape[1]
        output_size = 1
        
        # Randomly initialising weights
        #np.random.seed(1)
        # weights1 = (4*4)
        # Weights2 = (4*1)
        self.weights1 = np.random.uniform(low=-0.1, high=0.1, size=(input_size,layer1_unit))
        self.weights2 = np.random.uniform(low=-0.1, high=0.1, size=(layer1_unit,output_size))
        
        # FCNN
        self.train(self.x, self.y, self.weights1, self.weights2)
        plot_fn(self.incorrect)
    
    ## --- Sigmoid Function --- ##
    def sigmoid(self, g):
        return 1 / (1 + np.exp(-2 * g))

    def sigmoid_gradient(self, g):
        return g * (1 - g)
        
    def predict_fn(self, a):
        y = (a > 0.5).astype(int)
        return y
    
    def calculate_misclassified(self, pred_y):
        length = len(pred_y)
        correct = self.y == pred_y
        incorrect = length - np.sum(correct) 
        return incorrect
    
    def feed_forward(self, x, weights1, weights2):
        z2 = np.dot(x, weights1)
        a2 = self.sigmoid(z2)

        z3 = np.dot(a2, weights2)
        a3 = self.sigmoid(z3)
        
        return a2, a3
    
    def back_propagation(self, x, y, a2, a3, weights1, weights2):
        error_a3 = y - a3
        error_a2 = np.dot(error_a3, weights2.T) * self.sigmoid(np.dot(x, weights1))
        
        delta_a3 = error_a3 * self.sigmoid_gradient(a3)
        delta_a2 = error_a2 * self.sigmoid_gradient(a2)

        # Update weights
        weights2 += np.dot(a2.T, delta_a3)
        weights1 += np.dot(x.T, delta_a2)
        
        return weights1, weights2
            
    def train(self, x, y, weights1, weights2):
        for i in range(self.iter):
            # Feed forward
            a2, a3 = self.feed_forward(x, weights1, weights2)
            
            # Back-propagation
            weights1, weights2 = self.back_propagation(x, y, a2, a3, weights1, weights2)
            
            # Predict
            y_pred = self.predict_fn(a3)
            self.incorrect[i] = self.calculate_misclassified(y_pred)
            #print(self.incorrect[i])
         
        #print(a3)


# In[12]:


def main():
    
    X, Y, bias = load_data()
    
    layer1_unit = 4
    layer2_unit = 1
    FCNN(X, Y, bias, layer1_unit, layer2_unit)

if __name__ == '__main__':
    main()


# In[ ]:





# In[ ]:




