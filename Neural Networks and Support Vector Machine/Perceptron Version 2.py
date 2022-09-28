#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import csv


# In[2]:


PATH = "os.getcwd() + "\\"
DATA = "data_OR.csv"


# In[3]:


def load_data():
    
    data_path = PATH + DATA
    df = pd.read_csv(data_path, header = None)
    data = df.to_numpy()
    
    X = data[:,:-2]
    bias = data[:,3:4]
    Y = data[:,-1:]
    
    return X, Y, bias


# In[4]:


def plot_fn(incorrect_data):
    plt.plot(incorrect_data)
    plt.title(DATA)
    plt.xlabel('Iterations')
    plt.ylabel('Misclassified data')
    plt.show()


# In[5]:


class Perceptron(object):
    def __init__(self, x, y):
        
        self.x = x
        self.y = y
        no_feature = x.shape[1]
        
        # Init weights to zero
        self.weight = np.zeros((no_feature, 1))
        
        # Init no of iteration and alpha
        self.iter = 100
        self.alpha = 0.1
        self.incorrect = np.zeros((self.iter, 1))
        
        # Perceptron Learning rule to update weights
        self.learning_rule_algo()
        plot_fn(self.incorrect)
    
    def activation_fn(self, a):
        a = (a > 0).astype(int) # If value greater than 0, then 1, otherwise 0
        a = np.where(a == 0, -1, a) # Replace 0 with -1
        return a
 
    def predict_fn(self, x):
        z = np.dot(self.x, self.weight) # sum(wx + b)
        a = self.activation_fn(z)
        return a
    
    def cost_fn(self, error):
        cost = 0.5 * np.sum(np.square(error))
        return cost
    
    def calculate_misclassified(self, pred_y):
        length = len(pred_y)
        correct = self.y == pred_y
        incorrect = length - np.sum(correct) 
        return incorrect
    
    def learning_rule_algo(self):
        for i in range(self.iter):
            # Init pred_y
            input_size = self.x.shape[0]
            pred_y = np.zeros((input_size, 1))
            
            # Predict & Activation
            pred_y = self.predict_fn(self.x)
            # Error & Cost
            error = self.y - pred_y
            cost = self.cost_fn(error)
            # Weight update
            # w = w + alpha * (y - h) * x
            self.weight = self.weight + self.alpha * np.dot(self.x.T, error)
            
            #print("Iteration : ", str(i + 1))
            #print("Updated weights : ", self.weight.T)
            #print("Cost : ", str(cost))
            accuracy = self.calculate_misclassified(pred_y)
            #print("Incorrectly classified data : ", str(accuracy))
            self.incorrect[i] = accuracy
            


# In[6]:


def main():
    
    X, Y, bias = load_data()
    X = np.column_stack((bias, X))

    perceptron = Perceptron(X, Y)

if __name__ == '__main__':
    main()


# In[ ]:





# In[ ]:





# In[ ]:




