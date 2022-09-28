#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import csv


# In[2]:


# https://scikit-learn.org/stable/auto_examples/svm/plot_separating_hyperplane.html#sphx-glr-auto-examples-svm-plot-separating-hyperplane-py
from sklearn import svm
from sklearn.datasets import make_blobs


# In[3]:


PATH = os.getcwd() + "\\"
DATA = "data_SVM.csv"


# In[4]:


def load_data():
    
    data_path = PATH + DATA
    df = pd.read_csv(data_path, header = None)
    data = df.to_numpy()
    
    X = data[:,:-1]
    Y = data[:,-1:]
    
    # Replace -1 with 0 for sigmoid function
    X = np.where(X == -1, 0, X)
    Y = np.where(Y == -1, 0, Y)
    
    return X, Y


# In[5]:


def plot_fn(X, y, clf):

    plt.scatter(X[:, 0], X[:, 1], c=y, s=10, cmap=plt.cm.Paired)

    # plot the decision function
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)

    # plot decision boundary and margins
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    # plot support vectors
    ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
               linewidth=1, facecolors='none', edgecolors='k')
    plt.show()


# In[6]:


def SVM(x, y):
    y = y.ravel()
    clf = svm.SVC(kernel='linear', C=1000)
    clf.fit(x, y)
    plot_fn(x, y, clf)


# In[7]:


def main():
    
    X, Y = load_data()
    
    SVM(X, Y)

if __name__ == '__main__':
    main()


# In[ ]:





# In[ ]:




