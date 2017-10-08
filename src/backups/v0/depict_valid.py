# -*- coding: utf-8 -*-

# requirements: 
#   Anaconda3
#   Tensorflow

import random
import datetime as dt

import numpy as np 
import tensorflow as tf 


from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.cluster import KMeans
import sklearn.preprocessing as pre

import utils
from depict import NeuralNetwork

indim = 162
outdim = 128 

kms = KMeans(n_clusters=outdim)
nn = NeuralNetwork(indim=indim, outdim=outdim)

C1 = np.loadtxt(r"..\data\kth_ctrain_r9.txt")
C2 = np.loadtxt(r"..\data\kth_ctest_r9.txt")
X1 = np.loadtxt(r"..\data\kth_xtrain_r9.txt")
X2 = np.loadtxt(r"..\data\kth_xtest_r9.txt")
Y1 = np.loadtxt(r"..\data\kth_ytrain_r9.txt")
Y2 = np.loadtxt(r"..\data\kth_ytest_r9.txt")

# np.random.shuffle(X)
# X = X[:1024,:]
# X = pre.minmax_scale(X,(0,1), axis=1) # distribution

def bow(X, C):
    if not X.shape[0] == np.sum(C):
        raise Exception("Error! ")
    H = None
    e = 0
    for c in C:
        s = e
        e += c
        xs = X[s:e]
        hs = np.histogram(xs, bins=outdim)[0]

        if H: H = np.hstack((H, hs))
        else: H = hs
    return H

def kmeans():
    kms.fit(X1)
    T1 = kms.predict(X1)
    T2 = kms.predict(X2)
    H1 = bow(T1, C1)
    H2 = bow(T2, C2)
    return H1, H2

def depict():
    nn.loadModel('../model', 100000)
    _, T1 = nn.predict(X1)
    _, T2 = nn.predict(X2)
    H1 = bow(T1, C1)
    H2 = bow(T2, C2)
    return H1, H2








