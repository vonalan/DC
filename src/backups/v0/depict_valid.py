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
import rbfnn
from depict_v1 import NeuralNetwork

indim = 162
outdim = 128

C1 = np.loadtxt(r"..\data\kth_ctrain_r9.txt")
C2 = np.loadtxt(r"..\data\kth_ctest_r9.txt")
X0 = np.loadtxt(r"..\data\kth_xrand_r9.txt")
X1 = np.loadtxt(r"..\data\kth_xtrain_r9.txt")
X2 = np.loadtxt(r"..\data\kth_xtest_r9.txt")
T1 = np.loadtxt(r"..\data\kth_ytrain_r9.txt")
T2 = np.loadtxt(r"..\data\kth_ytest_r9.txt")

# kms = KMeans(n_clusters=outdim)
# kms.fit(X0)

nn = NeuralNetwork(indim=indim, outdim=outdim)
nn.loadModel('../model', 1000)

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

# def kmeans():
#     kms.fit(X1)
#     Y1 = kms.predict(X1)
#     Y2 = kms.predict(X2)
#     H1 = bow(Y1, C1)
#     H2 = bow(Y2, C2)
#     return H1, H2

def depict():
    _, Y1 = nn.predict(X1)
    _, Y2 = nn.predict(X2)
    H1 = bow(Y1, C1)
    H2 = bow(Y2, C2)
    return H1, H2

# classifier 
clsm = rbfnn.RBFNN(indim=outdim, numCenter=120, outdim=6)

H1, H2 = depict()
clsm.fit(H1, T1)

Y1 = clsm.predict(X1)
Y2 = clsm.predict(X2)
