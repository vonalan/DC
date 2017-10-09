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
from sklearn.metrics import mean_squared_error as sklmse

import utils
import rbfnn
from depict import NeuralNetwork


indim = 162
outdim = 4096

C1 = np.loadtxt(r"..\data\kth_ctrain_r9.txt").astype('int')
C2 = np.loadtxt(r"..\data\kth_ctest_r9.txt").astype('int')

# X0 = np.loadtxt(r"..\data\kth_xrand_r9.txt").astype('float32')
X1 = np.loadtxt(r"..\data\kth_xtrain_r9.txt").astype('float32')
X2 = np.loadtxt(r"..\data\kth_xtest_r9.txt").astype('float32')

T1 = np.loadtxt(r"..\data\kth_ytrain_r9.txt").astype('int')
T2 = np.loadtxt(r"..\data\kth_ytest_r9.txt").astype('int')


# *****************************************************************
# kms = utils.mini_kmeans('../model', 'kmeans', X, outdim, factor=4)
# kys = kms.predict(X)
# # khs = np.histogram(kys, bins=outdim)[0]
# ws = np.dot(np.linalg.inv(X), kys)
# # us = kms.cluster_centers_.astype(np.float32)
# *****************************************************************

nn = NeuralNetwork(indim=indim, outdim=outdim)
nn.loadModel('../model', 0)
print('depict_0.cpkt')

# scale to [-1,1]
def auto_scale(dataset, bias=None, scale=None, mode=None):
    xbias = bias
    xscale = scale

    if mode == 'train':
        xmin = dataset.min(axis=0)
        xmax = dataset.max(axis=0)
        xbias = (xmax + xmin) / 2.0
        xscale = xmax - bias
    elif mode == 'valid':
        pass
    else:
        raise Exception('Value Error! ')

    dataset = (dataset - xbias) / xscale

    return dataset, xbias, xscale

# api
def calc_err(y_true, y_predict):
    return sklmse(y_true, y_predict, multioutput='raw_values')

def calc_acc(y_true, y_predict):
    tidx = np.argmax(y_true, axis=1)
    pidx = np.argmax(y_predict, axis=1)

    eidx = (tidx == pidx)
    acc = eidx.sum() / eidx.shape[0]

    return acc

def bow(X, C):
    if not X.shape[0] == np.sum(C):
        raise Exception("Error! ")
    H = np.zeros((0,outdim))
    e = 0
    for c in C:
        s = e
        e += c
        xs = X[s:e]
        hs = np.histogram(xs, bins=outdim)[0]
        H = np.vstack((H, hs))
    return np.array(H)

# def kmeans():
#     kms.fit(X1)
#     Y1 = kms.predict(X1)
#     Y2 = kms.predict(X2)
#     H1 = bow(Y1, C1)
#     H2 = bow(Y2, C2)
#     return H1, H2

# C1 = C2
# X1 = X2
# T1 = T2

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

Y1 = clsm.predict(H1)
Y2 = clsm.predict(H2)

print('acc1: %.4f, acc2: %.4f'%(calc_acc(Y1, T1), calc_acc(Y2, T2)))


