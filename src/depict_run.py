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

# xs = np.random.random((1000,28*28))
X = np.loadtxt(r"..\data\kth_xrand_r9.txt") # histogram
# np.random.shuffle(X)
# X = X[:1024,:]
# X = pre.minmax_scale(X,(0,1), axis=1) # distribution
# us = np.array([[1/float(outdim)]*outdim]) # uniform distribution
print('X is loaded! ')

# Y = np.loadtxt('../data/kth_Y_6.txt')
# oys = np.argmax(Y, axis=1)
# delta = np.array([0]*Y.shape[0])
kys = np.argmin(kms.fit_transform(X), axis=1)
khs = np.histogram(kys, bins=outdim)[0]

epoch = 0
while True:
    m,n = X.shape
    for i in range(m/1000):
        idx = random.sample([i for i in range(m)], k=1000)
        xs = X[idx,:]
        ks = kys[idx,:]

        nn.partial_fit(xs)
        cost, ps = nn.predict(xs)

        nmi3 = NMI(ks, ps)
        log = '%s  epoch: %10d  cost: %.4e nmi-1: %.8f nmi-2: %.8f nmi-3: %.8f' % (dt.datetime.now(), epoch, cost, 0, 0, nmi3)

    if not epoch % 1000:
        cost, pys = nn.predict(X)

        phs = np.histogram(pys, bins=outdim)[0]

        # nmi1 = NMI(oys, pys)
        # nmi2 = NMI(oys, kys)
        nmi3 = NMI(kys, pys)

        # abs = np.sum(oys - delta)
        # delta = pys

        log = '%s  epoch: %10d  cost: %.4e nmi-1: %.8f nmi-2: %.8f nmi-3: %.8f'%(dt.datetime.now(), epoch, cost, 0, 0, nmi3)
        logFile = r'../data/log.txt'
        utils.writeLog(logFile, log)
        print(log)

        print(khs)
        print(phs)
    epoch += 1

    flag = False
    if flag: break


