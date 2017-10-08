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
outdim = 4096

kms = KMeans(n_clusters=outdim)
nn = NeuralNetwork(indim=indim, outdim=outdim)

X = np.loadtxt(r"..\data\kth_xtrain_r9.txt") # histogram
# np.random.shuffle(X)
# X = X[:1024,:]
# X = pre.minmax_scale(X,(0,1), axis=1) # distribution

m, n = X.shape
bs = int(outdim * 4)
nb = int(m/bs) + 1
print("%s m: %d n: %d batch_size: %d num_batch: %d"%(dt.datetime.now(), m, n, bs, nb))

# kys = np.argmin(kms.fit_transform(X), axis=1)
# khs = np.histogram(kys, bins=outdim)[0]

epoch = 0
while True:
    for i in range(nb):
        idx = random.sample([i for i in range(m)], k=bs)
        xs = X[idx,:]
        cost = nn.partial_fit(xs)
        # log = '%s  epoch: %10d  batch: %4d cost: %.4e' % (dt.datetime.now(), epoch, i, cost)
        # print(log)
    if not epoch % 1:
        cost, pys = nn.predict(X)

        # nmi = NMI(kys, pys)

        log = '%s  epoch: %10d  cost: %.4e nmi: %.8f'%(dt.datetime.now(), epoch, cost, 0)
        logFile = r'../data/log.txt'
        utils.writeLog(logFile, log)

        # phs = np.histogram(pys, bins=outdim)[0]

        print(log)
        # print(khs)
        # print(phs)

    if not epoch%100:
        nn.saveModel('../model/depict', epoch+1)

    epoch += 1

    flag = False
    if flag: break

