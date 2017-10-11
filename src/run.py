# -*- coding: utf-8 -*-

import os
import sys
import math
import datetime as dt

import numpy as np
from sklearn.metrics import mean_squared_error as sklmse
from sklearn.metrics import normalized_mutual_info_score as sklnmi

import utils 
import rbfnn
from v3 import dec 
from v3 import depict


# def main(): 
#     def loadData(): 
#         pass 
    
#     for numCenter in numCenterList: 
#         for numCluster in numClusterList: 
#             for epoch_index in epochList: 
#                 def valid(): 
#                     pass 

def classify(name):
    def bow(X, C):
        H = np.zeros((0,outdim))
        e = 0
        for c in C:
            s = e
            e += c
            xs = X[s:e,:]
            hs = np.histogram(xs, bins=outdim)[0]
            H = np.vstack((H, hs))
        return np.array(H)

    def calc_err(y_true, y_predict):
        return sklmse(y_true, y_predict, multioutput='raw_values')
    def calc_acc(y_true, y_predict):
        tidx = np.argmax(y_true, axis=1)
        pidx = np.argmax(y_predict, axis=1)
        eidx = (tidx == pidx)
        acc = eidx.sum() / eidx.shape[0]
        return acc
    
    pb_file_path = '../model/%s/%s'%(name, name)
    if name == 'depict':
        valid = depict.valid_network
    if name == 'dec':
        valid = dec.valid_network
    network = rbfnn.RBFNN(indim=outdim, numCenter=120, outdim=6)

    C1 = np.loadtxt(r"..\data\kth_ctrain_r9.txt").astype('int')
    Z1 = np.loadtxt(r"..\data\kth_xtrain_r9.txt").astype('float32')
    T1 = np.loadtxt(r"..\data\kth_ytrain_r9.txt").astype('int')
    
    C2 = np.loadtxt(r"..\data\kth_ctest_r9.txt").astype('int')
    Z2 = np.loadtxt(r"..\data\kth_xtest_r9.txt").astype('float32')
    T2 = np.loadtxt(r"..\data\kth_ytest_r9.txt").astype('int')
    
    kms = utils.mini_kmeans('../model', 'kmeans', Z1, outdim, factor=4)
    KT1 = np.reshape(kms.predict(Z1), (-1, 1))
    KT2 = np.reshape(kms.predict(Z2), (-1, 1))

    for epoch_index in range(0,100,10): 
        num_samples,num_featues = Z1.shape
        batch_size = int(outdim * 4)
        num_batches = int(math.ceil(num_samples / float(batch_size)))
        msg = "%s num_samples: %d num_features: %d batch_size: %d num_batches: %d"%(
            dt.datetime.now(), num_samples, num_featues, batch_size, num_batches)
        print(msg)
        A1, _, _  = valid(pb_file_path, Z1, KT1, batch_size, epoch_index)

        num_samples, num_featues = Z2.shape
        batch_size = int(outdim * 4)
        num_batches = int(math.ceil(num_samples / float(batch_size)))
        msg = "%s num_samples: %d num_features: %d batch_size: %d num_batches: %d" % (
            dt.datetime.now(), num_samples, num_featues, batch_size, num_batches)
        print(msg)
        A2, _, _ = valid(pb_file_path, Z2, KT2, batch_size, epoch_index)

        H1, H2 = bow(A1, C1), bow(A2, C2)

        network.fit(H1, T1)
        O1 = network.predict(H1)
        O2 = network.predict(H2)

        print('acc1: %.4f, acc2: %.4f'%(calc_acc(O1, T1), calc_acc(O2, T2)))


def valid(name, epoch_index):
    Z = np.loadtxt('../data/kth_xrand_r9_reduce.txt')
    kms = utils.mini_kmeans('../model', 'kmeans', Z, outdim, factor=4)
    T = np.reshape(kms.predict(Z), (-1, 1))

    pb_file_path = '../model/%s/%s'%(name, name)
    if name == 'depict':
        valid = depict.valid_network
    if name == 'dec':
        valid = dec.valid_network

    num_samples, num_featues = Z.shape
    batch_size = int(outdim * 4)
    num_batches = int(math.ceil(num_samples / float(batch_size)))
    msg = "%s num_samples: %d num_features: %d batch_size: %d num_batches: %d" % (
        dt.datetime.now(), num_samples, num_featues, batch_size, num_batches)
    print(msg)

    Y, loss, nmi = valid(pb_file_path, Z, T, batch_size, epoch_index)
    print(Y.shape, loss, nmi)

def train(name, indim=162, outdim=4096):
    Z = np.loadtxt('../data/kth_xtrain_r9.txt')
    kms = utils.mini_kmeans('../model', 'kmeans', Z, outdim, factor=4)
    T = np.reshape(kms.predict(Z), (-1,1))
    ws = np.dot(np.linalg.pinv(Z), T)
    us = kms.cluster_centers_.astype(np.float32)
    
    pb_file_path = '../model/%s/%s'%(name, name)
    if name == 'depict': 
        network = depict.build_network(indim=indim, outdim=outdim)
        train_network = depict.train_network
    if name == 'dec': 
        network = dec.build_network(indim=indim, outdim=outdim)
        train_network = dec.train_network

    num_samples, num_featues = Z.shape
    batch_size = int(outdim * 4)
    num_batches = int(math.ceil(num_samples / float(batch_size)))
    msg = "%s num_samples: %d num_features: %d batch_size: %d num_batches: %d" % (
        dt.datetime.now(), num_samples, num_featues, batch_size, num_batches)
    print(msg)

    num_epochs = 10000
    train_network(network, Z, T, batch_size, num_epochs, pb_file_path)

if __name__ == "__main__": 
    name = 'depict' 
    indim, outdim = 162, 1024
    train(name, indim, outdim)
    # classify(name)