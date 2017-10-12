# -*- coding: utf-8 -*-

import os
import sys
import math
import datetime as dt

import numpy as np
from sklearn.metrics import mean_squared_error as sklmse
from sklearn.preprocessing import minmax_scale as sklscale
from sklearn.metrics import normalized_mutual_info_score as sklnmi

import utils 
import rbfnn
from v3 import dec 
from v3 import depict


def valid(name):
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

    # load datesets 
    C1 = np.loadtxt(r"..\data\kth_ctrain_r9.txt").astype('int')
    Z1 = np.loadtxt(r"..\data\kth_xtrain_r9.txt").astype('float32')
    T1 = np.loadtxt(r"..\data\kth_ytrain_r9.txt").astype('int')
    
    C2 = np.loadtxt(r"..\data\kth_ctest_r9.txt").astype('int')
    Z2 = np.loadtxt(r"..\data\kth_xtest_r9.txt").astype('float32')
    T2 = np.loadtxt(r"..\data\kth_ytest_r9.txt").astype('int')

    # global settings 
    num_epochs = 10000
    numCenterList = [i for i in range(90, 150+1, 6)]
    numClusterList = [1<<i for i in range(7, 12+1, 1)]

    # model related
    pb_file_path = '../model/%s/'%(name)
    if not os.path.exists(pb_file_path): os.mkdir(pb_file_path)
    if name == "depict": 
        valid_network = depict.valid_network
    elif name == "dec": 
        valid_network = dec.valid_network
    else: 
        valid_network = None

    results = list()
    for numCenter in numCenterList:
        for numCluster in numClusterList:
            outdim = numCluster
            network = rbfnn.RBFNN(indim=outdim, numCenter=120, outdim=6)
            n_pb_file_path = pb_file_path + '%s_m%d_k%d'%(name, numCenter, numCluster)

            # kmeans transformation 
            kms = utils.mini_kmeans('../model', 'kmeans', Z1, outdim, factor=4)
            KT1 = np.reshape(kms.predict(Z1), (-1, 1))
            KT2 = np.reshape(kms.predict(Z2), (-1, 1))
            
            for epoch_index in range(num_epochs):
                if epoch_index % pow(10, len(str(epoch_index))-1): 
                    continue 
                
                # training set 
                num_samples,num_featues = Z1.shape
                batch_size = int(outdim * 4)
                num_batches = int(math.ceil(num_samples / float(batch_size)))
                msg = "%s num_samples: %d num_features: %d batch_size: %d num_batches: %d"%(
                    dt.datetime.now(), num_samples, num_featues, batch_size, num_batches)
                print(msg)
                A1, _, _  = valid_network(n_pb_file_path, Z1, KT1, batch_size, epoch_index)

                # validate set 
                num_samples, num_featues = Z2.shape
                batch_size = int(outdim * 4)
                num_batches = int(math.ceil(num_samples / float(batch_size)))
                msg = "%s num_samples: %d num_features: %d batch_size: %d num_batches: %d" % (
                    dt.datetime.now(), num_samples, num_featues, batch_size, num_batches)
                print(msg)
                A2, _, _ = valid_network(n_pb_file_path, Z2, KT2, batch_size, epoch_index)

                # classifier
                H1, H2 = bow(A1, C1), bow(A2, C2)
                H1, H2 = sklscale(H1, (-1,1), axis=1)
                network.fit(H1, T1)
                O1 = network.predict(H1)
                O2 = network.predict(H2)
                metrics = [
                    numCenter, numCluster, epoch_index, 
                    calc_err(O1, T1), calc_acc(O1, T1), calc_err(O1, T1), calc_acc(O2, T2), 0.25
                ]
                results.append(metrics)
                print('%s m: %4d k: %4d e: %8d err_1: %.8e, acc_1: %.10f, err_1: %.8e, acc_1: %.10f, stsm: %.10f'%(dt.datetime.now(),
                    numCenter, numCluster, epoch_index, 
                    calc_err(O1, T1), calc_acc(O2, T2), calc_err(O1, T1), calc_acc(O2, T2), 0.25))
    np.savetxt('../data/result.txt', np.array(results))

def train(name, indim=162, outdim=4096):
    # load datasets 
    Z = np.loadtxt('../data/kth_xtrain_r9.txt')

    # global settings 
    num_epochs = 10000
    numCenterList = [i for i in range(90, 150+1, 6)]
    numClusterList = [1<<i for i in range(7, 12+1, 1)]

    # model related
    pb_file_path = '../model/%s/'%(name)
    for numCenter in numCenterList:
        for numCluster in numClusterList:
            n_pb_file_path = pb_file_path + '%s_m%d_k%d'%(name, numCenter, numCluster)
            outdim = numCenter

            # kmeans transformation 
            kms = utils.mini_kmeans('../model', 'kmeans', Z, outdim, factor=4)
            T = np.reshape(kms.predict(Z), (-1,1))
            ws = np.dot(np.linalg.pinv(Z), T)
            us = kms.cluster_centers_.astype(np.float32)

            if name == 'depict': 
                network = depict.build_network(indim=indim, outdim=outdim)
                train_network = depict.train_network
            elif name == 'dec': 
                network = dec.build_network(indim=indim, outdim=outdim)
                train_network = dec.train_network
            else: 
                network = None 
                train_network = None

            num_samples, num_featues = Z.shape
            batch_size = int(outdim * 4)
            num_batches = int(math.ceil(num_samples / float(batch_size)))
            msg = "%s num_samples: %d num_features: %d batch_size: %d num_batches: %d" % (
                dt.datetime.now(), num_samples, num_featues, batch_size, num_batches)
            print(msg)
            train_network(network, Z, T, batch_size, num_epochs, n_pb_file_path)

if __name__ == "__main__": 
    name = 'depict' 
    train(name)
    # valid(name)