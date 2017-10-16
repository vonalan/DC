# -*- coding: utf-8 -*-

import os
import sys
import math
import datetime as dt

import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error as sklmse
from sklearn.preprocessing import minmax_scale as sklscale
from sklearn.metrics import normalized_mutual_info_score as sklnmi

import utils 
import rbfnn
from v3 import dec 
from v3 import depict


# global settings 
num_classes = 6

# num_epochs = 1000 + 1
# numCenterList = [i for i in range(90, 150 + 1, 6)]
# numClusterList = [1<<i for i in range(7, 12 + 1, 1)]

factor = -1 # factor <=0 | factor > 0
num_epochs = 1000 + 1 # 1000 is enough! 
numCenterList = [i for i in range(120, 120 + 1, 6)] # M = 120
numClusterList = [1<<i for i in range(10, 12 + 1, 1)] # K=1024


def valid(name):
    def bow(X, C, outdim=4096):
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

    # # load datesets
    C1 = np.loadtxt(r"..\data\kth_ctrain_r9.txt").astype('int')
    Z1 = np.loadtxt(r"..\data\kth_xtrain_r9.txt").astype('float32')
    T1 = np.loadtxt(r"..\data\kth_ytrain_r9.txt").astype('int')
    
    C2 = np.loadtxt(r"..\data\kth_ctest_r9.txt").astype('int')
    Z2 = np.loadtxt(r"..\data\kth_xtest_r9.txt").astype('float32')
    T2 = np.loadtxt(r"..\data\kth_ytest_r9.txt").astype('int')

    # debug
    # C1, Z1, T1 = C2, Z2, T2

    # # global settings 
    # num_epochs = 100
    # numCenterList = [i for i in range(90, 120+1, 6)]
    # numClusterList = [1<<i for i in range(7, 12+1, 1)]

    # model related
    pb_file_dir = '../model/%s/'%(name)
    pb_file_path = '../model/%s/%s'%(name, name)
    if not os.path.exists(pb_file_dir): os.mkdir(pb_file_dir)
    
    if name == "depict": 
        valid_network = depict.valid_network
    elif name == "dec": 
        valid_network = dec.valid_network
    else: 
        valid_network = None

    # results = list()
    for numCenter in numCenterList:
        for numCluster in numClusterList:
            network = rbfnn.RBFNN(indim=numCluster, numCenter=numCenter, outdim=num_classes)

            # kmeans transformation 
            kms = utils.kmeans('../model', 'kmeans', X=Z1, outdim=numCluster, factor=factor)
            KT1 = np.reshape(kms.predict(Z1), (-1, 1))
            KT2 = np.reshape(kms.predict(Z2), (-1, 1))

            '''NEW NEW NEW'''
            # classifier
            H1, H2 = bow(KT1, C1, outdim=numCluster), bow(KT2, C2, outdim=numCluster)
            H1, H2 = sklscale(H1, (-1, 1), axis=1), sklscale(H2, (-1, 1), axis=1)
            network.fit(H1, T1)
            O1 = network.predict(H1)
            O2 = network.predict(H2)
            # metrics = [
            #     numCenter, numCluster, epoch_index,
            #     calc_err(O1, T1), calc_acc(O1, T1), calc_err(O2, T2), calc_acc(O2, T2), 0.25
            # ]
            # results.append(metrics)
            #  print('%s m: %4d k: %4d e: %8d err_1: %.8e, acc_1: %.10f, err_1: %.8e, acc_1: %.10f, stsm: %.10f'%(dt.datetime.now(),
            #     numCenter, numCluster, epoch_index,
            #     calc_err(O1, T1), calc_acc(O2, T2), calc_err(O2, T2), calc_acc(O2, T2), 0.25))

            metrics = "%d %d %d %f %f %f %f %f" % (
                numCenter, numCluster, 0,
                calc_err(O1, T1).sum(), calc_acc(O1, T1), calc_err(O2, T2).sum(), calc_acc(O2, T2), 0.25)
            utils.writeLog("../data", "%s_results" % ('kmeans'), metrics)
            print('%s m: %4d k: %4d e: %8d err_1: %.8e, acc_1: %.10f, err_2: %.8e, acc_2: %.10f, stsm: %.10f' % (
            dt.datetime.now(), numCenter, numCluster, 0,
            calc_err(O1, T1).sum(), calc_acc(O1, T1), calc_err(O2, T2).sum(), calc_acc(O2, T2), 0.25))
            '''NEW NEW NEW'''

            for epoch_index in range(num_epochs):
                if epoch_index % pow(10, len(str(epoch_index))-1): 
                    continue 
                
                # training set 
                num_samples,num_featues = Z1.shape
                batch_size = int(numCluster * 4)
                num_batches = int(math.ceil(num_samples / float(batch_size)))
                msg = "%s num_samples: %d num_features: %d batch_size: %d num_batches: %d"%(
                    dt.datetime.now(), num_samples, num_featues, batch_size, num_batches)
                print(msg)
                A1, _, _  = valid_network(pb_file_path, Z1, KT1, batch_size, numCluster, epoch_index)

                # validate set 
                num_samples, num_featues = Z2.shape
                batch_size = int(numCluster * 4)
                num_batches = int(math.ceil(num_samples / float(batch_size)))
                msg = "%s num_samples: %d num_features: %d batch_size: %d num_batches: %d" % (
                    dt.datetime.now(), num_samples, num_featues, batch_size, num_batches)
                print(msg)
                A2, _, _ = valid_network(pb_file_path, Z2, KT2, batch_size, numCluster, epoch_index)

                if not (A1.shape[0] and A2.shape[0]): continue

                # classifier
                H1, H2 = bow(A1, C1, outdim=numCluster), bow(A2, C2, outdim=numCluster)
                H1, H2 = sklscale(H1, (-1,1), axis=1), sklscale(H2, (-1,1), axis=1)
                network.fit(H1, T1)
                O1 = network.predict(H1)
                O2 = network.predict(H2)
                # metrics = [
                #     numCenter, numCluster, epoch_index, 
                #     calc_err(O1, T1), calc_acc(O1, T1), calc_err(O2, T2), calc_acc(O2, T2), 0.25
                # ]
                # results.append(metrics)
                #  print('%s m: %4d k: %4d e: %8d err_1: %.8e, acc_1: %.10f, err_1: %.8e, acc_1: %.10f, stsm: %.10f'%(dt.datetime.now(),
                #     numCenter, numCluster, epoch_index, 
                #     calc_err(O1, T1), calc_acc(O2, T2), calc_err(O2, T2), calc_acc(O2, T2), 0.25))

                metrics = "%d %d %d %f %f %f %f %f"%(
                    numCenter, numCluster, epoch_index, 
                    calc_err(O1, T1).sum(), calc_acc(O1, T1), calc_err(O2, T2).sum(), calc_acc(O2, T2), 0.25)
                utils.writeLog("../data", "%s_results"%(name), metrics)
                print('%s m: %4d k: %4d e: %8d err_1: %.8e, acc_1: %.10f, err_2: %.8e, acc_2: %.10f, stsm: %.10f'%(dt.datetime.now(),numCenter, numCluster, epoch_index,
                    calc_err(O1, T1).sum(), calc_acc(O1, T1), calc_err(O2, T2).sum(), calc_acc(O2, T2), 0.25))
    # np.savetxt('../data/result.txt', np.array(results))

def train(name):
    # load datasets 
    Z = np.loadtxt(r'../data/kth_xtrain_r9.txt').astype('int')

    # # global settings 
    # num_epochs = 100
    # numCenterList = [i for i in range(90, 150+1, 6)]
    # numClusterList = [1<<i for i in range(7, 12+1, 1)]

    # model related
    pb_file_dir = '../model/%s/'%(name)
    pb_file_path = '../model/%s/%s'%(name, name)
    if not os.path.exists(pb_file_dir): os.mkdir(pb_file_dir)

    for numCluster in numClusterList:
        # kmeans transformation 
        kms = utils.kmeans('../model', 'kmeans', X=Z, outdim=numCluster, factor=factor)
        T = np.reshape(kms.predict(Z), (-1,1))
        # ''''''
        # tset = set(T.flatten().tolist())
        # print("len(tset): %d"%(len(tset)))
        # T = OneHotEncoder(n_values=numCluster).fit_transform(T)
        # print(T.shape)
        # ws = np.dot(np.linalg.pinv(Z), T)
        # ''''''
        ws = np.zeros((Z.shape[1], numCluster))
        us = kms.cluster_centers_.astype(np.float32)
        print(ws.shape, us.shape)
        print("%s ws: %s us: %s"%(dt.datetime.now(), ws.shape, us.shape))

        if name == 'depict': 
            network = depict.build_network(indim=Z.shape[1], outdim=numCluster)
            train_network = depict.train_network
        elif name == 'dec': 
            network = dec.build_network(indim=Z.shape[1], outdim=numCluster)
            train_network = dec.train_network
        else: 
            network = None 
            train_network = None

        num_samples, num_featues = Z.shape
        batch_size = int(numCluster * 4)
        num_batches = int(math.ceil(num_samples / float(batch_size)))
        msg = "%s num_samples: %d num_features: %d batch_size: %d num_batches: %d" % (
            dt.datetime.now(), num_samples, num_featues, batch_size, num_batches)
        print(msg)
        train_network(network, Z, T, batch_size, numCluster, num_epochs, pb_file_path)

if __name__ == "__main__": 
    name = sys.argv[1]
    mode = sys.argv[2]

    if mode == 'train': 
        train(name)
    if mode == 'valid': 
        valid(name)