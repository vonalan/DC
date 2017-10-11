# -*- coding: utf-8 -*-

import os
import sys

import utils 
import v3.dec as dec 
import v3.depict as depict

def valid(name, epoch_index): 
    C1 = np.loadtxt(r"..\data\kth_ctrain_r9.txt").astype('int')
    X1 = np.loadtxt(r"..\data\kth_xtrain_r9.txt").astype('float32')
    T1 = np.loadtxt(r"..\data\kth_ytrain_r9.txt").astype('int')
    C2 = np.loadtxt(r"..\data\kth_ctest_r9.txt").astype('int')
    X2 = np.loadtxt(r"..\data\kth_xtest_r9.txt").astype('float32')
    T2 = np.loadtxt(r"..\data\kth_ytest_r9.txt").astype('int')

    pb_file_path = '../model/%s/%s'%(name, name)
    if name == 'depict': 
        valid = depict.valid_network
    if name == 'dec': 
        valid = dec.valid_network
    
    num_samples,n_featues = Z1.shape
    batch_size = int(outdim * 4)
    num_batch = int(ceil(num_samples/float(batch_size)))
    msg = "%s m: %d n: %d batch_size: %d num_batch: %d"%(
        dt.datetime.now(), m, n, bs, nb)
    Y1 = valid(pb_file_path, Z1, T1, batch_size, epoch_index)

    num_samples,n_featues = Z2.shape
    batch_size = int(outdim * 4)
    num_batch = int(ceil(num_samples/float(batch_size)))
    msg = "%s m: %d n: %d batch_size: %d num_batch: %d"%(
        dt.datetime.now(), m, n, bs, nb)
    Y2 = valid(pb_file_path, Z2, T2, batch_size, epoch_index)

    # RNN
    def bow(X, C):
        H = np.zeros((0,outdim))
        e = 0
        for c in C:
            s = e
            e += c
            xs = X[s:e]
            hs = np.histogram(xs, bins=outdim)[0]
            H = np.vstack((H, hs))
        return np.array(H)
    H1, H2 = bow(Y1, C1), bow(Y2, C2)

    import rbfnn
    network = rbfnn.RBFNN(indim=outdim, numCenter=120, outdim=6)
    network.fit(H1, T1)
    O1 = network.predict(H1, T1)
    O2 = network.predict(H2, T2)

    def calc_err(y_true, y_predict):
        return sklmse(y_true, y_predict, multioutput='raw_values')
    def calc_acc(y_true, y_predict):
        tidx = np.argmax(y_true, axis=1)
        pidx = np.argmax(y_predict, axis=1)
        eidx = (tidx == pidx)
        acc = eidx.sum() / eidx.shape[0]
        return acc
    print('acc1: %.4f, acc2: %.4f'%(calc_acc(O1, T1), calc_acc(O2, T2)))


def train(name, mode, indim=162, outdim=4096): 
    Z = np.loadtxt('../data/kth_xtrain_r9.txt')
    kms = utils.mini_kmeans('../model', 'kmeans', X, outdim, factor=4)
    T = np.reshape(kms.predict(X), (-1,1))
    ws = np.dot(np.linalg.pinv(X), T)
    us = kms.cluster_centers_.astype(np.float32)
    
    pb_file_path = '../model/%s/%s'%(name, name)
    if name == 'depict': 
        network = depict.build_network(indim=indim, outdim=outdim)
        train_network = depict.train_network
    if name == 'dec': 
        network = dec.build_network(indim=indim, outdim=outdim)
        train_network = dec.train_network
    
    num_samples,n_featues = X.shape
    batch_size = int(outdim * 4)
    num_batch = int(ceil(num_samples/float(batch_size)))
    msg = "%s m: %d n: %d batch_size: %d num_batch: %d"%(
        dt.datetime.now(), m, n, bs, nb)
    train(network, Z, T, batch_size, num_epochs, pb_file_path) 

if __name__ == "__main__": 
    name = 'depict' 
    indim, outdim = 162, 4096
    main(name, indim, outdim)