# -*- coding: utf-8 -*-

# requirements: 
#   Anaconda3
#   Tensorflow

import os
import sys
import random

import numpy as np 
import tensorflow as tf

import numpy as np 
import tensorflow as tf 
from tensorflow.python.framework import graph_util


def build_network(indim=4096, outdim=6, U=None, a=1.0): 
    Z = tf.placeholder(tf.float32, shape=[None, indim], name='input')
    T = tf.placeholder(tf.int64, shape=[None, 1], name='label')

    with tf.name_scope('func_01') as scope: 
        U = tf.Variable(U)

        M1 = tf.reshape(tf.reduce_sum(tf.pow(Z, 2), axis=1), (-1, 1))
        M2 = tf.reshape(tf.reduce_sum(tf.pow(U, 2), axis=1), (1, outdim))
        M3 = tf.matmul(Z, tf.transpose(U))
        M4 = tf.matmul(U, tf.transpose(Z))

        D = M1 + M2 - (M3 + tf.transpose(M4))

        N = tf.pow(D / a + 1, -((a + 1) / 2))
        D = tf.reshape(tf.reduce_sum(N, axis=1), (-1, 1))
        Q = tf.div(N, D)

    with tf.name_scope('func_03') as scope: 
        F = tf.reshape(tf.reduce_sum(Q, axis=0), (-1, outdim))
        N = tf.div(tf.pow(Q, 2), F)
        D = tf.reshape(tf.reduce_sum(N, axis=1), (-1, 1))
        P = tf.div(N, D)
    
    with tf.name_scope('func_02') as scope: 
        C = tf.multiply(P, tf.log(tf.div(P, Q)))
        L = tf.reshape(tf.reduce_sum(C, axis=1), (-1, 1))
        L = tf.reduce_mean(L)

    with tf.name_scope('metrics') as scope: 
        Y = tf.argmax(Q, axis=1, name='output')
        # ACC = tf.reduce_mean(tf.cast(tf.equal(Y, T), tf.float32))
        # NMI = 0.0
    
    cost = L
    optimizer = tf.train.AdamOptimizer(0.01).minimize(cost)
    
    return dict(Z=Z, Y=Y, T=T, optimizer=optimizer, cost=cost)

if __name__ == "__main__":
    import random
    import datetime as dt

    from sklearn.metrics import normalized_mutual_info_score as NMI
    import sklearn.preprocessing as pre

    import utils

    # *****************************************************************
    name = 'depict'
    indim, outdim = 162, 4096 ###
    # *****************************************************************

    # *****************************************************************
    X = np.loadtxt('../data/kth_xtrain_r9.txt')
    # X = pre.minmax_scale(X,(0,1), axis=1) # distribution
    # *****************************************************************

    # *****************************************************************
    kms = utils.mini_kmeans('../model', 'kmeans', X, outdim, factor=4)
    kys = kms.predict(X)
    T = np.reshape(kys, (-1,1))
    # khs = np.histogram(kys, bins=outdim)[0]
    # ws = np.dot(np.linalg.inv(X), kys)
    us = kms.cluster_centers_.astype(np.float32)
    # *****************************************************************

    graph = build_network(indim=indim, outdim=outdim, U=us)
    init = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init)

    # *****************************************************************
    m,n = X.shape
    bs = int(outdim * 4)
    nb = int(m/bs) + 1
    msg = "%s m: %d n: %d batch_size: %d num_batch: %d"%(
        dt.datetime.now(), m, n, bs, nb)
    # *****************************************************************
    print(msg)
    epoch = 0
    while True:
        for i in range(nb):
            idx = random.sample([i for i in range(m)], k=bs)
            xs = X[idx,:]
            ts = T[idx,:]
            sess.run(graph['optimizer'], feed_dict={
                graph['Z']: xs,
                graph['T']: ts
            })

        if not epoch % 1:
            idx = random.sample([i for i in range(m)], k=bs)
            xs = X[idx, :]
            ts = T[idx, :]
            loss = sess.run(graph['cost'], feed_dict={
                graph['Z']: xs,
                graph['T']: ts
            })

            nmi1 = 0.5
            nmi2 = 0.5
            nmi3 = 0.5

            log = '%s  epoch: %10d  cost: %.8e nmi-1: %.8f nmi-2: %.8f nmi-3: %.8f' % (
                dt.datetime.now(), epoch, loss, nmi1, nmi2, nmi3)
            # utils.writeLog('../log', name, log)
            print(log)
        #
        #
        # if not epoch % 100:
        #     nn.saveModel('../model', epoch)

        epoch += 1

        flag = False
        if flag: break
    print("optimization is finished! ")