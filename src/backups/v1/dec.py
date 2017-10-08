# -*- coding: utf-8 -*-

# requirements: 
#   Anaconda3
#   Tensorflow

import os
import sys
import random

import numpy as np 
import tensorflow as tf


def mini_kmeans(root, name, X, outdim, factor=4):
    from sklearn.externals import joblib
    dstdir = os.path.join(root, name)
    if not os.path.exists(dstdir): os.mkdir(dstdir)
    mfile = os.path.join(dstdir, r"kth_kmeans_r9_k%d.m" % (outdim))
    if not os.path.exists(mfile):
        from sklearn.cluster import KMeans
        idx = random.sample([i for i in range(m)], outdim * factor)
        xs = X[idx, :]
        kms = KMeans(n_clusters=outdim).fit(xs)
        joblib.dump(kms, mfile, compress=3)
    else:
        kms = joblib.load(mfile)
    return kms


def _func_01_(Z, U, a=1.0):
    '''
    A noval implementaton of pair-wise distance between two vectors
    from two Arrays.

    z = Z[i,:]
    u = U[j,:]
    D[i,j] = (z-u)*(z-u).T = z*z.T + u*u.T - z*u.T - z.T*u

    :param Z:
    :param U:
    :param a:
    :return:
    '''

    m1, m2 = tf.shape(Z)[0], tf.shape(U)[0]

    ''''''
    # M1 = tf.matmul(Z, tf.transpose(Z)) # out of memory
    # M2 = tf.matmul(U, tf.transpose(U)) # out of memory
    # M3 = tf.matmul(Z, tf.transpose(U)) # out of memory
    # M4 = tf.matmul(U, tf.transpose(Z)) # out of memory
    #
    # M1 = tf.reshape(tf.diag_part(M1), (m1, 1))
    # M2 = tf.reshape(tf.diag_part(M2), (1, m2))
    # M4 = tf.transpose(M4)
    #
    # D = M1 + M2 - (M3 + M4)
    ''''''

    M1 = tf.reshape(tf.reduce_sum(tf.pow(Z,2), axis=1),(m1,1))
    M2 = tf.reshape(tf.reduce_sum(tf.pow(U,2), axis=1),(1,m2))
    M3 = tf.matmul(Z, tf.transpose(U))
    M4 = tf.matmul(U, tf.transpose(Z))

    D = M1 + M2 - (M3 + tf.transpose(M4))

    N = tf.pow(D/a + 1, -((a+1)/2))
    D = tf.reshape(tf.reduce_sum(N, axis=1), (-1, 1))
    Q = N/D

    return Q

def _func_03_(Q):
    f = tf.reshape(tf.reduce_sum(Q, axis=0),(-1, outdim))
    Q = tf.pow(Q, 2)
    N = tf.div(Q, f)
    D = tf.reshape(tf.reduce_sum(N, axis=1),(-1, 1))
    P = tf.div(N, D)
    return P

def _func_02_(Q, P):
    Q = tf.log(tf.div(P, Q))
    Q = tf.multiply(P, Q)

    L = tf.reshape(tf.reduce_sum(Q, axis=1),(-1,1))
    L = tf.reduce_mean(L)
    # L = tf.reduce_sum(Q)

    return L


if __name__ == "__main__":
    import datetime as dt
    from sklearn.metrics import normalized_mutual_info_score as NMI
    import sklearn.preprocessing as pre
    import utils

    # *****************************************************************
    name = "dec"
    indim, outdim = 162, 4096 ###
    # *****************************************************************

    # *****************************************************************
    X = np.loadtxt('../data/kth_xtrain_r9.txt')
    # X = pre.minmax_scale(X,(0,1), axis=1) # distribution
    # *****************************************************************

    # *****************************************************************
    m, n = X.shape
    bs = int(outdim * 4)
    nb = int(m / bs) + 1
    msg = "%s m: %d n: %d batch_size: %d num_batch: %d" % (
        dt.datetime.now(), m, n, bs, nb)
    # *****************************************************************

    # *****************************************************************
    kms = mini_kmeans('../model', 'kmeans', X, outdim, factor=4)
    kys = kms.predict(X)
    # khs = np.histogram(kys, bins=outdim)[0]
    us = kms.cluster_centers_.astype(np.float32)
    # *****************************************************************

    # *****************************************************************
    U = tf.Variable(us)

    Z = tf.placeholder('float', [None, indim])
    Q = _func_01_(Z, U)
    P = _func_03_(Q)
    L = _func_02_(Q, P)

    cost = L
    optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

    Y = tf.argmax(Q, axis=1)
    # *****************************************************************

    init = tf.global_variables_initializer()

    config = tf.ConfigProto(device_count={"CPU": 24, "GPU": 0})
    sess = tf.Session(config=config)
    sess.run(init)

    print(msg)
    epoch = 0
    while True:
        for i in range(nb):
            idx = random.sample([i for i in range(m)], k=bs)
            xs = X[idx, :]
            sess.run(optimizer, feed_dict={Z: xs})

            # loss = sess.run(cost, feed_dict={Z: xs})
            # log = '%s  batch: %10d  cost: %.8e nmi-1: %.8f nmi-2: %.8f nmi-3: %.8f' % (
            #     dt.datetime.now(), i, loss, 0, 0, 0)
            # print(log)

        if not epoch % 1:
            loss, dys = sess.run([cost, Y], feed_dict={Z: X})

            nmi1 = 0
            nmi2 = 0
            nmi3 = NMI(kys, dys)

            log = '%s  epoch: %10d  cost: %.8e nmi-1: %.8f nmi-2: %.8f nmi-3: %.8f' % (
                dt.datetime.now(), epoch, loss, nmi1, nmi2, nmi3)
            logFile = r'../data/%s_log.txt'%(name)
            utils.writeLog('../log', name, log)
            print(log)

            # dhs = np.histogram(dys, bins=outdim)[0]

            # print(khs)
            # print(dhs)

        if not epoch % 100:
            utils.saveModel(sess, '../model', name, epoch)

        epoch += 1

        flag = False
        if flag: break
    print("optimization is finished! ")
