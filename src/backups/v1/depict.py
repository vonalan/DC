# -*- coding: utf-8 -*-

# requirements: 
#   Anaconda3
#   Tensorflow


import numpy as np 
import tensorflow as tf 


def _func_01_(Z, W):
    '''
    A simple implementation of softmax function.
    Be careful of overflow problem!

    :param X:
    :param weights:
    :return:
    '''

    Z = tf.matmul(Z, W)
    Z = tf.exp(Z)
    S = tf.reshape(tf.reduce_sum(Z, 1), (-1, 1))
    P = tf.div(Z, S)

    return P

def _func_07_(P, n):
    '''
    A simple implementation of target function that is like softmax function.

    :param X:
    :return:
    '''

    Z = tf.reshape(tf.reduce_sum(P, axis=0), (-1, n))  # for every cluster
    Z = tf.pow(Z, 0.5)

    N = tf.div(P, Z)  # element-wise division
    D = tf.reshape(tf.reduce_sum(N, 1), (-1, 1))  # for every sample

    Q = tf.div(N, D)  # element-wise division

    return Q

def _func_03_(Q, n):
    prior = [1 / float(n)] * n # the prior distribution

    F = tf.reshape(tf.reduce_mean(Q, axis=0), (-1, n))
    U = tf.reshape(tf.constant(prior), (-1, n))
    R = tf.div(F, U)

    return R

def _func_02_(P, Q):
    P = tf.log(tf.div(Q, P))
    Q = tf.multiply(Q, P)

    L = tf.reshape(tf.reduce_sum(Q, axis=1), (-1, 1))
    L = tf.reduce_mean(L)
    # L = tf.reduce_sum(Q)

    return L

def func_04(P, Q, R):
    P = tf.log(tf.div(Q, P))
    P = tf.multiply(Q, P)

    R = tf.log(R)
    R = tf.multiply(Q, R)

    # L = tf.reduce_sum(P + R)
    L = tf.reshape(tf.reduce_sum(P + R, axis=1), (-1, 1))
    L = tf.reduce_mean(L)

    return L

if __name__ == "__main__":
    import datetime as dt
    import random

    from sklearn.metrics import normalized_mutual_info_score as NMI
    import sklearn.preprocessing as pre

    import utils

    # *****************************************************************
    name = 'depict'
    indim, outdim = 162, 4096
    # *****************************************************************

    # *****************************************************************
    X = np.loadtxt('../data/kth_xtrain_r9.txt')
    # X = pre.minmax_scale(X,(0,1), axis=1) # distribution
    # *****************************************************************

    # *****************************************************************
    m,n = X.shape
    bs = int(outdim * 4)
    nb = int(m/bs) + 1
    msg = "%s m: %d n: %d batch_size: %d num_batch: %d"%(
        dt.datetime.now(), m, n, bs, nb)
    # *****************************************************************

    # *****************************************************************
    # import os
    # cfile = os.path.join("../model", name, r"%s_centroids_r9_k%d.txt" % (name, outdim))
    # if not os.path.exists(cfile):
    #     import random
    #     from sklearn.cluster import KMeans
    #
    #     idx = random.sample([i for i in range(m)], outdim * 10)
    #     xs = X[idx, :]
    #     kms = KMeans(n_clusters=outdim).fit(xs)
    #     kys = kms.predict(X)
    #     # khs = np.histogram(kys, bins=outdim)[0]
    #     us = kms.cluster_centers_.astype(np.float32)
    #     np.savetxt(cfile, us)
    # else:
    #     us = np.loadtxt(cfile).astype(np.float32)
    # *****************************************************************

    # *****************************************************************
    W = tf.Variable(tf.random_normal([indim, outdim]))

    Z = tf.placeholder('float', [None, indim])
    P = _func_01_(Z, W)
    Q = _func_07_(P, outdim)
    R = _func_03_(Q, outdim)
    L = func_04(P, Q, R)

    cost = L
    optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

    Y = tf.argmax(P, axis=1)
    # *****************************************************************

    init = tf.global_variables_initializer()

    config = tf.ConfigProto(device_count={"CPU":24, "GPU":0})
    sess = tf.Session(config=config)
    sess.run(init)

    print(msg)
    epoch = 0
    while True:
        for i in range(nb):
            idx = random.sample([i for i in range(m)], k=bs)
            xs = X[idx,:]
            sess.run(optimizer, feed_dict={Z: xs})

            # loss = sess.run(cost, feed_dict={Z: xs})
            # log = '%s  batch: %10d  cost: %.8e nmi-1: %.8f nmi-2: %.8f nmi-3: %.8f' % (
            #     dt.datetime.now(), i, loss, 0, 0, 0)
            # print(log)

        if not epoch % 1:
            loss, dys = sess.run([cost, Y], feed_dict={Z: X})

            nmi1 = 0
            nmi2 = 0
            nmi3 = 0

            log = '%s  epoch: %10d  cost: %.8e nmi-1: %.8f nmi-2: %.8f nmi-3: %.8f' % (
                dt.datetime.now(), epoch, loss, nmi1, nmi2, nmi3)
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
