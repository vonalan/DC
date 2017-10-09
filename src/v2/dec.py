# -*- coding: utf-8 -*-

# requirements: 
#   Anaconda3
#   Tensorflow

import os
import sys
import random

import numpy as np 
import tensorflow as tf


class NeuralNetwork(object):
    def __init__(self, indim=4096, outdim=6, U=None, a=1.0):
        self.name = 'dec'
        self.indim = indim
        self.outdim = outdim
        self.a = a

        self.U = tf.Variable(U)

        self.Z = tf.placeholder('float', [None, self.indim])
        self.Q = self._func_01_()
        self.P = self._func_03_()
        self.L = self._func_02_()
        self.Y = tf.argmax(self.Q, axis=1)

        self.cost = self.L
        # self.optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(self.cost)
        self.optimizer = tf.train.MomentumOptimizer(0.01, 0.5).minimize(self.cost)

        self.init = tf.global_variables_initializer()

        self.config = tf.ConfigProto(device_count={"CPU": 24, "GPU": 0})
        self.sess = tf.Session(config=self.config)
        self.sess.run(self.init)

    def _func_01_(self):
        '''
        A noval implementaton of pair-wise distance between two vectors
        from two Arrays.

        z = Z[i,:]
        u = U[j,:]
        D[i,j] = (z-u)*(z-u).T = z*z.T + u*u.T - z*u.T - z.T*u
        '''

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

        M1 = tf.reshape(tf.reduce_sum(tf.pow(self.Z, 2), axis=1), (-1, 1))
        M2 = tf.reshape(tf.reduce_sum(tf.pow(self.U, 2), axis=1), (1, self.outdim))
        M3 = tf.matmul(self.Z, tf.transpose(self.U))
        M4 = tf.matmul(self.U, tf.transpose(self.Z))

        D = M1 + M2 - (M3 + tf.transpose(M4))

        N = tf.pow(D / self.a + 1, -((self.a + 1) / 2))
        D = tf.reshape(tf.reduce_sum(N, axis=1), (-1, 1))
        return tf.div(N, D)

    def _func_03_(self):
        F = tf.reshape(tf.reduce_sum(self.Q, axis=0), (-1, self.outdim))
        N = tf.div(tf.pow(self.Q, 2), F)
        D = tf.reshape(tf.reduce_sum(N, axis=1), (-1, 1))
        return tf.div(N, D)

    def _func_02_(self):
        C = tf.multiply(self.P, tf.log(tf.div(self.P, self.Q)))
        L = tf.reshape(tf.reduce_sum(C, axis=1), (-1, 1))
        return tf.reduce_mean(L)

    def partial_fit(self, Z):
        cost, opt = self.sess.run([self.cost, self.optimizer], feed_dict={self.Z: Z})
        return cost

    def predict(self, Z):
        cost, Y = self.sess.run([self.cost, self.Y], feed_dict={self.Z: Z})
        return cost, Y

    def saveModel(self, root, epoch):
        dstdir = os.path.join(root, self.name)
        if not os.path.exists(dstdir): os.mkdir(dstdir)
        saver = tf.train.Saver()
        saver.save(self.sess, os.path.join(dstdir, '%s_%d.ckpt' % (self.name, epoch)))

    def loadModel(self, root, epoch):
        dstdir = os.path.join(root, self.name)
        saver = tf.train.Saver()
        saver.restore(self.sess, os.path.join(dstdir, '%s_%d.ckpt' % (self.name, epoch)))

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
    kms = utils.mini_kmeans('../model', 'kmeans', X, outdim, factor=4)
    kys = kms.predict(X)
    # khs = np.histogram(kys, bins=outdim)[0]
    # ws = np.dot(np.linalg.inv(X), kys)
    us = kms.cluster_centers_.astype(np.float32)
    # *****************************************************************

    nn = NeuralNetwork(indim=indim, outdim=outdim, U=us)

    # *****************************************************************
    m, n = X.shape
    bs = int(outdim * 4)
    nb = int(m / bs) + 1
    msg = "%s m: %d n: %d batch_size: %d num_batch: %d" % (
        dt.datetime.now(), m, n, bs, nb)
    # *****************************************************************

    print(msg)
    epoch = 0
    while True:
        for i in range(nb):
            idx = random.sample([i for i in range(m)], k=bs)
            xs = X[idx, :]
            loss = nn.partial_fit(xs)

            # log = '%s  batch: %10d  cost: %.8e nmi-1: %.8f nmi-2: %.8f nmi-3: %.8f' % (
            #     dt.datetime.now(), i, loss, 0, 0, 0)
            # print(log)

        if not epoch % 1:
            loss, dys = nn.predict(X)

            nmi1 = 0.5
            nmi2 = 0.5
            nmi3 = NMI(kys, dys)

            log = '%s  epoch: %10d  cost: %.8e nmi-1: %.8f nmi-2: %.8f nmi-3: %.8f' % (
                dt.datetime.now(), epoch, loss, nmi1, nmi2, nmi3)
            utils.writeLog('../log', name, log)
            print(log)

            # dhs = np.histogram(dys, bins=outdim)[0]

            # print(khs)
            # print(dhs)

        if not epoch % 100:
            nn.saveModel('../model', epoch)

        epoch += 1

        flag = False
        if flag: break
    print("optimization is finished! ")
