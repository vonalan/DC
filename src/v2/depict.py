# -*- coding: utf-8 -*-

# requirements: 
#   Anaconda3
#   Tensorflow

import os
import random

import numpy as np 
import tensorflow as tf 


class NeuralNetwork(object):
    def __init__(self, indim=4096, outdim=6, W=None):
        self.name = 'depict'
        self.indim = indim
        self.outdim = outdim
        self.prior = [1 / float(self.outdim)] * self.outdim

        self.W = tf.Variable(tf.random_normal([self.indim, self.outdim]))

        self.Z = tf.placeholder('float', [None, self.indim])
        self.U = tf.reshape(tf.constant(self.prior), (-1, self.outdim))

        self.P = self._func_01_()
        self.Q = self._func_07_()
        self.F = tf.reshape(tf.reduce_mean(self.Q, axis=0), (-1, self.outdim))
        self.L = self._func_04_()
        self.Y = tf.argmax(self.P, axis=1)

        self.cost = self.L
        # self.optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(self.cost)
        self.optimizer = tf.train.AdamOptimizer(0.01).minimize(self.cost)

        self.init = tf.global_variables_initializer()

        self.config = tf.ConfigProto(device_count={"CPU": 24, "GPU": 0})
        self.sess = tf.Session(config=self.config)
        self.sess.run(self.init)

    def _func_01_(self):
        Z = tf.exp(tf.matmul(self.Z, self.W))
        S = tf.reshape(tf.reduce_sum(Z, 1), (-1, 1))
        return tf.div(Z, S)

        # # weighted sum
        # Z = tf.matmul(self.Z, self.W)
        #
        # # batch normalization
        # m,v = tf.nn.moments(Z, axes=0)
        # Z = tf.nn.batch_normalization(Z, m, v, offset=0.0, scale=1.0, variance_epsilon=0.001)
        #
        # # softmax activation
        # Z = tf.exp(Z)
        # S = tf.reshape(tf.reduce_sum(Z, 1), (-1, 1))
        # return tf.div(Z, S)

    def _func_07_(self):
        N = tf.reshape(tf.reduce_sum(self.P, axis=0), (-1, self.outdim))
        N = tf.div(self.P, tf.pow(N, 0.5))
        D = tf.reshape(tf.reduce_sum(N, 1), (-1, 1))
        return tf.div(N, D)

    def _func_02_(self):
        C = tf.multiply(self.Q, tf.log(tf.div(self.Q, self.P)))
        L = tf.reshape(tf.reduce_sum(C, axis=1), (-1, 1))
        return tf.reduce_mean(L)

    def _func_04_(self):
        C = tf.multiply(self.Q, tf.log(tf.div(self.Q, self.P)))
        R = tf.multiply(self.Q, tf.log(tf.div(self.F, self.U)))
        L = tf.reshape(tf.reduce_sum(C + R, axis=1), (-1, 1))
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
        saver.save(self.sess, os.path.join(dstdir, '%s_%d.ckpt'%(self.name, epoch)))

    def loadModel(self, root, epoch):
        dstdir = os.path.join(root, self.name)
        saver = tf.train.Saver()
        saver.restore(self.sess, os.path.join(dstdir, '%s_%d.ckpt'%(self.name, epoch)))

if __name__ == "__main__":
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
    # kms = utils.mini_kmeans('../model', 'kmeans', X, outdim, factor=4)
    # kys = kms.predict(X)
    # # khs = np.histogram(kys, bins=outdim)[0]
    # ws = np.dot(np.linalg.inv(X), kys)
    # # us = kms.cluster_centers_.astype(np.float32)
    # *****************************************************************

    nn = NeuralNetwork(indim=indim, outdim=outdim)

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
            loss = nn.partial_fit(xs)

            # log = '%s  batch: %10d  cost: %.8e nmi-1: %.8f nmi-2: %.8f nmi-3: %.8f' % (
            #     dt.datetime.now(), i, loss, 0, 0, 0)
            # print(log)

        if not epoch % 1:
            loss, dys = nn.predict(X)

            nmi1 = 0.5
            nmi2 = 0.5
            nmi3 = 0.5

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
