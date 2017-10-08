# -*- coding: utf-8 -*-

# requirements: 
#   Anaconda3
#   Tensorflow

import numpy as np 
import tensorflow as tf 


class NeuralNetwork(object):
    def __init__(self, indim=4096, outdim=6):
        self.indim = indim
        self.outdim = outdim

        self.weights = tf.Variable(tf.random_normal([self.indim, self.outdim]))

        self.X = tf.placeholder('float', [None, self.indim])
        self.Y = self._func_01_()
        self.T = self._func_07_()
        self.Q = tf.log(tf.div(self.Y, self.T))

        self.f = tf.reshape(tf.reduce_mean(self.T, 0), (-1, self.outdim))
        self.u = tf.reshape(tf.constant([1/float(self.outdim)] * self.outdim), (-1, self.outdim)) # how to
        self.r = tf.log(tf.div(self.f, self.u))

        self.KL = tf.multiply(self.T, self.Q)
        self.R = tf.multiply(self.T, self.r)
        self.O = self.KL + self.R

        self.cost = tf.reduce_mean(tf.reduce_sum(self.O, 1))
        self.optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(self.cost)

        self.P = tf.argmax(self.Y, axis=1)

        self.init = tf.global_variables_initializer()
        # self.sess = tf.Session()
        self.sess = tf.Session(config = tf.ConfigProto(device_count={'GPU': 0}))
        self.sess.run(self.init)

    def _func_01_(self):
        '''
        A simple implementation of softmax function.
        Be careful of overflow problem!

        :param X:
        :param weights:
        :return:
        '''

        Z = tf.matmul(self.X, self.weights)
        Z = tf.exp(Z)
        S = tf.reshape(tf.reduce_sum(Z, 1), (-1, 1))
        A = tf.div(Z, S)

        return A

    def _func_07_(self):
        '''
        A simple implementation of target function that is like softmax function.

        :param X:
        :return:
        '''

        Z = tf.reshape(tf.reduce_sum(self.Y, 0), (-1, self.outdim))  # for every cluster
        H = tf.pow(Z, 0.5)

        N = tf.div(self.Y, H)  # element-wise division
        D = tf.reshape(tf.reduce_sum(N, 1), (-1, 1))  # for every sample

        T = tf.div(N, D)  # element-wise division

        return T

    def partial_fit(self, X):
        cost, opt = self.sess.run([self.cost, self.optimizer], feed_dict={self.X: X})
        return cost

    def predict(self, X):
        cost, opt = self.sess.run([self.cost, self.P], feed_dict={self.X: X})
        return cost, opt

    def saveModel(self, dstdir, epoch):
        import os
        if not os.path.exists(dstdir): os.mkdir(dstdir)
        saver = tf.train.Saver()
        saver.save(self.sess, os.path.join(dstdir, 'depic_%d.ckpt'%epoch))

    def loadModel(self, dstdir, epoch):
        import os
        saver = tf.train.Saver()
        saver.restore(self.sess, os.path.join(dstdir, 'depict_%d.ckpt'%epoch))


if __name__ == "__main__":
    import sklearn.preprocessing as pre
    import datetime as dt

    from sklearn.metrics import normalized_mutual_info_score as NMI
    from sklearn.cluster import KMeans

    import utils

    indim = 4096
    outdim = 6

    # xs = np.random.random((1000,28*28))
    X = np.loadtxt('../data/kth_X_4096.txt') # histogram
    X = pre.minmax_scale(X,(0,1), axis=1) # distribution
    # us = np.array([[1/float(outdim)]*outdim]) # uniform distribution

    Y = np.loadtxt('../data/kth_Y_6.txt')
    oys = np.argmax(Y, axis=1)
    # delta = np.array([0]*Y.shape[0])

    kms = KMeans(n_clusters=outdim)
    nn = NeuralNetwork(indim=4096, outdim=6)

    epoch = 0
    while True:
        nn.partial_fit(X)
        if not epoch % 1000:
            cost, pys = nn.predict(X)
            kys = np.argmin(kms.fit_transform(X), axis=1)

            nmi1 = NMI(oys, pys)
            nmi2 = NMI(oys, kys)
            nmi3 = NMI(kys, pys)

            # abs = np.sum(oys - delta)
            # delta = pys

            log = '%s  epoch: %10d  cost: %.4e nmi-1: %.8f nmi-2: %.8f nmi-3: %.8f'%(dt.datetime.now(), epoch, cost, nmi1, nmi2, nmi3)
            logFile = r'../data/log.txt'
            utils.writeLog(logFile, log)
            print(log)
        epoch += 1

        flag = False
        if flag: break


