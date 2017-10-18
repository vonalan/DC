# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf 

def build_network(indim=768, outdim=10, U=None): 
    with tf.name_scope('placeholder') as scope: 
        Z = tf.placeholder(tf.float32, shape=[None, indim], name='input')
        # T = tf.placeholder(tf.int64, shape=[None, 1], name='label')

    with tf.name_scope('pair_wise_distance') as scope: 
        U = tf.Variable(U, name='centroids')

        M1 = tf.reshape(tf.reduce_sum(tf.pow(Z, 2), axis=1), (-1, 1))
        M2 = tf.reshape(tf.reduce_sum(tf.pow(U, 2), axis=1), (1, outdim))
        M3 = tf.matmul(Z, tf.transpose(U))
        M4 = tf.matmul(U, tf.transpose(Z))

        distMat = M1 + M2 - (M3 + tf.transpose(M4))

    output = tf.argmin(distMat, axis=1)
    oneHotLabels = tf.one_hot(output, depth=outdim)
    distMat = tf.multiply(distMat, ontHotLabels)
    cost = tf.reduce_sum(distMat)
    optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
    
    return dict(Z=Z, T=T, Y=output, cost=cost, optimizer=optimizer)

if __name__ == '__main__': 
    numClusters = 10 

    zs = np.loadtxt('../data/ztest.txt')
    zmax = np.max(zs, axis=0) 
    zmin = np.min(zs, axis=0)
    zrange = zmax - zmin 
    
    centroids = zmin + zrange/numClusters * \
                np.reshape(np.arange(numClusters), (-1,1))
    network = build_network(indim=zs.shape[0], outdim=10, U=centroids)