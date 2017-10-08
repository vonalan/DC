# -*- coding: utf-8 -*-

import os
import sys

import tensorflow as tf

def writeLog(root, name, string):
    if not os.path.exists(root): os.mkdir(root)
    file = r'%s/%s.log'%(root, name)
    wf = open(file, 'a')
    wf.write(string)
    wf.write('\n')
    wf.close()

def saveModel(sess, root, name, epoch):
    dstdir = os.path.join(root, name)
    if not os.path.exists(dstdir): os.mkdir(dstdir)
    saver = tf.train.Saver()
    saver.save(sess, os.path.join(dstdir, '%s_%d.ckpt' % (name, epoch)))

def loadModel(root, name, epoch):
    dstdir = os.path.join(root, name)
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.save(sess, os.path.join(dstdir, '%s_%d.ckpt' % (name, epoch)))
    return sess

if __name__ == "__main__":
    import numpy as np
    import tensorflow as tf

    a = 1.0

    # z = Z[i,:]
    # u = U[j,:]
    # d = D[i,j] = (z-u)*(z-u).T = z*z.T + u*u.T - z*u.T - z.T*u
    Z = tf.placeholder('float', [4, 5])
    U = tf.placeholder('float', [3, 5])

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

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    zs = [[1,2,3,4,5],[2,3,4,5,6],[3,4,5,6,7],[4,5,6,7,8]]
    us = [[2,3,4,5,6],[3,4,5,6,7],[4,5,6,7,8]]
    print(sess.run(Q, feed_dict={Z:zs,U:us}))
