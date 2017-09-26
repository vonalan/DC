# -*- coding: utf-8 -*-

# requirements: 
#   Python 3
#   Numpy 
#   Tensorflow

import numpy as np 
import tensorflow as tf 

indim = 4096
outdim = 6
weights = tf.Variable(tf.random_normal([indim, outdim]))

def softmax(X, weights):
    Z = tf.matmul(X, weights)
    Z = tf.exp(Z)
    S = tf.reshape(tf.reduce_sum(Z,1),(-1,1))
    Z = tf.div(Z,S)
    return Z

def softmin(X):
    ''''''
    _N1 = tf.reshape(tf.reduce_sum(X, 0), (-1,outdim))  # for every cluster
    _N2 = tf.pow(_N1, 0.5)
    N = tf.div(X, _N2)  # element-wise division
    ''''''
    D = tf.reshape(tf.reduce_sum(N, 1), (-1,1)) # for every sample
    T = tf.div(N, D) # element-wise division
    return T

X = tf.placeholder('float', [None, indim])
Y = softmax(X, weights)
T = softmin(Y)
Q = tf.log(tf.div(Y, T))

F = tf.reshape(tf.reduce_mean(T, 0), (-1, outdim))
# U = tf.random_uniform((1, outdim))
U = tf.placeholder('float', [1, outdim])
R = tf.log(tf.div(F, U))

term1 = tf.multiply(T, Q)
term2 = tf.multiply(T, R)
loss = term1 + term2
cost = tf.reduce_mean(tf.reduce_sum(loss, 1))

# cost = - tf.reduce_mean(tf.reduce_sum(tf.multiply(T, tf.log(Y)), 1))
optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

import sklearn.preprocessing as pre
import datetime as dt

# xs = np.random.random((1000,28*28))
xs = np.loadtxt('../data/kth.txt')
xs = pre.minmax_scale(xs,(0,1))
us = np.array([[1/float(outdim)]*outdim])
# print(sess.run(U, feed_dict={U:us}).flatten())

count = 0
while True:
    sess.run(optimizer, feed_dict={X: xs, U: us})
    if not count % 1000:
        # print('%12d'%count, sess.run(cost, feed_dict={X: xs, U: us}).flatten())
        mycost = sess.run(cost, feed_dict={X: xs, U: us})
        log = '%s  count: %10d, cost: %.8e'%(dt.datetime.now(), count, mycost)
        wf = open('../data/log.txt', 'a')
        wf.write(log)
        wf.write('\n')
        wf.close()
        print(log)
    count += 1

    flag = False
    if flag: break

# print(sess.run(Y, feed_dict={X:xs}).flatten())
# print(sess.run(T, feed_dict={X:xs}).flatten())
# print(sess.run(_N1, feed_dict={X:xs}).flatten())
# print(sess.run(_N2, feed_dict={X:xs}).flatten())
# # print(sess.run(_N3, feed_dict={X:xs}).flatten())
# print(sess.run(N, feed_dict={X:xs}).flatten())
# print(sess.run(D, feed_dict={X:xs}).flatten())
# print(sess.run(T, feed_dict={X:xs}).flatten())

