# -*- coding: utf-8 -*-

# requirements: 
#   Python 3
#   Numpy 
#   Tensorflow

import numpy as np 
import tensorflow as tf 

indim = 28 * 28
outdim = 10
weights = tf.Variable(tf.random_normal([indim, outdim]))

def softmax(X, weights):
    Z = tf.matmul(X, weights)
    Z = tf.exp(Z)
    S = tf.reshape(tf.reduce_sum(Z,1),(-1,1))
    Z = tf.div(Z,S)
    return Z

def softmin(X):
    ''''''
    _N1 = tf.reshape(tf.reduce_sum(X, 1), (-1, 1))  # for every sample
    _N2 = tf.sqrt(_N1)  # not pow
    N = tf.div(X, _N2)  # element-division
    ''''''
    D = tf.reduce_sum(N, 0)
    T = tf.divide(N, D)
    return T

X = tf.placeholder('float', [None, indim])
# Y1 = tf.nn.softmax(tf.matmul(X, weights))
Y2 = softmax(X, weights)
T = softmin(Y2)

cost = - tf.reduce_sum(tf.multiply(T, tf.log(Y1)))
optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

xs = np.random.random((5,28*28))

print(sess.run(Y1, feed_dict={X:xs}).flatten())
print(sess.run(Y2, feed_dict={X:xs}).flatten())
# print(sess.run(_N1, feed_dict={X:xs}).flatten())
# print(sess.run(_N2, feed_dict={X:xs}).flatten())
# # print(sess.run(_N3, feed_dict={X:xs}).flatten())
# print(sess.run(N, feed_dict={X:xs}).flatten())
# print(sess.run(D, feed_dict={X:xs}).flatten())
# print(sess.run(T, feed_dict={X:xs}).flatten())
# print(sess.run(cost, feed_dict={X:xs}).flatten())



