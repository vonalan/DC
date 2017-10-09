#!/usr/bin/env python
# coding=utf-8

import numpy as np 
import tensorflow as tf

def cnn():
    pass

def batch_normalization(X):
    def moments(X):
        return tf.nn.moments(X)
    def batch_normalization(X, mean, variance):
        return tf.nn.batch_normalization(X, mean, variance)
    mean, std = moments(X)
    return batch_normalization(X, mean, std)


# attention block 1
def single_attention_block(F):
    Q = tf.Variable(tf.random_normal([outdim, 1]))
    E = tf.exp(tf.matmul(F, Q))
    A = tf.div(E, tf.reduce_sum(E))
    R = tf.reduce_sum(tf.multiply(F, A), axis=0)
    return tf.reshape(R,(-1, outdim))

# attention block 2
def cascaded_attention_block(r):
    W = tf.Variable(tf.random_normal([outdim, outdim]))
    b = tf.Variable(tf.random_normal([outdim]))
    Q = tf.nn.tanh((tf.add(tf.matmul(r,W), b)))
    return Q

k=3
outdim=4

F = tf.placeholder('float', [k, outdim])
R = single_attention_block(F)
Q = cascaded_attention_block(R)

cost = tf.norm(Q)
# optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
optimizer = tf.train.RMSPropOptimizer(0.01).minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

fs = np.array([[1,1,1,1],[2,2,2,2],[3,3,3,3]])

epoch = 0
while True:
    sess.run(optimizer, feed_dict={F:fs})

    if epoch % 10000 == 0:
        # As = sess.run(R, feed_dict={F:fs})
        # print(As)
        rs = sess.run(Q, feed_dict={F:fs})
        ls = sess.run(cost, feed_dict={F: fs})
        print(rs, ls)
    epoch += 1

