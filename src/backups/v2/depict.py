# -*- coding: utf-8 -*-

# requirements: 
#   Anaconda3
#   Tensorflow


import numpy as np 
import tensorflow as tf 

def build_network(indim, outdim): 
    Z = tf.placeholder('float', [None, indim], name='input')

    with tf.name_scope('func_01') as scope: 
        W = tf.Variable(tf.random_normal([indim, outdim]), name='weights')
        P = tf.nn.softmax(tf.matmul(Z, W), name=scope)

    with tf.name_scope('func_07') as scope: 
        N = tf.reshape(tf.reduce_sum(P, axis=0), (-1, outdim))
        N = tf.div(P, tf.pow(N, 0.5))
        D = tf.reshape(tf.reduce_sum(N, 1), (-1, 1))
        Q = tf.div(N, D)
    
    with tf.name_scope('func_03') as scope: 
        prior = [1 / float(outdim)] * outdim
        U = tf.reshape(tf.constant(prior), (-1, outdim))
        F = tf.reshape(tf.reduce_mean(Q, axis=0), (-1, outdim))
    
    with tf.name_scope('func_04') as scope: 
        C = tf.multiply(Q, tf.log(tf.div(Q, P)))
        R = tf.multiply(Q, tf.log(tf.div(F, U)))
        L = tf.reshape(tf.reduce_sum(C + R, axis=1), (-1, 1))
        L = tf.reduce_mean(L)
    
    # with tf.name_scope('func_02') as scope: 
    #     C = tf.multiply(Q, tf.log(tf.div(Q, P)))
    #     L = tf.reshape(tf.reduce_sum(C, axis=1), (-1, 1))
    #     L = tf.reduce_mean(L)
    optimizer = tf.train.AdamOptimizer(0.01).minimize(cost)
    
    return dict(Z=Z, optimizer=optimizer, cost=L)

def train_network(): 
    pass 

def valid_network(): 
    pass 

def main(): 
    pass 