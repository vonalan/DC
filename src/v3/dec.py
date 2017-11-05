# -*- coding: utf-8 -*-

# requirements: 
#   Anaconda3
#   Tensorflow

# feature works:
#     moving average
#     batch norminalization
#     RNN

import os
import sys
import math
import datetime as dt

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import graph_util
from sklearn.metrics import normalized_mutual_info_score as sklnmi


def build_network(indim=4096, outdim=6, U=None, a=1.0): 
    with tf.name_scope('placeholder') as scope: 
        Z = tf.placeholder(tf.float32, shape=[None, indim], name='input')
        T = tf.placeholder(tf.int64, shape=[None, 1], name='label')

    with tf.name_scope('func_01') as scope: 
        U = tf.Variable(U, name='centroids')

        M1 = tf.reshape(tf.reduce_sum(tf.pow(Z, 2), axis=1), (-1, 1))
        M2 = tf.reshape(tf.reduce_sum(tf.pow(U, 2), axis=1), (1, outdim))
        M3 = tf.matmul(Z, tf.transpose(U))
        M4 = tf.matmul(U, tf.transpose(Z))

        D = M1 + M2 - (M3 + tf.transpose(M4))

        N = tf.pow(D / a + 1, -((a + 1) / 2))
        D = tf.reshape(tf.reduce_sum(N, axis=1), (-1, 1))
        Q = tf.div(N, D, name='predicted_distribution')

    with tf.name_scope('func_03') as scope: 
        F = tf.reshape(tf.reduce_sum(Q, axis=0), (-1, outdim))
        N = tf.div(tf.pow(Q, 2), F)
        D = tf.reshape(tf.reduce_sum(N, axis=1), (-1, 1))
        P = tf.div(N, D, name='target_distribution')
    
    with tf.name_scope('func_02') as scope: 
        C = tf.multiply(P, tf.log(tf.div(P, Q)))
        L = tf.reshape(tf.reduce_sum(C, axis=1), (-1, 1))
        # L = tf.reduce_mean(L, name='cost')

    with tf.name_scope('metrics') as scope: 
        cost = tf.reduce_mean(L, name='cost')
        Y = tf.argmax(Q, axis=1, name='output')
        # ACC = tf.reduce_mean(tf.cast(tf.equal(Y, T), tf.float32))
        # NMI = 0.0
    optimizer = tf.train.AdamOptimizer(0.01).minimize(cost)
    
    return dict(Z=Z, Y=Y, T=T, optimizer=optimizer, cost=cost)

def train_network(graph, Z, T, batch_size, numCluster, num_epochs, pb_file_path):
    init = tf.global_variables_initializer()

    config = tf.ConfigProto(device_count={"CPU": 24, "GPU": 0})
    with tf.Session(config=config) as sess:
        sess.run(init)
        
        show_steps = 1
        # save_steps = 100
        for epoch_index in range(num_epochs):
            num_samples = Z.shape[0]
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            num_batches = int(math.ceil(num_samples / float(batch_size)))
            for batch_index in range(num_batches): 
                begin = batch_index * batch_size
                end = min((batch_index + 1) * batch_size, num_samples)
                zs, ts = Z[indices[begin:end],:], T[indices[begin:end],:]
                
                sess.run([graph['optimizer']], feed_dict={
                    graph['Z']: zs, 
                    graph['T']: ts})
            
            if not epoch_index % show_steps: 
                num_samples = Z.shape[0]
                indices = np.arange(num_samples)
                num_batches = int(math.ceil(num_samples / float(batch_size)))
                
                pys = np.zeros((0,1)) # .astype(np.float32) 
                loss = 0.0
                for batch_index in range(num_batches): 
                    begin = batch_index * batch_size
                    end = min((batch_index + 1) * batch_size, num_samples)
                    zs, ts = Z[indices[begin:end],:], T[indices[begin:end],:]
                    ys = sess.run(graph['Y'], feed_dict={
                        graph['Z']: zs, 
                        graph['T']: ts})
                    cost = sess.run(graph['cost'], feed_dict={
                        graph['Z']: zs, 
                        graph['T']: ts})
                    pys = np.vstack((pys, np.reshape(ys, (-1,1))))
                    loss += cost * (end - begin)
                nmi = sklnmi(pys.flatten(), T.flatten())                
                print("%s k: %4d e: %8d cost: %.8e nmi: %.10f"%(dt.datetime.now(), numCluster, epoch_index, loss/float(num_samples), nmi))
            
            # if not epoch_index % save_steps: 
            if not epoch_index % pow(10, len(str(epoch_index))-1): 
                x_pb_file_path = r"%s_k%d_e%d.pb"%(pb_file_path, numCluster, epoch_index)
                # if os.path.exists(x_pb_file_path): continue 
                constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ["metrics/output", "metrics/cost"])
                with tf.gfile.FastGFile(x_pb_file_path, mode='wb') as f:
                    f.write(constant_graph.SerializeToString())

def valid_network(pb_file_path, Z, T, batch_size, numCluster, epoch_index):
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()

        x_pb_file_path = r"%s_k%d_e%d.pb"%(pb_file_path, numCluster, epoch_index)
        if not os.path.exists(x_pb_file_path): return (np.zeros((0,1)), 0.0, 0.0)
        
        with open(x_pb_file_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(output_graph_def, name="")

        config = tf.ConfigProto(device_count={"CPU": 24, "GPU": 0})
        with tf.Session(config=config) as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            
            input_0 = sess.graph.get_tensor_by_name("placeholder/input:0")
            output_0 = sess.graph.get_tensor_by_name("metrics/output:0")
            cost_0 = sess.graph.get_tensor_by_name("metrics/cost:0")
            
            num_samples = Z.shape[0]
            indices = np.arange(num_samples)
            num_batches = int(math.ceil(num_samples / float(batch_size)))
            
            pys = np.zeros((0,1)) # .astype(np.float32) 
            loss = 0.0
            for batch_index in range(num_batches): 
                begin = batch_index * batch_size
                end = min((batch_index + 1) * batch_size, num_samples)
                zs, ts = Z[indices[begin:end],:], T[indices[begin:end],:]
                ys = sess.run(output_0, feed_dict={
                    input_0: zs})
                cost = sess.run(cost_0, feed_dict={
                    input_0: zs})
                pys = np.vstack((pys, np.reshape(ys, (-1,1))))
                loss += cost * (end - begin)
            nmi = sklnmi(pys.flatten(), T.flatten())        
            print("%s k: %4d e: %8d cost: %.8e nmi: %.10f"%(dt.datetime.now(), numCluster, epoch_index, loss/float(num_samples), nmi))
    return (pys, loss/float(num_samples), nmi)