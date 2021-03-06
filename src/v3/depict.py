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


def build_network(indim=4096, outdim=6, prior=None, W=None):
    with tf.name_scope('placeholder') as scope: 
        Z = tf.placeholder(tf.float32, shape=[None, indim], name='input')
        T = tf.placeholder(tf.int64, shape=[None, 1], name='label')

    with tf.name_scope('func_01') as scope: 
        W = tf.Variable(tf.random_normal([indim, outdim]), name='weights')
        P = tf.nn.softmax(tf.matmul(Z, W), name='predicted_distribution')

    with tf.name_scope('func_07') as scope: 
        N = tf.reshape(tf.reduce_sum(P, axis=0), (-1, outdim))
        N = tf.div(P, tf.pow(N, 0.5))
        D = tf.reshape(tf.reduce_sum(N, 1), (-1, 1))
        Q = tf.div(N, D, name='target_distribution')
    
    with tf.name_scope('func_03') as scope: 
        prior = [1 / float(outdim)] * outdim
        U = tf.reshape(tf.constant(prior), (-1, outdim), name='prior_distribution')
        F = tf.reshape(tf.reduce_mean(Q, axis=0), (-1, outdim), name='posterior_distribution')
    
    # with tf.name_scope('func_04') as scope: 
    #     C = tf.multiply(Q, tf.log(tf.div(Q, P)))
    #     R = tf.multiply(Q, tf.log(tf.div(F, U)))
    #     L = tf.reshape(tf.reduce_sum(C + R, axis=1), (-1, 1))
    
    with tf.name_scope('func_02') as scope: 
        C = tf.multiply(Q, tf.log(tf.div(Q, P)))
        L = tf.reshape(tf.reduce_sum(C, axis=1), (-1, 1))

    # with tf.name_scope('func_08') as scope: 
    #     C = -tf.multiply(Q, tf.log(P))
    #     L = tf.reshape(tf.reduce_sum(C, axis=1), (-1, 1))

    with tf.name_scope('metrics') as scope: 
        cost = tf.reduce_mean(L, name='cost')
        Y = tf.argmax(P, axis=1, name='output')
    optimizer = tf.train.AdamOptimizer(0.01).minimize(cost)
    
    return dict(Z=Z, Y=Y, T=T, optimizer=optimizer, cost=cost)

def train_network(graph, Z, T, batch_size, numCluster, num_epochs, pb_file_path):
    init = tf.global_variables_initializer()

    # config = tf.ConfigProto(device_count={"CPU": 24, "GPU": 0})
    with tf.Session() as sess:
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

        # config = tf.ConfigProto(device_count={"CPU": 24, "GPU": 0})
        with tf.Session() as sess:
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