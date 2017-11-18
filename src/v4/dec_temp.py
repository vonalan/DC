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
import time
import datetime as dt
import itertools

import numpy as np
import tensorflow as tf 
from tensorflow.python.framework import graph_util
from sklearn.metrics import normalized_mutual_info_score as sklnmi
from sklearn.cluster import KMeans


data_type = tf.float32
input_dim = 128
output_dim = 10
EVAL_STEPS=10
INFER_STEPS=100
class flags(object):
    summaries_dir = './temp/logs/'
FLAGS = flags()
data_to_eval = True

def build_text_line_reader(filenames=None, shuffle=False, batch_size=1):
    filenames = tf.placeholder(tf.string, shape=[None])
    dataset = tf.data.TextLineDataset(filenames)
    dataset = dataset.map(lambda string: tf.string_split([string]).values)
    dataset = dataset.map(lambda string: tf.string_to_number(string, out_type=tf.float32))
    if shuffle: dataset = dataset.shuffle(10000)
    if batch_size > 1: dataset = dataset.batch(batch_size)
    iterator = dataset.make_initializable_iterator()
    next_elements = iterator.get_next()
    return filenames, iterator, next_elements

def build_dec_graph(inputs, us, output_dim, a=1):
    # U = tf.Variable(U, name='centroids')
    centroids = tf.get_variable('centroids', dtype=data_type,
                                initializer=tf.constant(us))
    # an implementation of sklearn.metrics.pairwise.pairwise_distances() with tensorflow
    # dist(x[i], u[j]) = (x[i] - u[j])(x[i] - u[j]).T
    #                  = x[i]x[i].T - x[i]u[j].T - u[j]x[i].T + u[j]u[j].T
    #                  = M1 - M3 - M4 + M2
    # or
    # (a - b) ** 2 = a ** 2 - 2ab + b ** 2

    M1 = tf.reshape(tf.reduce_sum(tf.pow(inputs, 2), axis=1), (-1, 1))
    M2 = tf.reshape(tf.reduce_sum(tf.pow(centroids, 2), axis=1), (1, output_dim))
    M3 = tf.matmul(inputs, tf.transpose(centroids))
    M4 = tf.matmul(centroids, tf.transpose(inputs))

    D = M1 + M2 - (M3 + tf.transpose(M4)) # or
    # D = M1 + M2 - 2 * M3

    N = tf.pow(D / a + 1, -((a + 1) / 2))
    D = tf.reshape(tf.reduce_sum(N, axis=1), (-1, 1))
    return tf.div(N, D)

def build_train_graph(defalut_inputs, us, input_dim, output_dim):
    inputs = tf.placeholder_with_default(defalut_inputs, shape=[None, input_dim], name='train_input')
    with tf.variable_scope('dec', reuse=False):
        graph = build_dec_graph(inputs, us, output_dim)
    with tf.variable_scope('target_distribution'):
        Q = graph
        F = tf.reshape(tf.reduce_sum(Q, axis=0), (-1, output_dim))
        N = tf.div(tf.pow(Q, 2), F)
        D = tf.reshape(tf.reduce_sum(N, axis=1), (-1, 1))
        P = tf.div(N, D, name='target_distribution')
    with tf.variable_scope('cost'):
        with tf.variable_scope('func_02'):
            C = tf.multiply(P, tf.log(tf.div(P, Q)))
            L = tf.reshape(tf.reduce_sum(C, axis=1), (-1, 1))
            cost = tf.reduce_mean(L, name='cost')
    tf.summary.scalar('training_cost', cost)
    optimizer = tf.train.AdamOptimizer(0.01).minimize(cost)
    return inputs, cost, optimizer

def main():
    if(not os.path.exists('../../data/x_1000_128_kmeans_10.txt')):
        kms_centroids = KMeans(n_clusters=output_dim). \
            fit(np.loadtxt('../../data/x_1000_128.txt')). \
            cluster_centers_.astype(np.float32)
        np.savetxt('../../data/x_1000_128_kmeans_10.txt', kms_centroids)
    else:
        kms_centroids = np.loadtxt('../../data/x_1000_128_kmeans_10.txt').astype(np.float32)

    train_graph = tf.Graph()
    with train_graph.as_default():
        train_filenames, train_iterator, train_elements = \
            build_text_line_reader(shuffle=True, batch_size=100)
        train_input, train_cost, optimizer = build_train_graph(train_elements, kms_centroids, input_dim, output_dim)
        # dec_centroids, train_input, optimizer = build_train_graph(train_elements, input_dim, output_dim)
        # assign_centroids = tf.assign(dec_centroids, kms_centroids)
        initializer = tf.global_variables_initializer()
        train_saver = tf.train.Saver()
        train_merger = tf.summary.merge_all()
        # train_sess = tf.Session(config=tf.ConfigProto(device_count={'gpu':0}))
        train_sess = tf.Session()

    checkpoints_path = "/tmp/model/checkpoints"
    # merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train', train_graph)
    validation_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/validation')

    train_sess.run(initializer)
    train_sess.run(train_iterator.initializer, feed_dict={train_filenames: [r'../../data/x_1000_128.txt']})
    # train_sess.run(train_iterator.initializer)
    for i in itertools.count():
        try:
            xs = train_sess.run(train_elements)
        except tf.errors.OutOfRangeError:
            train_sess.run(train_iterator.initializer, feed_dict={train_filenames: [r'../../data/x_1000_128.txt']})
            xs = train_sess.run(train_elements)
        # print(i, xs.shape)
        # train_summary, _ = train_sess.run([optimizer, train_merger]) #
        _, training_cost, train_summary = train_sess.run([optimizer, train_cost, train_merger], feed_dict={train_input: xs})
        train_writer.add_summary(train_summary, i)
        print('epoch: %6d, training cost: %.8f'%(i, training_cost))
        # time.sleep(1)

if __name__ == "__main__":
    # train_filenames, train_iterator, train_elements = \
    #     build_text_line_reader(shuffle=True, batch_size=100)
    # train_sess = tf.Session()
    # train_sess.run(train_iterator.initializer, feed_dict={train_filenames:[r'../../data/x_1000_128.txt']})
    # # train_sess.run(train_iterator.initializer)
    # for i in range(30):
    #     try:
    #         print(i, train_sess.run(train_elements)[0,:4])
    #     except tf.errors.OutOfRangeError:
    #         train_sess.run(train_iterator.initializer, feed_dict={train_filenames: [r'../../data/x_1000_128.txt']})
    #         print(i, train_sess.run(train_elements)[0, :4])
    #         # raise Exception()
    #     time.sleep(1)
    main()