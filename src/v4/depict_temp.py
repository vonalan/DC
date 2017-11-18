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

def build_depict_graph(inputs, kernel_shape, bias_shape):
    weights = tf.get_variable("weights", kernel_shape,
        initializer=tf.random_normal_initializer())
    biases = tf.get_variable("biases", bias_shape,
        initializer=tf.constant_initializer(0.0))
    weighted_sum = tf.add(tf.matmul(inputs, weights), biases)
    return tf.nn.softmax(weighted_sum)

def build_train_graph(defalut_inputs, input_dim, output_dim, func=''):
    inputs = tf.placeholder_with_default(defalut_inputs, shape=[None, input_dim], name='train_input')
    with tf.variable_scope('depict', reuse=False):
        graph = build_depict_graph(inputs, [input_dim, output_dim], [output_dim])
    with tf.variable_scope('target_distribution'):
        P = graph
        N = tf.reshape(tf.reduce_sum(P, axis=0), (-1, output_dim))
        N = tf.div(P, tf.pow(N, 0.5))
        D = tf.reshape(tf.reduce_sum(N, 1), (-1, 1))
        Q = tf.div(N, D)
    with tf.variable_scope('regularization'):
        prior = [1 / float(output_dim)] * output_dim
        U = tf.reshape(tf.constant(prior), (-1, output_dim))
        F = tf.reshape(tf.reduce_mean(Q, axis=0), (-1, output_dim))
    with tf.variable_scope('cost'):
        if func == 'func_04':
            with tf.variable_scope('func_04'):
                C = tf.multiply(Q, tf.log(tf.div(Q, P)))
                R = tf.multiply(Q, tf.log(tf.div(F, U)))
                L = tf.reshape(tf.reduce_sum(C + R, axis=1), (-1, 1))
                cost = tf.reduce_mean(L, name='cost')
        elif func == 'func_02':
            with tf.variable_scope('func_02'):
                C = tf.multiply(Q, tf.log(tf.div(Q, P)))
                L = tf.reshape(tf.reduce_sum(C, axis=1), (-1, 1))
                cost = tf.reduce_mean(L, name='cost')
        elif func == 'func_08':
            with tf.variable_scope('func_08'):
                C = -tf.multiply(Q, tf.log(P))
                L = tf.reshape(tf.reduce_sum(C, axis=1), (-1, 1))
                cost = tf.reduce_mean(L, name='cost')
        else:
            with tf.variable_scope('func_02'):
                C = tf.multiply(Q, tf.log(tf.div(Q, P)))
                L = tf.reshape(tf.reduce_sum(C, axis=1), (-1, 1))
                cost = tf.reduce_mean(L, name='cost')
    tf.summary.scalar('training_cost', cost)
    optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)
    return inputs, cost, optimizer

def main():
    train_graph = tf.Graph()
    with train_graph.as_default():
        train_filenames, train_iterator, train_elements = \
            build_text_line_reader(shuffle=True, batch_size=100)
        train_input, train_cost, optimizer = build_train_graph(train_elements, input_dim, output_dim)
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