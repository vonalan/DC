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

def read_data_from_text(filenames=None, shuffle=False):
  filenames = tf.placeholder(tf.string, shape=[None])
  dataset = tf.data.TextLineDataset(filenames)
  # dataset = dataset.map(lambda item : tf.string_split(item, " "))
  # dataset = dataset.map(lambda item : tf.string_to_number(item))
  # dataset = dataset.map(lambda item : tf.to_float(item))

  '''
  map, shuffle, batch
  '''

  if shuffle:
    pass

  iterator = dataset.make_initializable_iterator()
  next_elements = iterator.get_next()
  return filenames, iterator, next_elements

data_type = tf.float32
input_dim = 1024
output_dim = 128

def build_depict_graph(input, kernel_shape, bias_shape):
    weights = tf.get_variable("weights", kernel_shape,
        initializer=tf.random_normal_initializer())
    biases = tf.get_variable("biases", bias_shape,
        initializer=tf.constant_initializer(0.0))
    weighted_sum = tf.add(tf.matmul(input, weights), biases)
    return tf.nn.softmax(weighted_sum)

def build_train_graph(input_dim, output_dim, func=''):
    input = tf.placeholder(data_type, shape=[None, output_dim], name='train_input')
    with tf.variable_scope('depict', reuse=False):
        graph = build_depict_graph(input, [input_dim, output_dim], [output_dim])
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
    optimizer = tf.train.AdamOptimizer(0.01).minimize(cost)
    return input, optimizer

def build_eval_graph(input_dim, output_dim):
    input = tf.placeholder(data_type, shape=[None, output_dim], name='eval')
    with tf.variable_scope('depict', reuse=True):
        graph = build_depict_graph(input, [input_dim, output_dim], [output_dim])
    with tf.variable_scope('output'):
        P = graph
        output = tf.argmax(P, axis=1, name='output')
    return input, output

def build_infer_graph():
    input = tf.placeholder(data_type, shape=[1, output_dim], name='eval')
    with tf.variable_scope('depict', reuse=True):
        graph = build_depict_graph(input, [input_dim, output_dim], [output_dim])
    with tf.variable_scope('output'):
        P = graph
        output = tf.argmax(P, axis=1, name='output')
    return input, output

def main(_):
    train_graph = tf.Graph()
    eval_graph = tf.Graph()
    infer_graph = tf.Graph()

    with train_graph.as_default():
        train_filenames, train_iterator, train_elements = read_data_from_text()
        train_input, optimizer = build_train_graph(input_dim, output_dim)
        initializer = tf.global_variables_initializer()

    with eval_graph.as_default():
        eval_filenames, eval_iterator, eval_elements = read_data_from_text()
        eval_input, eval_output = build_eval_graph()

    with infer_graph.as_default():
        infer_iterator, infer_inputs = ...
        infer_model = BuildInferenceModel(infer_iterator)

    checkpoints_path = "/tmp/model/checkpoints"

    train_sess = tf.Session(graph=train_graph)
    eval_sess = tf.Session(graph=eval_graph)
    infer_sess = tf.Session(graph=infer_graph)
    #
    # train_sess.run(initializer)
    # train_sess.run(train_iterator.initializer)
    #
    # for i in itertools.count():
    #
    #   train_model.train(train_sess)
    #
    #   if i % EVAL_STEPS == 0:
    #     checkpoint_path = train_model.saver.save(train_sess, checkpoints_path, global_step=i)
    #     eval_model.saver.restore(eval_sess, checkpoint_path)
    #     eval_sess.run(eval_iterator.initializer)
    #     while data_to_eval:
    #       eval_model.eval(eval_sess)
    #
    #   if i % INFER_STEPS == 0:
    #     checkpoint_path = train_model.saver.save(train_sess, checkpoints_path, global_step=i)
    #     infer_model.saver.restore(infer_sess, checkpoint_path)
    #     infer_sess.run(infer_iterator.initializer, feed_dict={infer_inputs: infer_input_data})
    #     while data_to_infer:
    #       infer_model.infer(infer_sess)

if __name__ == "__main__":
  train_filenames, train_iterator, train_elements = read_data_from_text(shuffle=True)

  sess = tf.Session()
  sess.run(train_iterator.initializer, feed_dict={train_filenames:[r'../../data/x_1000_128.txt']})

  count = 0
  while True:
    try:
      print(sess.run(train_elements))
    except Exception:
      break

    if count > 10: break
    count += 1

  # tfstr = tf.constant(['2012 2012 2012'])
  # tfstr1 = tf.string_split(tfstr, delimiter=' ')
  # tfstr = tf.map_fn(lambda x:tf.string_to_number(x), tfstr.values)
  # # tfstr = tf.string_to_number(tfstr)
  # sess = tf.Session()
  # print(sess.run(tfstr))