# -*- coding: utf-8 -*-

# requirements: 
#   Anaconda3
#   Tensorflow

# feature works: 
#     moving average 
#     batch norminalization
#     RNN
#     distributions
#     vector representations of words
#         continuous bag-of-words model (CBOW)
#         Skip-Gram Model

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

import classifier

# import parser

# data_type = tf.float32
# input_dim = 128
# output_dim = 10
# EVAL_STEPS=10
# INFER_STEPS=100
# data_to_eval = True
# data_to_infer = True

FLAGS = None

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

def build_eval_graph(default_inputs, input_dim, output_dim):
    inputs = tf.placeholder_with_default(default_inputs, shape=[None, input_dim], name='eval_input')
    with tf.variable_scope('depict', reuse=False):
        graph = build_depict_graph(inputs, [input_dim, output_dim], [output_dim])
    with tf.variable_scope('eval_output'):
        P = graph
        outputs = tf.argmax(P, axis=1, name='output')
    return inputs, outputs

def build_infer_graph(default_inputs, input_dim, output_dim):
    inputs = tf.placeholder_with_default(default_inputs, shape=[None, input_dim], name='infer_inputs')
    with tf.variable_scope('depict', reuse=False):
        graph = build_depict_graph(inputs, [input_dim, output_dim], [output_dim])
    with tf.variable_scope('infer_output'):
        P = graph
        outputs = tf.argmax(P, axis=1, name='output')
    return inputs, outputs

def prepare_file_system():
  # Setup the directory we'll write summaries to for TensorBoard
  if tf.gfile.Exists(FLAGS.summaries_dir):
    tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
  tf.gfile.MakeDirs(FLAGS.summaries_dir)
  if tf.gfile.Exists(FLAGS.checkpoints_dir):
    tf.gfile.DeleteRecursively(FLAGS.checkpoints_dir)
  tf.gfile.MakeDirs(FLAGS.summaries_dir)
  return

def main():
    tf.logging.set_verbosity(tf.logging.INFO)
    prepare_file_system()

    # FLAGS.eval_step_interval = 1
    # FLAGS.infer_step_interal = 1
    # FLAGS.train_batch_size = 10000
    # FLAGS.infer_batch_size = 10000
    # FLAGS.eval_batch_size = 10000

    train_graph = tf.Graph()
    with train_graph.as_default():
        train_filenames, train_iterator, train_elements = \
            build_text_line_reader(shuffle=True, batch_size=FLAGS.train_batch_size)
        train_inputs, train_cost, optimizer = build_train_graph(
            train_elements, FLAGS.depict_input_dim, FLAGS.depict_output_dim)
        train_saver = tf.train.Saver()
        train_merger = tf.summary.merge_all()
        train_initializer = tf.global_variables_initializer()
    eval_graph = tf.Graph()
    with eval_graph.as_default():
        eval_filenames, eval_iterator, eval_elements = \
            build_text_line_reader(shuffle=True, batch_size=FLAGS.eval_batch_size)
        eval_inputs, eval_outputs = build_eval_graph(
            eval_elements, FLAGS.depict_input_dim, FLAGS.depict_output_dim)
        eval_saver = tf.train.Saver()
        eval_merger = tf.summary.merge_all()
        eval_initializer = tf.global_variables_initializer()
    infer_graph = tf.Graph()
    with infer_graph.as_default():
        infer_filenames, infer_iterator, infer_elements = \
            build_text_line_reader(shuffle=False, batch_size=FLAGS.infer_batch_size)
        infer_inputs, infer_outputs = build_infer_graph(
            infer_elements, FLAGS.depict_input_dim, FLAGS.depict_output_dim)
        infer_saver = tf.train.Saver()
        infer_merger = tf.summary.merge_all()
        infer_initializer = tf.global_variables_initializer()

    train_sess = tf.Session(graph=train_graph)
    eval_sess = tf.Session(graph=eval_graph)
    infer_sess = tf.Session(graph=infer_graph)

    train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train', train_graph)
    validation_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/validation', eval_graph)
    infer_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/inference', infer_graph)

    train_sess.run(train_initializer)
    train_sess.run(train_iterator.initializer, feed_dict={train_filenames: [FLAGS.path_to_xtrain]})
    # train_sess.run(train_iterator.initializer)
    for i in itertools.count():
        try:
            xs_train = train_sess.run(train_elements)
        except tf.errors.OutOfRangeError:
            train_sess.run(train_iterator.initializer, feed_dict={train_filenames: [FLAGS.path_to_xtrain]})
            xs_train = train_sess.run(train_elements)
        # train_summary, _ = train_sess.run([optimizer, train_merger]) #
        _, training_cost, train_summary = train_sess.run([optimizer, train_cost, train_merger],
                                                         feed_dict={train_inputs: xs_train})
        train_writer.add_summary(train_summary, i)
        # print('epoch: %6d, training cost: %.8f'%(i, training_cost))
        # time.sleep(1)

        if i % FLAGS.eval_step_interval == 0:
            checkpoint_path = train_saver.save(train_sess, FLAGS.checkpoints_dir, global_step=i)
            eval_saver.restore(eval_sess, checkpoint_path)
            eval_sess.run(eval_iterator.initializer, feed_dict={eval_filenames: [FLAGS.path_to_xtest]})
            while FLAGS.data_to_eval:
                try:
                    xs_eval = eval_sess.run(eval_elements)
                except tf.errors.OutOfRangeError:
                    # eval_sess.run(eval_iterator.initializer,
                    #                feed_dict={eval_filenames: [r'../../data/x_1000_128.txt']})
                    # xs_eval = eval_sess.run(eval_elements)
                    break
                # training_outputs = eval_sess.run(eval_outputs, feed_dict={eval_inputs: xs_train})
                # evaluation_outputs = eval_sess.run(eval_outputs, feed_dict={eval_inputs: xs_eval})
                evaluation_cost, eval_summary = train_sess.run([train_cost, train_merger], feed_dict={train_inputs: xs_eval})
                tf.logging.info("epoch: %d, training cost: %f"%(i, training_cost))
                tf.logging.info("epoch: %d, evaluation cost: %f" % (i, evaluation_cost))
                validation_writer.add_summary(eval_summary, i)
                break

        if i % FLAGS.infer_step_interval == 0:
            checkpoint_path = train_saver.save(train_sess, FLAGS.checkpoints_dir, global_step=i)
            infer_saver.restore(infer_sess, checkpoint_path)

            infers_train = []
            infer_sess.run(infer_iterator.initializer, feed_dict={infer_filenames: [FLAGS.path_to_xtest]})
            while FLAGS.data_to_infer:
                try:
                    xs_infer = infer_sess.run(infer_elements)
                except tf.errors.OutOfRangeError:
                    break
                ys_infer = infer_sess.run(infer_outputs, feed_dict={infer_inputs: xs_infer})
                infers_train.extend(ys_infer)
            print(infers_train)

            infers_test = []
            infer_sess.run(infer_iterator.initializer, feed_dict={infer_filenames: [FLAGS.path_to_xtrain]})
            while FLAGS.data_to_infer:
                try:
                    xs_infer = infer_sess.run(infer_elements)
                except tf.errors.OutOfRangeError:
                    break
                ys_infer = infer_sess.run(infer_outputs, feed_dict={infer_inputs: xs_infer})
                infers_test.extend(ys_infer)
            print(infers_test)
            metrics = classifier.run(infers_train, infers_test, FLAGS)
            print(metrics)

if __name__ == "__main__":
    # train_filenames, train_iterator, train_elements = \
    #     build_text_line_reader(shuffle=True, batch_size=20000)
    # train_sess = tf.Session()
    # train_sess.run(train_iterator.initializer, feed_dict={train_filenames:[r'../../data/x_1000_128.txt']})
    # # train_sess.run(train_iterator.initializer)
    # for i in range(30):
    #     try:
    #         print(i, train_sess.run(train_elements).shape)
    #     except tf.errors.OutOfRangeError:
    #         train_sess.run(train_iterator.initializer, feed_dict={train_filenames: [r'../../data/x_1000_128.txt']})
    #         print(i, train_sess.run(train_elements).shape)
    #         # raise Exception()
    #     time.sleep(1)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--depict_input_dim',
        type=int,
        default=162
    )
    parser.add_argument(
        '--depict_output_dim',
        type=int,
        default=128
    )
    parser.add_argument(
        '--path_to_ctrain',
        type=str,
        default=r'D:\Users\kingdom\GIT\DC_OLD\data\kth_ctrain_r9.txt'
    )
    parser.add_argument(
        '--path_to_xtrain',
        type=str,
        default=r'D:\Users\kingdom\GIT\DC_OLD\data\kth_xtrain_r9.txt'
    )
    parser.add_argument(
        '--path_to_ytrain',
        type=str,
        default=r'D:\Users\kingdom\GIT\DC_OLD\data\kth_ytrain_r9.txt'
    )
    parser.add_argument(
        '--path_to_ctest',
        type=str,
        default=r'D:\Users\kingdom\GIT\DC_OLD\data\kth_ctest_r9.txt'
    )
    parser.add_argument(
        '--path_to_xtest',
        type=str,
        default=r'D:\Users\kingdom\GIT\DC_OLD\data\kth_xtest_r9.txt'
    )
    parser.add_argument(
        '--path_to_ytest',
        type=str,
        default=r'D:\Users\kingdom\GIT\DC_OLD\data\kth_ytest_r9.txt'
    )
    parser.add_argument(
        '--summaries_dir',
        type=str,
        default='../../temp/logs',
        help='Where to save summary logs for TensorBoard.'
    )
    parser.add_argument(
        '--checkpoints_dir',
        type=str,
        default='../../temp/model/checkpoints',
        help='Where to save summary logs for TensorBoard.'
    )
    parser.add_argument(
        '--how_many_training_steps',
        type=int,
        default=40000,
        help='How many training steps to run before ending.'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.01,
        help='How large a learning rate to use when training.'
    )
    parser.add_argument(
        '--eval_step_interval',
        type=int,
        default=1,
        help='How often to evaluate the training results.'
    )
    parser.add_argument(
        '--infer_step_interval',
        type=int,
        default=1,
        help='How often to evaluate the training results.'
    )
    parser.add_argument(
        '--train_batch_size',
        type=int,
        default=10000,
        help='How many images to train on at a time.'
    )
    parser.add_argument(
        '--infer_batch_size',
        type=int,
        default=10000,  # 1 for attention, -1 for others
        help='How many images to test on at a time.'
    )
    parser.add_argument(
        '--eval_batch_size',
        type=int,
        default=100000,
        help='How many images to use in an evaluation batch.'
    )
    parser.add_argument(
        '--data_to_eval',
        type=bool,
        default=True
    )
    parser.add_argument(
        '--data_to_infer',
        type=bool,
        default=True
    )
    parser.add_argument(
        '--rbfnn_num_center',
        type=int,
        default=90
    )
    parser.add_argument(
        '--rbfnn_output_dim',
        type=int,
        default=6
    )
    FLAGS, unparsed = parser.parse_known_args()
    main()