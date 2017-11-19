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
from sklearn.preprocessing import minmax_scale as sklscale
from sklearn.metrics import normalized_mutual_info_score as sklnmi

import utils
import rbfnn
import depict_temp
depict = depict_temp
build_text_line_reader = depict.build_text_line_reader
build_depict_graph = depict.build_depict_graph
build_train_graph = depict.build_train_graph
build_eval_graph = depict.build_eval_graph
build_infer_graph = depict.build_infer_graph

data_type = tf.float32
input_dim = 128
output_dim = 10
EVAL_STEPS=10
INFER_STEPS=100
class flags(object):
    summaries_dir = './temp/logs/'
FLAGS = flags()
data_to_eval = True
data_to_infer = True

def inference(checkpoint_path):
    infer_graph = tf.Graph()
    with infer_graph.as_default():
        infer_filenames, infer_iterator, infer_elements = \
            build_text_line_reader(shuffle=False, batch_size=10000)
        infer_inputs, infer_outputs = build_infer_graph(infer_elements, input_dim, output_dim)
        infer_saver = tf.train.Saver()
        # infer_merger = tf.summary.merge_all()
        initializer = tf.global_variables_initializer()

    infer_sess = tf.Session(graph=infer_graph)
    infer_sess.run(initializer)
    infer_saver.restore(infer_sess, checkpoint_path)

    infer_sess.run(infer_iterator.initializer, feed_dict={infer_filenames: [r'../../data/x_1000_128.txt']})
    outputs_infer = []
    while data_to_infer:
        try:
            xs_infer = infer_sess.run(infer_elements)
        except tf.errors.OutOfRangeError:
            break
        ys_infer = infer_sess.run(infer_outputs, feed_dict={infer_inputs: xs_infer})
        outputs_infer.extend(ys_infer)
    return outputs_infer

def evaluation_rbfnn(checkpoint_path):
    # step_01 generating outputs from tensorflow
    outputs_train = inference(checkpoint_path)
    outputs_eval = inference(checkpoint_path)

    # step_02 build word vectors
    c_train = None
    c_test = None
    hist_train = utils.build_word_vector(outputs_train, c_train, bins)
    hist_test = utils.build_word_vector(outputs_eval, c_test, bins)
    xs_train = sklscale(hist_train)
    xs_test = sklscale(hist_test)

    # step_03 load
    ts_train = None
    ts_test = None

    # step_04 train and validate rbfnn
    network = rbfnn.Network()
    network.train(xs_train)

    ys_train = network.predict(xs_train)
    ys_test = network.predict(xs_test)

    err_train = utils.calc_err(ts_train, ys_train)
    acc_train = utils.calc_acc(ts_train, ys_train)

    err_test = utils.calc_err(ts_test, ys_test)
    acc_test = utils.calc_acc(ts_test, ys_test)

    stsm_train = 0.25

    return